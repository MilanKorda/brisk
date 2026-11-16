/* qpstage.c - the stage-wise factorization of the QP interior-point method's system (the QP developer, 14 November 2026).
 *
 * The system of an iteration (qpipm.c) is  K (dx; dy) = (b_x; b_y),
 *
 *     K = [ -(H + diag(d))   A'      ]      d: the bounds' terms and the regularization of the columns,
 *         [  A               diag(e) ]      e: the regularization of an equality row, 1 / Ds of an inequality row.
 *
 * The variables are sorted into levels: breadth-first from an end of the graph whose edges join two variables that share
 * a row of A or an entry of H. Entries then join only variables of the same level or of two neighbouring levels, so
 * every row lies in one level or in two neighbouring ones. A model predictive control problem written with its states
 * gives levels of about one stage; a banded problem gives narrow levels. Nothing is assumed about which variables are
 * states or inputs, how the dynamics are written, or whether the stages repeat.
 *
 * The inequality rows are eliminated first (dy_i = (b_i - a_i dx) / e_i), which adds a_i' a_i / e_i to H + diag(d): the
 * matrix Phi, block tridiagonal over the levels, diagonal blocks Phi_k and coupling blocks Psi_k (level k + 1 against
 * level k). An equality row belongs to the lowest level k it touches: its entries there are a row of E_k, its entries in
 * level k + 1 a row of F_k. With x_k the unknowns of level k and y_k the multipliers of block k's equality rows:
 *
 *     -Phi_k x_k - Psi_{k-1} x_{k-1} - Psi_k' x_{k+1} + E_k' y_k + F_{k-1}' y_{k-1} = a_k
 *      E_k x_k + F_k x_{k+1} + delta y_k = b_k
 *
 * Eliminated in the order x_0, y_0, x_1, y_1, ... on dense blocks (LAPACK's Cholesky factorization and level-3 BLAS):
 *
 *     Phi~_k = L_k L_k'              X_k = E_k L_k^-T          P_k = Psi_k L_k^-T
 *     S_k = delta I + X_k X_k' = M_k M_k'                      F~_k = F_k - X_k P_k'          Y_k = M_k^-1 F~_k
 *     Phi~_{k+1} = Phi_{k+1} - P_k P_k' + Y_k' Y_k
 *
 * K is quasi-definite (Vanderbei 1995), so every Phi~_k and S_k is positive definite in exact arithmetic whatever the
 * order of the levels. In stage order this is the block elimination behind the Riccati recursion of model predictive
 * control (Rao, Wright and Rawlings 1998; Wang and Boyd 2010), written for the system as given. Work and memory grow
 * linearly with the number of levels; there is no symbolic analysis and no fill outside the blocks. Only the columns of
 * level k + 1 that block k touches (the coupling columns, sorted first in their level) enter F~_k, Y_k and the update. */
#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include <string.h>
#include <math.h>
#include "qpstage.h"
#ifdef _OPENMP
#include <omp.h>
#else
static inline int omp_in_parallel(void) { return 0; }
#endif

#ifndef BLAS_PREFIX
#define BLAS_PREFIX
#endif
#define ST_CAT2(a, b) a##b
#define ST_CAT(a, b) ST_CAT2(a, b)
void ST_CAT(BLAS_PREFIX, dpotrf_)(const char *, const int *, double *, const int *, int *);
void ST_CAT(BLAS_PREFIX, dtrsm_)(const char *, const char *, const char *, const char *, const int *, const int *, const double *, const double *, const int *, double *, const int *);
void ST_CAT(BLAS_PREFIX, dsyrk_)(const char *, const char *, const int *, const int *, const double *, const double *, const int *, const double *, double *, const int *);
void ST_CAT(BLAS_PREFIX, dgemm_)(const char *, const char *, const int *, const int *, const int *, const double *, const double *, const int *, const double *, const int *, const double *, double *, const int *);
void ST_CAT(BLAS_PREFIX, dgemv_)(const char *, const int *, const int *, const double *, const double *, const int *, const double *, const int *, const double *, double *, const int *);
void ST_CAT(BLAS_PREFIX, dtrsv_)(const char *, const char *, const char *, const int *, const double *, const int *, double *, const int *);
#define st_dpotrf ST_CAT(BLAS_PREFIX, dpotrf_)
#define st_dtrsm ST_CAT(BLAS_PREFIX, dtrsm_)
#define st_dsyrk ST_CAT(BLAS_PREFIX, dsyrk_)
#define st_dgemm ST_CAT(BLAS_PREFIX, dgemm_)
#define st_dgemv ST_CAT(BLAS_PREFIX, dgemv_)
#define st_dtrsv ST_CAT(BLAS_PREFIX, dtrsv_)

/* The dense kernels. Below ST_SMALL rows and columns they are plain loops over columns (a library call costs more than
 * the arithmetic of a small block); above, LAPACK and level-3 BLAS. Lower triangles, column-major storage. */
#ifndef ST_SMALL
#define ST_SMALL 16
#endif
/* a = a - (c0 t0 + c1 t1 + c2 t2 + c3 t3) over m entries: four columns applied to a target column in one pass, so that
 * the target is read and written once for four multiply-adds (the passes of the blocked kernels below). With AVX-512
 * the passes are written out: eight entries a step with fused multiply-adds and the last step under a mask, so that a
 * column of 5 or of 60 entries runs at the vector rate (the compiler's own loop finishes each column with up to seven
 * scalar steps, most of the work of a block of 16 to 64 rows). */
#if defined(__AVX512F__) && !defined(BRISK_QPSTAGE_NOSIMD)
#include <immintrin.h>
static inline void sk_sub4(int m, double *a, const double *c0, const double *c1, const double *c2, const double *c3, double t0, double t1, double t2, double t3) {
    const __m512d v0 = _mm512_set1_pd(t0), v1 = _mm512_set1_pd(t1), v2 = _mm512_set1_pd(t2), v3 = _mm512_set1_pd(t3);
    int i = 0;
    for (; i + 8 <= m; i += 8) {
        __m512d x = _mm512_loadu_pd(a + i);
        x = _mm512_fnmadd_pd(_mm512_loadu_pd(c0 + i), v0, x); x = _mm512_fnmadd_pd(_mm512_loadu_pd(c1 + i), v1, x);
        x = _mm512_fnmadd_pd(_mm512_loadu_pd(c2 + i), v2, x); x = _mm512_fnmadd_pd(_mm512_loadu_pd(c3 + i), v3, x);
        _mm512_storeu_pd(a + i, x);
    }
    /* (15 November 2026) the last entries by steps of four, two and one, not under a mask: a masked store cannot be
     * forwarded to the next pass's load of the same entries, and every pass of a block whose size is not a multiple of
     * eight waited for it (B L^-T on 36 rows: 7 times the time of 24 rows for 3.4 times the operations). Each entry
     * gets the same four fused multiply-adds in the same order: the same bits. */
    if (i + 4 <= m) {
        __m256d x = _mm256_loadu_pd(a + i);
        x = _mm256_fnmadd_pd(_mm256_loadu_pd(c0 + i), _mm512_castpd512_pd256(v0), x); x = _mm256_fnmadd_pd(_mm256_loadu_pd(c1 + i), _mm512_castpd512_pd256(v1), x);
        x = _mm256_fnmadd_pd(_mm256_loadu_pd(c2 + i), _mm512_castpd512_pd256(v2), x); x = _mm256_fnmadd_pd(_mm256_loadu_pd(c3 + i), _mm512_castpd512_pd256(v3), x);
        _mm256_storeu_pd(a + i, x); i += 4;
    }
    if (i + 2 <= m) {
        __m128d x = _mm_loadu_pd(a + i);
        x = _mm_fnmadd_pd(_mm_loadu_pd(c0 + i), _mm512_castpd512_pd128(v0), x); x = _mm_fnmadd_pd(_mm_loadu_pd(c1 + i), _mm512_castpd512_pd128(v1), x);
        x = _mm_fnmadd_pd(_mm_loadu_pd(c2 + i), _mm512_castpd512_pd128(v2), x); x = _mm_fnmadd_pd(_mm_loadu_pd(c3 + i), _mm512_castpd512_pd128(v3), x);
        _mm_storeu_pd(a + i, x); i += 2;
    }
    if (i < m) {
        __m128d x = _mm_load_sd(a + i);
        x = _mm_fnmadd_sd(_mm_load_sd(c0 + i), _mm512_castpd512_pd128(v0), x); x = _mm_fnmadd_sd(_mm_load_sd(c1 + i), _mm512_castpd512_pd128(v1), x);
        x = _mm_fnmadd_sd(_mm_load_sd(c2 + i), _mm512_castpd512_pd128(v2), x); x = _mm_fnmadd_sd(_mm_load_sd(c3 + i), _mm512_castpd512_pd128(v3), x);
        _mm_store_sd(a + i, x);
    }
}
/* a += c0 t0 (sk_add1) and a -= c0 t0 (sk_sub1) */
static inline void sk_add1(int m, double *a, const double *c0, double t0) {
    const __m512d v0 = _mm512_set1_pd(t0); int i = 0;
    for (; i + 8 <= m; i += 8) _mm512_storeu_pd(a + i, _mm512_fmadd_pd(_mm512_loadu_pd(c0 + i), v0, _mm512_loadu_pd(a + i)));
    if (i + 4 <= m) { _mm256_storeu_pd(a + i, _mm256_fmadd_pd(_mm256_loadu_pd(c0 + i), _mm512_castpd512_pd256(v0), _mm256_loadu_pd(a + i))); i += 4; }
    if (i + 2 <= m) { _mm_storeu_pd(a + i, _mm_fmadd_pd(_mm_loadu_pd(c0 + i), _mm512_castpd512_pd128(v0), _mm_loadu_pd(a + i))); i += 2; }
    if (i < m) _mm_store_sd(a + i, _mm_fmadd_sd(_mm_load_sd(c0 + i), _mm512_castpd512_pd128(v0), _mm_load_sd(a + i)));
}
static inline void sk_sub1(int m, double *a, const double *c0, double t0) { sk_add1(m, a, c0, -t0); }
/* the sum of a_i b_i over m entries */
static inline double sk_dot(int m, const double *a, const double *b) {
    __m512d s = _mm512_setzero_pd(); int i = 0;
    for (; i + 8 <= m; i += 8) s = _mm512_fmadd_pd(_mm512_loadu_pd(a + i), _mm512_loadu_pd(b + i), s);
    if (i < m) { const __mmask8 k = (__mmask8)((1u << (m - i)) - 1u); s = _mm512_fmadd_pd(_mm512_maskz_loadu_pd(k, a + i), _mm512_maskz_loadu_pd(k, b + i), s); }
    return _mm512_reduce_add_pd(s);
}
/* d_q = the sum of c_q,i x_i over m entries, q = 0..3: four columns against one vector, the vector read once */
static inline void sk_dot4(int m, const double *c0, const double *c1, const double *c2, const double *c3, const double *x, double *d) {
    __m512d s0 = _mm512_setzero_pd(), s1 = s0, s2 = s0, s3 = s0; int i = 0;
    for (; i + 8 <= m; i += 8) { const __m512d xv = _mm512_loadu_pd(x + i);
        s0 = _mm512_fmadd_pd(_mm512_loadu_pd(c0 + i), xv, s0); s1 = _mm512_fmadd_pd(_mm512_loadu_pd(c1 + i), xv, s1);
        s2 = _mm512_fmadd_pd(_mm512_loadu_pd(c2 + i), xv, s2); s3 = _mm512_fmadd_pd(_mm512_loadu_pd(c3 + i), xv, s3); }
    if (i < m) { const __mmask8 k = (__mmask8)((1u << (m - i)) - 1u); const __m512d xv = _mm512_maskz_loadu_pd(k, x + i);
        s0 = _mm512_fmadd_pd(_mm512_maskz_loadu_pd(k, c0 + i), xv, s0); s1 = _mm512_fmadd_pd(_mm512_maskz_loadu_pd(k, c1 + i), xv, s1);
        s2 = _mm512_fmadd_pd(_mm512_maskz_loadu_pd(k, c2 + i), xv, s2); s3 = _mm512_fmadd_pd(_mm512_maskz_loadu_pd(k, c3 + i), xv, s3); }
    d[0] = _mm512_reduce_add_pd(s0); d[1] = _mm512_reduce_add_pd(s1); d[2] = _mm512_reduce_add_pd(s2); d[3] = _mm512_reduce_add_pd(s3);
}
/* a := a t */
static inline void sk_scal(int m, double *a, double t) {
    const __m512d v = _mm512_set1_pd(t); int i = 0;
    for (; i + 8 <= m; i += 8) _mm512_storeu_pd(a + i, _mm512_mul_pd(_mm512_loadu_pd(a + i), v));
    for (; i < m; i++) a[i] *= t;
}
#else
static inline void sk_sub4(int m, double *restrict a, const double *restrict c0, const double *restrict c1, const double *restrict c2, const double *restrict c3,
                           double t0, double t1, double t2, double t3) {
    for (int i = 0; i < m; i++) a[i] -= (c0[i] * t0 + c1[i] * t1) + (c2[i] * t2 + c3[i] * t3);
}
static inline void sk_sub1(int m, double *restrict a, const double *restrict c0, double t0) { for (int i = 0; i < m; i++) a[i] -= c0[i] * t0; }
static inline void sk_add1(int m, double *restrict a, const double *restrict c0, double t0) { for (int i = 0; i < m; i++) a[i] += c0[i] * t0; }
static inline double sk_dot(int m, const double *a, const double *b) { double s = 0; for (int i = 0; i < m; i++) s += a[i] * b[i]; return s; }
static inline void sk_dot4(int m, const double *c0, const double *c1, const double *c2, const double *c3, const double *x, double *d) {
    double s0 = 0, s1 = 0, s2 = 0, s3 = 0;
    for (int i = 0; i < m; i++) { const double xi = x[i]; s0 += c0[i] * xi; s1 += c1[i] * xi; s2 += c2[i] * xi; s3 += c3[i] * xi; }
    d[0] = s0; d[1] = s1; d[2] = s2; d[3] = s3;
}
static inline void sk_scal(int m, double *a, double t) { for (int i = 0; i < m; i++) a[i] *= t; }
#endif
/* The solves' kernels (a triangular solve with one right-hand side, a product of a block with a vector) are plain loops
 * below ST_L2 rows, four columns at a time: at these sizes the arithmetic of one call is a few hundred vector operations
 * and a library call's fixed cost is of that order. */
#ifndef ST_L2
#if defined(__AVX512F__) && !defined(BRISK_QPSTAGE_NOSIMD)
#define ST_L2 256
#else
#define ST_L2 ST_SMALL
#endif
#endif
/* (15 November 2026) Where the three steps of a level's elimination go to the library, with the AVX-512 passes: the
 * Cholesky factorization from 128 rows, B L^-T from 96, the symmetric update from 64 (they went at 16). One thread of
 * this host, a level of n rows, the three steps by the loops against LAPACK and OpenBLAS: n = 16: 1.06 against 1.53
 * microseconds; 24: 1.9 against 3.5; 36: 6.1 against 10.0; 54: 16.6 against 26.6; 81: 58 against 67 (the update alone:
 * 21 against 18); 121: 219 against 196 (the factorization alone still 30 against 44). */
#if defined(__AVX512F__) && !defined(BRISK_QPSTAGE_NOSIMD)
#define ST_POTRF 128
#define ST_TRSM 96
#define ST_SYRK 64
#else
#define ST_POTRF ST_SMALL
#define ST_TRSM ST_SMALL
#define ST_SYRK ST_SMALL
#endif
/* the Cholesky factorization of the lower triangle in place, four columns at a time; d0 (may be NULL): the diagonal before
 * any elimination, for the tiny-pivot rule (a pivot at or below 1e-16 of it becomes 1e128); without d0, returns 1 at a
 * pivot that is not positive */
static int sk_potrf_blk(int n, double *A, int ld, const double *d0) {
    for (int j0 = 0; j0 < n; j0 += 4) {
        const int nb = n - j0 < 4 ? n - j0 : 4;
        for (int jj = j0; jj < j0 + nb; jj++) {              /* the panel's columns, the panel's earlier columns applied */
            double *aj = A + (size_t)jj * ld;
            for (int p = j0; p < jj; p++) { const double *ap = A + (size_t)p * ld; sk_sub1(n - jj, aj + jj, ap + jj, ap[jj]); }
            double d = aj[jj];
            if (d0) { if (!(d > 1e-16 * d0[jj]) || !(d < HUGE_VAL)) d = 1e128; }
            else if (!(d > 0) || !(d < HUGE_VAL)) return 1;
            d = sqrt(d); aj[jj] = d; sk_scal(n - jj - 1, aj + jj + 1, 1.0 / d);
        }
        const int k0 = j0 + nb; if (k0 >= n) break;
        const double *c0 = A + (size_t)j0 * ld;
        if (nb == 4) { const double *c1 = c0 + ld, *c2 = c1 + ld, *c3 = c2 + ld;
            for (int k = k0; k < n; k++) sk_sub4(n - k, A + (size_t)k * ld + k, c0 + k, c1 + k, c2 + k, c3 + k, c0[k], c1[k], c2[k], c3[k]); }
        else for (int p = 0; p < nb; p++) { const double *cp = c0 + (size_t)p * ld; for (int k = k0; k < n; k++) sk_sub1(n - k, A + (size_t)k * ld + k, cp + k, cp[k]); }
    }
    return 0;
}
static int sk_potrf(int n, double *A, int ld) {
    if (n >= ST_SMALL) { int info = 0; st_dpotrf("L", &n, A, &ld, &info); return info != 0; }
    return sk_potrf_blk(n, A, ld, NULL);
}
/* B (m x n) := B L^-T, L (n x n) lower: column j from the columns before it, four at a time */
static void sk_trsm_rlt(int m, int n, const double *Lm, int ldl, double *B, int ldb) {
    if (m >= ST_TRSM || n >= ST_TRSM) { const double one = 1.0; st_dtrsm("R", "L", "T", "N", &m, &n, &one, Lm, &ldl, B, &ldb); return; }
    for (int j = 0; j < n; j++) {
        double *bj = B + (size_t)j * ldb; int k = 0;
        for (; k + 4 <= j; k += 4) { const double *b0 = B + (size_t)k * ldb;
            sk_sub4(m, bj, b0, b0 + ldb, b0 + 2 * (size_t)ldb, b0 + 3 * (size_t)ldb, Lm[(size_t)k * ldl + j], Lm[(size_t)(k + 1) * ldl + j], Lm[(size_t)(k + 2) * ldl + j], Lm[(size_t)(k + 3) * ldl + j]); }
        for (; k < j; k++) sk_sub1(m, bj, B + (size_t)k * ldb, Lm[(size_t)k * ldl + j]);
        sk_scal(m, bj, 1.0 / Lm[(size_t)j * ldl + j]);
    }
}
/* B (m x n) := M^-1 B, M (m x m) lower */
static void sk_trsm_lln(int m, int n, const double *M, int ldm, double *B, int ldb) {
    if (m >= ST_SMALL || n >= ST_SMALL) { const double one = 1.0; st_dtrsm("L", "L", "N", "N", &m, &n, &one, M, &ldm, B, &ldb); return; }
    for (int j = 0; j < n; j++) {
        double *b = B + (size_t)j * ldb;
        for (int k = 0; k < m; k++) { const double *mk = M + (size_t)k * ldm; const double t = b[k] / mk[k]; b[k] = t; sk_sub1(m - k - 1, b + k + 1, mk + k + 1, t); }
    }
}
/* C (n x n, lower) := alpha A A' + beta C, A (n x k); beta 0 or 1; alpha -1: four columns of A at a time */
static void sk_syrk_ln(int n, int k, double alpha, const double *A, int lda, double beta, double *C, int ldc) {
    if (n >= ST_SYRK || k >= ST_SYRK) { st_dsyrk("L", "N", &n, &k, &alpha, A, &lda, &beta, C, &ldc); return; }
    if (beta == 0.0) for (int j = 0; j < n; j++) for (int i = j; i < n; i++) C[(size_t)j * ldc + i] = 0.0;
    int p = 0;
    if (alpha == -1.0) for (; p + 4 <= k; p += 4) { const double *a0 = A + (size_t)p * lda, *a1 = a0 + lda, *a2 = a1 + lda, *a3 = a2 + lda;
        for (int j = 0; j < n; j++) sk_sub4(n - j, C + (size_t)j * ldc + j, a0 + j, a1 + j, a2 + j, a3 + j, a0[j], a1[j], a2[j], a3[j]); }
    for (; p < k; p++) { const double *ap = A + (size_t)p * lda;
        for (int j = 0; j < n; j++) sk_add1(n - j, C + (size_t)j * ldc + j, ap + j, alpha * ap[j]); }
}
/* The two level-3 steps of a wide level on the threads (ST_PAR rows or more): B := B L^-T by fixed chunks of B's rows,
 * C -= B B' by fixed blocks of C's columns (a block's diagonal tile by dsyrk, its rows below by dgemm). The chunks do not
 * depend on the number of threads and each is computed by one library call on one thread, so the bits are the same at
 * any thread count; the library itself runs on one thread during a solve. */
#ifndef ST_PAR
#define ST_PAR 160
#endif
/* (15 November 2026) lb: B's entries (i, j) with i > j + lb are zero (a coupling block of the dynamics x+ = F x + G u is
 * F's pattern: lb = 0 for a diagonal or upper triangular F, 1 for a tridiagonal one; at least m - 1: no such structure).
 * B L^-T has the same zeros (L^-T is upper triangular), so the chunk of rows from i0 on is a solve with the trailing
 * triangle of L from the column i0 - lb on, and the entries left of it are never touched: with lb small, a third of the
 * arithmetic. The same chunks at any thread count. */
static int g_st_ch[4] = { 0, 0, 0, 0 };      /* (a test, BRISK_STCH=a,b,c,d: the rows of a solve's chunk, the columns of an update's block, a panel's columns and its chunks' rows) */
static void st_ch_init(void) { if (g_st_ch[0]) return; int a = 48, b = 64, c = 96, d = 48; const char *e = getenv("BRISK_STCH"); if (e) sscanf(e, "%d,%d,%d,%d", &a, &b, &c, &d); g_st_ch[1] = b; g_st_ch[2] = c; g_st_ch[3] = d; g_st_ch[0] = a; }
static void sk_trsm_rlt_par(int m, int n, const double *Lm, int ldl, double *B, int ldb, int lb) {
    st_ch_init();
    const int ch = g_st_ch[0], nch = (m + ch - 1) / ch;
    if (n < ST_PAR || nch < 2) { sk_trsm_rlt(m, n, Lm, ldl, B, ldb); return; }
    if (omp_in_parallel()) {        /* (one end of the chain inside the two ends' region: the chunks as tasks for the idle threads) */
        for (int t = 0; t < nch; t++) {
            #pragma omp task firstprivate(t) shared(Lm, B)
            { const int i0 = t * ch, mm = m - i0 < ch ? m - i0 : ch, c0 = i0 > lb ? (i0 - lb < n ? i0 - lb : n) : 0, nn = n - c0; const double one = 1.0;
              if (nn > 0) st_dtrsm("R", "L", "T", "N", &mm, &nn, &one, Lm + (size_t)c0 * ((size_t)ldl + 1), &ldl, B + (size_t)c0 * ldb + i0, &ldb); }
        }
        #pragma omp taskwait
        return;
    }
    #pragma omp parallel for schedule(dynamic, 1)
    for (int t = 0; t < nch; t++) {
        const int i0 = t * ch, mm = m - i0 < ch ? m - i0 : ch, c0 = i0 > lb ? (i0 - lb < n ? i0 - lb : n) : 0, nn = n - c0; const double one = 1.0;
        if (nn > 0) st_dtrsm("R", "L", "T", "N", &mm, &nn, &one, Lm + (size_t)c0 * ((size_t)ldl + 1), &ldl, B + (size_t)c0 * ldb + i0, &ldb);
    }
}
/* one block of C's columns (from j0, nb of them) of C -= A A', A (n x k) with A's entries (i, j), i > j + lb, zero: its
 * diagonal tile from the columns j0 - lb on, and each chunk of the rows below from its own first nonzero column on */
static void sk_syrk_blk(int t, int ch, int n, int k, const double *A, int lda, double *C, int ldc, int lb) {
    const int j0 = t * ch, nb = n - j0 < ch ? n - j0 : ch, mr = n - j0 - nb; const double one = 1.0, mone = -1.0;
    const int c0 = j0 > lb ? (j0 - lb < k ? j0 - lb : k) : 0, kk = k - c0;
    if (kk > 0) st_dsyrk("L", "N", &nb, &kk, &mone, A + (size_t)c0 * lda + j0, &lda, &one, C + (size_t)j0 * ldc + j0, &ldc);
    if (mr <= 0) return;
    if (lb >= n - 1) { st_dgemm("N", "T", &mr, &nb, &k, &mone, A + j0 + nb, &lda, A + j0, &lda, &one, C + (size_t)j0 * ldc + j0 + nb, &ldc); return; }
    for (int i0 = j0 + nb; i0 < n; i0 += ch) {
        const int ms = n - i0 < ch ? n - i0 : ch, cs = i0 > lb ? (i0 - lb < k ? i0 - lb : k) : 0, ks = k - cs;
        if (ks > 0) st_dgemm("N", "T", &ms, &nb, &ks, &mone, A + (size_t)cs * lda + i0, &lda, A + (size_t)cs * lda + j0, &lda, &one, C + (size_t)j0 * ldc + i0, &ldc);
    }
}
static void sk_syrk_ln_sub_par(int n, int k, const double *A, int lda, double *C, int ldc, int lb) {      /* C -= A A' */
    st_ch_init();
    const int ch = g_st_ch[1], nch = (n + ch - 1) / ch;
    if (k < ST_PAR || nch < 2) { sk_syrk_ln(n, k, -1.0, A, lda, 1.0, C, ldc); return; }
    if (omp_in_parallel()) {
        for (int t = 0; t < nch; t++) {
            #pragma omp task firstprivate(t) shared(A, C)
            sk_syrk_blk(t, ch, n, k, A, lda, C, ldc, lb);
        }
        #pragma omp taskwait
        return;
    }
    #pragma omp parallel for schedule(dynamic, 1)
    for (int t = 0; t < nch; t++) sk_syrk_blk(t, ch, n, k, A, lda, C, ldc, lb);
}
/* (15 November 2026) The Cholesky factorization of a wide diagonal block (ST_PAR rows or more) on the threads: panels of
 * 96 columns, each factored by one library call, the rows below it solved by fixed chunks of 48 rows and the trailing
 * block updated by fixed blocks of 64 columns (sk_syrk_blk), the chunks on the team as the two level-3 steps above. With
 * the couplings banded this factorization, a sixth of a level's arithmetic before, was most of its time, on one thread
 * while the others waited (a level of 384 rows: 0.6 ms of 1.4). The same tiles at any thread count. Returns LAPACK's
 * info (not 0: a pivot that is not positive; the caller then takes the loop with the tiny-pivot rule). */
static int sk_potrf_par(int n, double *A, int lda) {
    st_ch_init();
    const int pb = g_st_ch[2], ch = g_st_ch[3], cb = g_st_ch[1];
    static int tiled = -1; if (tiled < 0) { const char *e = getenv("BRISK_STPOTRF"); tiled = !(e && atoi(e) == 0); }      /* (BRISK_STPOTRF=0: one library call, as before) */
    if (n < ST_PAR || !tiled) { int info = 0; st_dpotrf("L", &n, A, &lda, &info); return info; }
    const int inpar = omp_in_parallel();
    for (int k0 = 0; k0 < n; k0 += pb) {
        const int kb = n - k0 < pb ? n - k0 : pb, m2 = n - k0 - kb; int kbb = kb;
        double *Akk = A + (size_t)k0 * lda + k0;
        int info = 0; st_dpotrf("L", &kbb, Akk, &lda, &info);
        if (info) return info + k0;
        if (m2 <= 0) break;
        double *P = Akk + kb, *A22 = A + (size_t)(k0 + kb) * lda + k0 + kb;
        const int nch = (m2 + ch - 1) / ch, ncb = (m2 + cb - 1) / cb;
        if (inpar) {
            for (int t = 0; t < nch; t++) {
                #pragma omp task firstprivate(t) shared(Akk, P)
                { const int i0 = t * ch, mm = m2 - i0 < ch ? m2 - i0 : ch; int kb2 = kb; const double one = 1.0;
                  st_dtrsm("R", "L", "T", "N", &mm, &kb2, &one, Akk, &lda, P + i0, &lda); }
            }
            #pragma omp taskwait
            for (int t = 0; t < ncb; t++) {
                #pragma omp task firstprivate(t) shared(P, A22)
                sk_syrk_blk(t, cb, m2, kb, P, lda, A22, lda, 1 << 30);
            }
            #pragma omp taskwait
        } else {
            #pragma omp parallel for schedule(dynamic, 1) if (nch > 1)
            for (int t = 0; t < nch; t++) { const int i0 = t * ch, mm = m2 - i0 < ch ? m2 - i0 : ch; int kb2 = kb; const double one = 1.0;
                st_dtrsm("R", "L", "T", "N", &mm, &kb2, &one, Akk, &lda, P + i0, &lda); }
            #pragma omp parallel for schedule(dynamic, 1) if (ncb > 1)
            for (int t = 0; t < ncb; t++) sk_syrk_blk(t, cb, m2, kb, P, lda, A22, lda, 1 << 30);
        }
    }
    return 0;
}
/* C (n x n, lower) += A' A, A (k x n) */
static void sk_syrk_lt(int n, int k, const double *A, int lda, double *C, int ldc) {
    if (n >= ST_SMALL || k >= ST_SMALL) { const double one = 1.0; st_dsyrk("L", "T", &n, &k, &one, A, &lda, &one, C, &ldc); return; }
    for (int j = 0; j < n; j++) { const double *aj = A + (size_t)j * lda; double *cj = C + (size_t)j * ldc;
        for (int i = j; i < n; i++) cj[i] += sk_dot(k, A + (size_t)i * lda, aj); }
}
/* C (m x n) -= A (m x k) B' (B: n x k) */
static void sk_gemm_nt(int m, int n, int k, const double *A, int lda, const double *B, int ldb, double *C, int ldc) {
    if (m >= ST_SMALL || n >= ST_SMALL || k >= ST_SMALL) { const double one = 1.0, mone = -1.0; st_dgemm("N", "T", &m, &n, &k, &mone, A, &lda, B, &ldb, &one, C, &ldc); return; }
    for (int j = 0; j < n; j++) { double *cj = C + (size_t)j * ldc;
        for (int p = 0; p < k; p++) sk_sub1(m, cj, A + (size_t)p * lda, B[(size_t)p * ldb + j]); }
}
/* x := L^-1 x and x := L^-T x */
static void sk_trsv_n(int n, const double *Lm, int ld, double *x) {
    if (n >= ST_L2) { const int i1 = 1; st_dtrsv("L", "N", "N", &n, Lm, &ld, x, &i1); return; }
    int k = 0;
    for (; k + 4 <= n; k += 4) {                 /* four unknowns from the 4 x 4 triangle, then their columns applied below */
        const double *c0 = Lm + (size_t)k * ld, *c1 = c0 + ld, *c2 = c1 + ld, *c3 = c2 + ld;
        const double t0 = x[k] / c0[k];
        const double t1 = (x[k + 1] - c0[k + 1] * t0) / c1[k + 1];
        const double t2 = (x[k + 2] - c0[k + 2] * t0 - c1[k + 2] * t1) / c2[k + 2];
        const double t3 = (x[k + 3] - c0[k + 3] * t0 - c1[k + 3] * t1 - c2[k + 3] * t2) / c3[k + 3];
        x[k] = t0; x[k + 1] = t1; x[k + 2] = t2; x[k + 3] = t3;
        if (n - k - 4 > 0) sk_sub4(n - k - 4, x + k + 4, c0 + k + 4, c1 + k + 4, c2 + k + 4, c3 + k + 4, t0, t1, t2, t3);
    }
    for (; k < n; k++) { const double *lk = Lm + (size_t)k * ld; const double t = x[k] / lk[k]; x[k] = t; sk_sub1(n - k - 1, x + k + 1, lk + k + 1, t); }
}
static void sk_trsv_t(int n, const double *Lm, int ld, double *x) {
    if (n >= ST_L2) { const int i1 = 1; st_dtrsv("L", "T", "N", &n, Lm, &ld, x, &i1); return; }
    int k = n - 1;
    for (; k >= (n & ~3); k--) { const double *lk = Lm + (size_t)k * ld; x[k] = (x[k] - sk_dot(n - k - 1, lk + k + 1, x + k + 1)) / lk[k]; }
    for (k = (n & ~3) - 4; k >= 0; k -= 4) {      /* four columns against the unknowns below them, then the 4 x 4 triangle */
        const double *c0 = Lm + (size_t)k * ld, *c1 = c0 + ld, *c2 = c1 + ld, *c3 = c2 + ld; double d[4] = { 0, 0, 0, 0 };
        if (n - k - 4 > 0) sk_dot4(n - k - 4, c0 + k + 4, c1 + k + 4, c2 + k + 4, c3 + k + 4, x + k + 4, d);
        const double t3 = (x[k + 3] - d[3]) / c3[k + 3];
        const double t2 = (x[k + 2] - d[2] - c2[k + 3] * t3) / c2[k + 2];
        const double t1 = (x[k + 1] - d[1] - c1[k + 2] * t2 - c1[k + 3] * t3) / c1[k + 1];
        const double t0 = (x[k] - d[0] - c0[k + 1] * t1 - c0[k + 2] * t2 - c0[k + 3] * t3) / c0[k];
        x[k] = t0; x[k + 1] = t1; x[k + 2] = t2; x[k + 3] = t3;
    }
}
/* y += alpha A x (A: m x n) and y += alpha A' x */
static void sk_gemv_n(int m, int n, double alpha, const double *A, int lda, const double *x, double *y) {
    if (m >= ST_L2 || n >= ST_L2) { const int i1 = 1; const double one = 1.0; st_dgemv("N", &m, &n, &alpha, A, &lda, x, &i1, &one, y, &i1); return; }
    int j = 0;
    for (; j + 4 <= n; j += 4) { const double *a0 = A + (size_t)j * lda;
        sk_sub4(m, y, a0, a0 + lda, a0 + 2 * (size_t)lda, a0 + 3 * (size_t)lda, -alpha * x[j], -alpha * x[j + 1], -alpha * x[j + 2], -alpha * x[j + 3]); }
    for (; j < n; j++) sk_add1(m, y, A + (size_t)j * lda, alpha * x[j]);
}
static void sk_gemv_t(int m, int n, double alpha, const double *A, int lda, const double *x, double *y) {
    if (m >= ST_L2 || n >= ST_L2) { const int i1 = 1; const double one = 1.0; st_dgemv("T", &m, &n, &alpha, A, &lda, x, &i1, &one, y, &i1); return; }
    int j = 0;
    for (; j + 4 <= n; j += 4) { const double *a0 = A + (size_t)j * lda; double d[4];
        sk_dot4(m, a0, a0 + lda, a0 + 2 * (size_t)lda, a0 + 3 * (size_t)lda, x, d);
        y[j] += alpha * d[0]; y[j + 1] += alpha * d[1]; y[j + 2] += alpha * d[2]; y[j + 3] += alpha * d[3]; }
    for (; j < n; j++) y[j] += alpha * sk_dot(m, A + (size_t)j * lda, x);
}

/* the equilibration's sweep over a run of neighbouring rows of one column (qpipm.c): a_i := a_i (fj fr_i) when fr is
 * given, then rmax_i = max(rmax_i, |a_i|); returns max(cmax, the largest |a_i|). The products and the maxima are those of
 * the entry-by-entry loop (a maximum does not depend on the order): the same bits. */
double qp_vscale_max(int m, double *a, double fj, const double *fr, double *rmax, double cmax) {
#if defined(__AVX512F__) && !defined(BRISK_QPSTAGE_NOSIMD)
    const __m512d vf = _mm512_set1_pd(fj); __m512d vc = _mm512_set1_pd(cmax); int i = 0;
    for (; i + 8 <= m; i += 8) {
        __m512d x = _mm512_loadu_pd(a + i);
        if (fr) { x = _mm512_mul_pd(x, _mm512_mul_pd(vf, _mm512_loadu_pd(fr + i))); _mm512_storeu_pd(a + i, x); }
        x = _mm512_abs_pd(x);
        _mm512_storeu_pd(rmax + i, _mm512_max_pd(_mm512_loadu_pd(rmax + i), x)); vc = _mm512_max_pd(vc, x);
    }
    cmax = _mm512_reduce_max_pd(vc);
    for (; i < m; i++) { if (fr) a[i] *= fj * fr[i]; const double v = fabs(a[i]); if (v > rmax[i]) rmax[i] = v; if (v > cmax) cmax = v; }
    return cmax;
#else
    for (int i = 0; i < m; i++) { if (fr) a[i] *= fj * fr[i]; const double v = fabs(a[i]); if (v > rmax[i]) rmax[i] = v; if (v > cmax) cmax = v; }
    return cmax;
#endif
}
/* the vector kernels for the interior-point method's products with A over runs of neighbouring rows (qpipm.c) */
double qp_vdot(int m, const double *a, const double *b) { return sk_dot(m, a, b); }
void qp_vaxpy(int m, double *a, const double *c, double t) { sk_add1(m, a, c, t); }

struct QpStage {
    int n, m, L;                /* variables, rows, levels */
    int *lev, *pos;             /* a variable's level and its place in the level */
    int *vst;                   /* L + 1: the levels' starts in the stage order of the variables */
    int *vord;                  /* n: the variables in stage order */
    int *rblk, *rpos;           /* an equality row's block and its place in it (-1 for an inequality row) */
    int *rst, *rord;            /* L + 1: the blocks' starts in the order of the equality rows; the rows in that order */
    int *cc;                    /* L: the coupling columns of level k + 1 against level k (the first cc[k] of level k + 1) */
    char *psi;                  /* L: level k + 1 is coupled to level k through Phi (an entry of H or an inequality row) */
    size_t *oPhi, *oE, *oF, *oPsi, *oS;   /* the blocks' places in val */
    size_t nval; double *val;
    size_t *hH, *hA;            /* the place of each entry of H and of each entry of A in an equality row */
    int ni, *irow, *ip, *ij, *iq;   /* the inequality rows: their indices, starts, columns and positions in A */
    double *xs, *ys, *wi;       /* the solve's work: n, the equality rows, the inequality rows' weights */
    double wfact, wsolve; int wmax;
};

static void *st_alloc(size_t s) { return malloc(s ? s : 1); }
#include <time.h>
static double st_now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + 1e-9 * t.tv_nsec; }

void qp_stage_free(QpStage *S) {
    if (!S) return;
    free(S->lev); free(S->pos); free(S->vst); free(S->vord); free(S->rblk); free(S->rpos); free(S->rst); free(S->rord);
    free(S->cc); free(S->psi); free(S->oPhi); free(S->oE); free(S->oF); free(S->oPsi); free(S->oS); free(S->val);
    free(S->hH); free(S->hA); free(S->irow); free(S->ip); free(S->ij); free(S->iq); free(S->xs); free(S->ys); free(S->wi);
    free(S);
}
int qp_stage_levels(const QpStage *S) { return S->L; }
int qp_stage_widest(const QpStage *S) { return S->wmax; }
double qp_stage_fact_work(const QpStage *S) { return S->wfact; }
double qp_stage_solve_work(const QpStage *S) { return S->wsolve; }
size_t qp_stage_entries(const QpStage *S) { return S->nval; }

/* breadth-first levels of the variables from the variable s0 over the rows (rp, rj: A by rows) and H (hp, hj: both
 * triangles by columns); returns the number of levels of s0's component, numbered from l0; q: the queue (n) */
static int st_bfs(int s0, int l0, const int *Ap, const int *Ai, const int *rp, const int *rj, const int *hp, const int *hj,
                  int *lev, char *rseen, int *q, int *qn_out) {
    int qh = 0, qn = 0; lev[s0] = l0; q[qn++] = s0; int last = l0;
    while (qh < qn) {
        const int v = q[qh++], l = lev[v]; last = l;
        for (int p = Ap[v]; p < Ap[v + 1]; p++) {
            const int r = Ai[p]; if (rseen[r]) continue;
            rseen[r] = 1;
            for (int t = rp[r]; t < rp[r + 1]; t++) { const int w = rj[t]; if (lev[w] < 0) { lev[w] = l + 1; q[qn++] = w; } }
        }
        if (hj) for (int t = hp[v]; t < hp[v + 1]; t++) { const int w = hj[t]; if (lev[w] < 0) { lev[w] = l + 1; q[qn++] = w; } }
    }
    *qn_out = qn;
    return last - l0 + 1;
}

QpStage *qp_stage_build(int n, int m, const int *Hp, const int *Hi, const int *Ap, const int *Ai, const char *eq, size_t cap) {
    if (n <= 0) return NULL;
    QpStage *S = calloc(1, sizeof *S); if (!S) return NULL;
    S->n = n; S->m = m;
    const size_t na = (size_t)Ap[n], nh = Hp ? (size_t)Hp[n] : 0;
    int *rp = calloc((size_t)m + 2, sizeof(int)), *rj = st_alloc(sizeof(int) * (na + 1)), *rq = st_alloc(sizeof(int) * (na + 1));
    int *hp = calloc((size_t)n + 2, sizeof(int)), *hj = st_alloc(sizeof(int) * (2 * nh + 1));
    int *q = st_alloc(sizeof(int) * ((size_t)n + 1)), *lv2 = st_alloc(sizeof(int) * ((size_t)n + 1));
    char *rseen = st_alloc((size_t)m + 1);
    S->lev = st_alloc(sizeof(int) * ((size_t)n + 1)); S->pos = st_alloc(sizeof(int) * ((size_t)n + 1)); S->vord = st_alloc(sizeof(int) * ((size_t)n + 1));
    S->rblk = st_alloc(sizeof(int) * ((size_t)m + 1)); S->rpos = st_alloc(sizeof(int) * ((size_t)m + 1)); S->rord = st_alloc(sizeof(int) * ((size_t)m + 1));
    int *cnt = NULL; char *csup = NULL;
    if (!rp || !rj || !rq || !hp || !hj || !q || !lv2 || !rseen || !S->lev || !S->pos || !S->vord || !S->rblk || !S->rpos || !S->rord) goto fail;
    /* A by rows, with each entry's place in A; H's two triangles by columns, the diagonal left out */
    for (int j = 0; j < n; j++) for (int p = Ap[j]; p < Ap[j + 1]; p++) rp[Ai[p] + 2]++;
    for (int i = 0; i < m; i++) rp[i + 2] += rp[i + 1];
    for (int j = 0; j < n; j++) for (int p = Ap[j]; p < Ap[j + 1]; p++) { const int t = rp[Ai[p] + 1]++; rj[t] = j; rq[t] = p; }
    for (int j = 0; j < n && Hp; j++) for (int p = Hp[j]; p < Hp[j + 1]; p++) if (Hi[p] != j) { hp[j + 2]++; hp[Hi[p] + 2]++; }
    for (int j = 0; j < n; j++) hp[j + 2] += hp[j + 1];
    for (int j = 0; j < n && Hp; j++) for (int p = Hp[j]; p < Hp[j + 1]; p++) if (Hi[p] != j) { hj[hp[j + 1]++] = Hi[p]; hj[hp[Hi[p] + 1]++] = j; }
    /* the levels: each connected part from an end, found by a first sweep from its lowest variable (the variable of the
     * last level with the fewest entries); the parts' levels follow one another */
    for (int j = 0; j < n; j++) S->lev[j] = lv2[j] = -1;
    memset(rseen, 0, (size_t)m);
    int L = 0;
    for (int s = 0; s < n; s++) {
        if (S->lev[s] >= 0) continue;
        int qn = 0; const int l1 = st_bfs(s, 0, Ap, Ai, rp, rj, hp, hj, lv2, rseen, q, &qn);
        int e = s, ebest = -1;
        for (int t = 0; t < qn; t++) { const int v = q[t]; if (lv2[v] != l1 - 1) continue;
            const int d = (Ap[v + 1] - Ap[v]) + (hp[v + 1] - hp[v]); if (ebest < 0 || d < ebest) { ebest = d; e = v; } }
        for (int t = 0; t < qn; t++) for (int p = Ap[q[t]]; p < Ap[q[t] + 1]; p++) rseen[Ai[p]] = 0;   /* (the part's rows only: the cost stays linear with many parts) */
        L += st_bfs(e, L, Ap, Ai, rp, rj, hp, hj, S->lev, rseen, q, &qn);
    }
    S->L = L;
    /* a row's block: the lowest level it touches; the equality rows by block */
    S->vst = calloc((size_t)L + 2, sizeof(int)); S->rst = calloc((size_t)L + 2, sizeof(int)); S->cc = calloc((size_t)L + 1, sizeof(int)); S->psi = calloc((size_t)L + 1, 1);
    cnt = calloc((size_t)L + 2, sizeof(int)); csup = calloc((size_t)n + 1, 1);
    if (!S->vst || !S->rst || !S->cc || !S->psi || !cnt || !csup) goto fail;
    int ni = 0; size_t nie = 0;
    for (int i = 0; i < m; i++) {
        int k = L; for (int t = rp[i]; t < rp[i + 1]; t++) if (S->lev[rj[t]] < k) k = S->lev[rj[t]];
        if (rp[i] == rp[i + 1]) k = 0;
        S->rblk[i] = k; S->rpos[i] = -1;
        if (eq[i]) S->rst[k + 1]++; else { ni++; nie += (size_t)(rp[i + 1] - rp[i]); }
        /* its variables in level k + 1 are coupling columns of that level (through F for an equality row, through Phi
         * for an inequality row) */
        for (int t = rp[i]; t < rp[i + 1]; t++) if (S->lev[rj[t]] == k + 1) { csup[rj[t]] = 1; if (!eq[i]) S->psi[k] = 1; }
    }
    for (int j = 0; j < n; j++) for (int t = hp[j]; t < hp[j + 1]; t++) { const int w = hj[t]; if (S->lev[w] == S->lev[j] + 1) { csup[w] = 1; S->psi[S->lev[j]] = 1; } }
    for (int k = 0; k < L; k++) S->rst[k + 1] += S->rst[k];
    for (int i = 0; i < m; i++) if (eq[i]) { const int k = S->rblk[i]; S->rpos[i] = cnt[k]; S->rord[S->rst[k] + cnt[k]++] = i; }
    /* the variables by level, the coupling columns of each level first */
    for (int j = 0; j < n; j++) S->vst[S->lev[j] + 1]++;
    for (int k = 0; k < L; k++) S->vst[k + 1] += S->vst[k];
    memset(cnt, 0, sizeof(int) * ((size_t)L + 1));
    for (int j = 0; j < n; j++) if (csup[j]) { const int k = S->lev[j]; S->pos[j] = cnt[k]++; }
    for (int k = 1; k < L; k++) S->cc[k - 1] = cnt[k];
    for (int j = 0; j < n; j++) if (!csup[j]) { const int k = S->lev[j]; S->pos[j] = cnt[k]++; }
    for (int j = 0; j < n; j++) S->vord[S->vst[S->lev[j]] + S->pos[j]] = j;
    /* the blocks' places and the work */
    S->oPhi = st_alloc(sizeof(size_t) * ((size_t)L + 1)); S->oE = st_alloc(sizeof(size_t) * ((size_t)L + 1)); S->oF = st_alloc(sizeof(size_t) * ((size_t)L + 1));
    S->oPsi = st_alloc(sizeof(size_t) * ((size_t)L + 1)); S->oS = st_alloc(sizeof(size_t) * ((size_t)L + 1));
    if (!S->oPhi || !S->oE || !S->oF || !S->oPsi || !S->oS) goto fail;
    size_t o = 0; double wf = 0, ws = 0; int wmax = 0;
    for (int k = 0; k < L; k++) {
        const double nz = S->vst[k + 1] - S->vst[k], me = S->rst[k + 1] - S->rst[k], c = S->cc[k];
        if (nz > wmax) wmax = (int)nz;
        if (me > wmax) wmax = (int)me;
        S->oPhi[k] = o; o += (size_t)(nz * nz);
        S->oE[k] = o; o += (size_t)(me * nz);
        S->oF[k] = o; o += (size_t)(me * c);
        S->oPsi[k] = o; if (S->psi[k]) o += (size_t)(c * nz);
        S->oS[k] = o; o += (size_t)(me * me);
        wf += nz * nz * nz / 3 + nz * nz * me + me * me * nz + me * me * me / 3 + me * me * c + c * c * me;
        ws += 2 * (nz * nz + 2 * nz * me + me * me + 2 * me * c);
        if (S->psi[k]) { wf += nz * nz * c + c * c * nz + me * nz * c; ws += 4 * c * nz; }
        if ((double)o > 0.9 * (double)cap || o > cap) goto fail;
    }
    S->nval = o; S->wfact = wf; S->wsolve = ws; S->wmax = wmax;
    S->val = st_alloc(sizeof(double) * (o + 1)); S->hH = st_alloc(sizeof(size_t) * (nh + 1)); S->hA = st_alloc(sizeof(size_t) * (na + 1));
    S->ni = ni; S->irow = st_alloc(sizeof(int) * ((size_t)ni + 1)); S->ip = st_alloc(sizeof(int) * ((size_t)ni + 2)); S->ij = st_alloc(sizeof(int) * (nie + 1)); S->iq = st_alloc(sizeof(int) * (nie + 1));
    S->xs = st_alloc(sizeof(double) * ((size_t)n + 1)); S->ys = st_alloc(sizeof(double) * ((size_t)m + 1)); S->wi = st_alloc(sizeof(double) * ((size_t)ni + 1));
    if (!S->val || !S->hH || !S->hA || !S->irow || !S->ip || !S->ij || !S->iq || !S->xs || !S->ys || !S->wi) goto fail;
    /* where each entry goes: H's (lower triangle of Phi_k by the places in the level, or Psi_k), A's in the equality rows */
    for (int j = 0; j < n && Hp; j++) for (int p = Hp[j]; p < Hp[j + 1]; p++) {
        const int i = Hi[p], li = S->lev[i], lj = S->lev[j];
        if (li == lj) { const int a = S->pos[i], b = S->pos[j], r = a > b ? a : b, c = a > b ? b : a; S->hH[p] = S->oPhi[li] + (size_t)c * (size_t)(S->vst[li + 1] - S->vst[li]) + (size_t)r; }
        else { const int hi_ = li > lj ? i : j, lo_ = li > lj ? j : i, k = S->lev[lo_]; S->hH[p] = S->oPsi[k] + (size_t)S->pos[lo_] * (size_t)S->cc[k] + (size_t)S->pos[hi_]; }
    }
    for (int j = 0; j < n; j++) for (int p = Ap[j]; p < Ap[j + 1]; p++) {
        const int i = Ai[p]; if (!eq[i]) { S->hA[p] = (size_t)-1; continue; }
        const int k = S->rblk[i]; const size_t me = (size_t)(S->rst[k + 1] - S->rst[k]);
        S->hA[p] = (S->lev[j] == k ? S->oE[k] : S->oF[k]) + (size_t)S->pos[j] * me + (size_t)S->rpos[i];
    }
    { int a = 0, t2 = 0; S->ip[0] = 0;
      for (int i = 0; i < m; i++) if (!eq[i]) { S->irow[a] = i; for (int t = rp[i]; t < rp[i + 1]; t++) { S->ij[t2] = rj[t]; S->iq[t2] = rq[t]; t2++; } S->ip[++a] = t2; } }
    free(rp); free(rj); free(rq); free(hp); free(hj); free(q); free(lv2); free(rseen); free(cnt); free(csup);
    return S;
fail:
    free(rp); free(rj); free(rq); free(hp); free(hj); free(q); free(lv2); free(rseen); free(cnt); free(csup);
    qp_stage_free(S);
    return NULL;
}

/* the blocks filled and factored: Hx and Ax the entries of H (lower triangle) and A, d (n) what is added to H's diagonal,
 * e (m) the rows' diagonal (an equality row's regularization, 1 / Ds (+ its regularization) of an inequality row).
 * Returns 0, or 1 when a block is not positive definite. */
int qp_stage_factor(QpStage *S, const int *Hp, const double *Hx, const int *Ap, const double *Ax, const double *d, const double *e) {
    const int n = S->n, L = S->L; double *v = S->val;
    memset(v, 0, sizeof(double) * S->nval);
    if (Hp) for (int p = 0; p < Hp[n]; p++) v[S->hH[p]] += Hx[p];
    for (int j = 0; j < n; j++) { const int k = S->lev[j]; v[S->oPhi[k] + (size_t)S->pos[j] * (size_t)(S->vst[k + 1] - S->vst[k] + 1)] += d[j]; }
    for (int p = 0; p < Ap[n]; p++) if (S->hA[p] != (size_t)-1) v[S->hA[p]] += Ax[p];
    /* an inequality row i adds a_i' a_i / e_i to Phi: each pair of its entries into Phi_k or Psi_k */
    for (int a = 0; a < S->ni; a++) {
        const double w = 1.0 / e[S->irow[a]]; S->wi[a] = w;
        for (int t = S->ip[a]; t < S->ip[a + 1]; t++) {
            const int j = S->ij[t], lj = S->lev[j], pj = S->pos[j]; const double wj = w * Ax[S->iq[t]];
            const size_t nzj = (size_t)(S->vst[lj + 1] - S->vst[lj]);
            for (int u = S->ip[a]; u <= t; u++) {
                const int i = S->ij[u], li = S->lev[i], pi = S->pos[i]; const double x = wj * Ax[S->iq[u]];
                if (li == lj) { const int r = pi > pj ? pi : pj, c = pi > pj ? pj : pi; v[S->oPhi[lj] + (size_t)c * nzj + (size_t)r] += x; }
                else if (li > lj) v[S->oPsi[lj] + (size_t)pj * (size_t)S->cc[lj] + (size_t)pi] += x;
                else v[S->oPsi[li] + (size_t)pi * (size_t)S->cc[li] + (size_t)pj] += x;
            }
        }
    }
    for (int k = 0; k < L; k++) {
        const int nz = S->vst[k + 1] - S->vst[k], me = S->rst[k + 1] - S->rst[k], c = S->cc[k];
        double *Phi = v + S->oPhi[k], *E = v + S->oE[k], *F = v + S->oF[k], *Psi = v + S->oPsi[k], *Sk = v + S->oS[k];
        if (sk_potrf(nz, Phi, nz)) return 1;
        if (me) {
            sk_trsm_rlt(me, nz, Phi, nz, E, me);                                 /* X = E L^-T */
            sk_syrk_ln(me, nz, 1.0, E, me, 0.0, Sk, me);                         /* S = X X' + the rows' diagonal */
            for (int i = 0; i < me; i++) Sk[(size_t)i * me + i] += e[S->rord[S->rst[k] + i]];
            if (sk_potrf(me, Sk, me)) return 1;
        }
        if (k + 1 == L || !c) continue;
        const int nz1 = S->vst[k + 2] - S->vst[k + 1]; double *Phi1 = v + S->oPhi[k + 1];
        if (S->psi[k]) {
            sk_trsm_rlt(c, nz, Phi, nz, Psi, c);                                 /* P = Psi L^-T */
            sk_syrk_ln(c, nz, -1.0, Psi, c, 1.0, Phi1, nz1);                     /* Phi_{k+1} -= P P' */
            if (me) sk_gemm_nt(me, c, nz, E, me, Psi, c, F, me);                 /* F~ = F - X P' */
        }
        if (me) {
            sk_trsm_lln(me, c, Sk, me, F, me);                                   /* Y = M^-1 F~ */
            sk_syrk_lt(c, me, F, me, Phi1, nz1);                                 /* Phi_{k+1} += Y' Y */
        }
    }
    return 0;
}

/* K (dx; dy) = b solved in place on b (n + m) with the factors of qp_stage_factor (Ax: A's entries) */
void qp_stage_solve(QpStage *S, const double *Ax, double *b) {
    const int n = S->n, L = S->L; double *v = S->val, *xs = S->xs, *ys = S->ys;
    /* a = b_x - sum over the inequality rows of a_i' b_i / e_i, in stage order; the equality rows' b by block */
    for (int a = 0; a < S->ni; a++) { const double t = S->wi[a] * b[n + S->irow[a]]; if (t != 0) for (int u = S->ip[a]; u < S->ip[a + 1]; u++) b[S->ij[u]] -= Ax[S->iq[u]] * t; }
    for (int t = 0; t < n; t++) xs[t] = b[S->vord[t]];
    const int ne = S->rst[L];
    for (int t = 0; t < ne; t++) ys[t] = b[n + S->rord[t]];
    for (int k = 0; k < L; k++) {
        const int nz = S->vst[k + 1] - S->vst[k], me = S->rst[k + 1] - S->rst[k], c = S->cc[k];
        double *xk = xs + S->vst[k], *yk = ys + S->rst[k], *x1 = xs + S->vst[k + 1];
        sk_trsv_n(nz, v + S->oPhi[k], nz, xk);                                   /* u = L^-1 a */
        if (me) {
            sk_gemv_n(me, nz, 1.0, v + S->oE[k], me, xk, yk);                    /* b~ = b + X u */
            sk_trsv_n(me, v + S->oS[k], me, yk);                                 /* v = M^-1 b~ */
        }
        if (k + 1 == L || !c) continue;
        if (S->psi[k]) sk_gemv_n(c, nz, -1.0, v + S->oPsi[k], c, xk, x1);        /* a+ -= P u */
        if (me) sk_gemv_t(me, c, -1.0, v + S->oF[k], me, yk, x1);                /* a+ -= Y' v */
    }
    for (int k = L - 1; k >= 0; k--) {
        const int nz = S->vst[k + 1] - S->vst[k], me = S->rst[k + 1] - S->rst[k], c = S->cc[k];
        double *xk = xs + S->vst[k], *yk = ys + S->rst[k], *x1 = xs + S->vst[k + 1];
        const int cpl = k + 1 < L && c;
        if (me) {
            if (cpl) sk_gemv_n(me, c, -1.0, v + S->oF[k], me, x1, yk);           /* v -= Y x+ */
            sk_trsv_t(me, v + S->oS[k], me, yk);                                 /* y = M^-T v */
        }
        for (int t = 0; t < nz; t++) xk[t] = -xk[t];                             /* t = X' y - P' x+ - u */
        if (me) sk_gemv_t(me, nz, 1.0, v + S->oE[k], me, yk, xk);
        if (cpl && S->psi[k]) sk_gemv_t(c, nz, -1.0, v + S->oPsi[k], c, x1, xk);
        sk_trsv_t(nz, v + S->oPhi[k], nz, xk);                                   /* x = L^-T t */
    }
    for (int t = 0; t < n; t++) b[S->vord[t]] = xs[t];
    for (int t = 0; t < ne; t++) b[n + S->rord[t]] = ys[t];
    /* the inequality rows' dy_i = (b_i - a_i dx) / e_i */
    for (int a = 0; a < S->ni; a++) { double s = b[n + S->irow[a]]; for (int u = S->ip[a]; u < S->ip[a + 1]; u++) s -= Ax[S->iq[u]] * b[S->ij[u]]; b[n + S->irow[a]] = S->wi[a] * s; }
}

/* ---- The normal equations by stage (H diagonal).
 *
 * With H diagonal the columns are eliminated first, each by a division: M = A W A' + diag(e), W = diag(w) the inverse of
 * the columns' diagonal, e an inequality row's 1 / Ds (0 for an equality row). Here the rows are sorted into levels,
 * breadth-first over the graph whose edges join two rows that share a column; M is then block tridiagonal over the
 * levels, diagonal blocks D_k and coupling blocks C_k (level k + 1 against level k), and its Cholesky factorization is
 *
 *     D~_k = L_k L_k'        B_k = C_k L_k^-T        D~_{k+1} = D_{k+1} - B_k B_k'
 *
 * on dense blocks, with no ordering and no symbolic analysis: for a model predictive control problem the levels are the
 * stages' dynamics rows, the blocks have the number of states, and the work grows linearly with the horizon. A pivot that
 * falls to 1e-16 of its diagonal entry before the elimination is replaced by a huge one (the row is linearly dependent
 * on those before it: its multiplier's step is zero), as in the sparse factorization of the normal equations. */
#define SNE_PW 256        /* the columns of one product of the forming */
#define SNE_PG 16         /* a dense group of at least this many rows, in a run of at least 4 columns, goes through products */
struct QpStageNe {
    int n, m, L;
    int *lev, *pos;             /* a row's level and its place in the level */
    int *rst, *rord;            /* L + 1: the levels' starts; the rows in level order */
    size_t *oD, *oC;            /* the blocks' places in val */
    size_t nval; double *val, *d0, *ys, *wk;
    /* the forming: each column's entries grouped, those in its lower level first (fsp[j] of them), each group by place:
     * fq the entries' places in A, fps their rows' places in the level, flo the column's lower level */
    int *fq, *fps, *fsp, *flo; double *fx;
    /* runs of columns with one dense group (sne_form): frun[j] the run's length at its first column, -1 inside it, 0
     * for a column of no run; frhi[j]: the dense group is the upper one; fpan: the panel of a product */
    int *frun; char *frhi; double *fpan;
    /* the forming on the threads: the columns by their lower level (lcol, starts lst), and a work vector (fxn entries)
     * and a panel (fpn entries) for each of nth threads */
    int *lcol, *lst, nth; size_t fxn, fpn;
    int mid;      /* the level where the eliminations from the chain's two ends meet */
    int *clb, *cub; /* (15 November 2026) the coupling block C_k (rows: level k + 1, columns: level k): its entries (i, j) are zero for i > j + clb[k] and for j > i + cub[k] */
    double wfact, wsolve, wform, dense, lnz, wbar; int wmax;
};
void qp_stagene_free(QpStageNe *S) {
    if (!S) return;
    free(S->lev); free(S->pos); free(S->rst); free(S->rord); free(S->oD); free(S->oC); free(S->val); free(S->d0); free(S->ys); free(S->wk);
    free(S->fq); free(S->fps); free(S->fsp); free(S->flo); free(S->fx); free(S->frun); free(S->frhi); free(S->fpan); free(S->lcol); free(S->lst); free(S->clb); free(S->cub);
    free(S);
}
int qp_stagene_levels(const QpStageNe *S) { return S->L; }
int qp_stagene_widest(const QpStageNe *S) { return S->wmax; }
double qp_stagene_fact_work(const QpStageNe *S) { return S->wfact; }
double qp_stagene_solve_work(const QpStageNe *S) { return S->wsolve; }
double qp_stagene_form_work(const QpStageNe *S) { return S->wform; }
double qp_stagene_dense(const QpStageNe *S) { return S->dense; }
double qp_stagene_lnz(const QpStageNe *S) { return S->lnz; }
double qp_stagene_width(const QpStageNe *S) { return S->wbar; }
size_t qp_stagene_entries(const QpStageNe *S) { return S->nval; }

/* (15 November 2026) The rows' order within each level, chosen so that the coupling blocks are banded where the
 * constraints allow it (sk_trsm_rlt_par). The levels of a model predictive control QP found by the breadth-first sweep
 * are its stages shifted by one row: a level is a stage's rows but one, and one row of the next stage; that row shares
 * the next stage's dense input columns with the whole next level, a dense column of the coupling block. From the chain's
 * first level to the meeting level c the elimination goes upwards and wants every coupling block C_k (rows: level k + 1,
 * columns: level k) zero below a band under its diagonal: the rows of level k that couple to more than half of level
 * k + 1 are put last, the others in the order of their first neighbour in level k - 1. From the last level down to c + 1
 * it goes downwards and wants C_k zero above a band: the rows that couple to more than half of the level above first,
 * the others in the order of their first neighbour in the level above. A heuristic on the order only: the bandwidths
 * are measured afterwards (clb, cub) and any order is valid. */
typedef struct { int d, key, i; } SneOrd;
static int sne_ord_cmp(const void *a, const void *b) {
    const SneOrd *x = a, *y = b;
    if (x->d != y->d) return x->d - y->d;
    if (x->key != y->key) return x->key < y->key ? -1 : 1;
    return x->i - y->i;
}
static void sne_order_levels(QpStageNe *S, int n, const int *Ap, const int *Ai, int c) {
    const int m = S->m, L = S->L;
    int *clo = malloc(sizeof(int) * ((size_t)n + 1)), *cst = calloc((size_t)L + 2, sizeof(int)), *cl = malloc(sizeof(int) * ((size_t)n + 1));
    int *dup = calloc((size_t)m + 1, sizeof(int)), *key = malloc(sizeof(int) * ((size_t)m + 1)); SneOrd *o = malloc(sizeof(SneOrd) * ((size_t)S->wmax + 1));
    if (!clo || !cst || !cl || !dup || !key || !o) { free(clo); free(cst); free(cl); free(dup); free(key); free(o); return; }
    for (int j = 0; j < n; j++) {
        int lo = L, hi = -1; for (int p = Ap[j]; p < Ap[j + 1]; p++) { const int k = S->lev[Ai[p]]; if (k < lo) lo = k; if (k > hi) hi = k; }
        clo[j] = hi > lo ? lo : -1;
        if (hi > lo) { cst[lo + 1]++; int c2 = 0; for (int p = Ap[j]; p < Ap[j + 1]; p++) c2 += S->lev[Ai[p]] != lo;
            for (int p = Ap[j]; p < Ap[j + 1]; p++) if (S->lev[Ai[p]] == lo) dup[Ai[p]] += c2; }
    }
    for (int k = 0; k < L; k++) cst[k + 1] += cst[k];
    { int *nx = malloc(sizeof(int) * ((size_t)L + 1)); if (!nx) { free(clo); free(cst); free(cl); free(dup); free(key); free(o); return; }
      memcpy(nx, cst, sizeof(int) * (size_t)L); for (int j = 0; j < n; j++) if (clo[j] >= 0) cl[nx[clo[j]]++] = j; free(nx); }
    #define SNE_DENSE_UP(i, k) ((k) + 1 < L && 2 * dup[i] > S->rst[(k) + 2] - S->rst[(k) + 1])
    for (int k = 0; k <= c && k < L; k++) {               /* upwards: the dense couplers last */
        const int r = S->rst[k + 1] - S->rst[k], *rows = S->rord + S->rst[k];
        for (int t = 0; t < r; t++) key[rows[t]] = k > 0 ? INT_MAX : S->pos[rows[t]];
        if (k > 0) for (int t = cst[k - 1]; t < cst[k]; t++) { const int j = cl[t]; int amin = INT_MAX;
            for (int p = Ap[j]; p < Ap[j + 1]; p++) { const int i = Ai[p]; if (S->lev[i] == k - 1 && !SNE_DENSE_UP(i, k - 1) && S->pos[i] < amin) amin = S->pos[i]; }
            for (int p = Ap[j]; p < Ap[j + 1]; p++) { const int i = Ai[p]; if (S->lev[i] == k && amin < key[i]) key[i] = amin; } }
        for (int t = 0; t < r; t++) { const int i = rows[t]; o[t].d = SNE_DENSE_UP(i, k) ? 1 : 0; o[t].key = key[i]; o[t].i = i; }
        qsort(o, (size_t)r, sizeof(SneOrd), sne_ord_cmp);
        for (int t = 0; t < r; t++) { S->pos[o[t].i] = t; S->rord[S->rst[k] + t] = o[t].i; }
    }
    for (int k = L - 1; k > c; k--) {                     /* downwards: the dense couplers first */
        const int r = S->rst[k + 1] - S->rst[k], *rows = S->rord + S->rst[k];
        for (int t = 0; t < r; t++) key[rows[t]] = k + 1 < L ? INT_MAX : S->pos[rows[t]];
        if (k + 1 < L) for (int t = cst[k]; t < cst[k + 1]; t++) { const int j = cl[t]; int bmin = INT_MAX;
            for (int p = Ap[j]; p < Ap[j + 1]; p++) { const int i = Ai[p]; if (S->lev[i] == k + 1 && S->pos[i] < bmin) bmin = S->pos[i]; }
            for (int p = Ap[j]; p < Ap[j + 1]; p++) { const int i = Ai[p]; if (S->lev[i] == k && bmin < key[i]) key[i] = bmin; } }
        for (int t = 0; t < r; t++) { const int i = rows[t]; o[t].d = SNE_DENSE_UP(i, k) ? 0 : 1; o[t].key = key[i]; o[t].i = i; }
        qsort(o, (size_t)r, sizeof(SneOrd), sne_ord_cmp);
        for (int t = 0; t < r; t++) { S->pos[o[t].i] = t; S->rord[S->rst[k] + t] = o[t].i; }
    }
    #undef SNE_DENSE_UP
    free(clo); free(cst); free(cl); free(dup); free(key); free(o);
}
static void sne_form(QpStageNe *S, const int *Ap, const double *Ax, const double *w);
QpStageNe *qp_stagene_build(int n, int m, const int *Ap, const int *Ai, size_t cap, double own, double *info) {
    if (info) info[0] = info[1] = info[2] = info[3] = 0;
    if (m <= 0 || n <= 0) return NULL;
    QpStageNe *S = calloc(1, sizeof *S); if (!S) return NULL;
    S->n = n; S->m = m;
    const size_t na = (size_t)Ap[n];
    int *rp = calloc((size_t)m + 2, sizeof(int)), *rj = st_alloc(sizeof(int) * (na + 1)), *zp = calloc((size_t)m + 2, sizeof(int));
    int *q = st_alloc(sizeof(int) * ((size_t)m + 1)), *lv2 = st_alloc(sizeof(int) * ((size_t)m + 1)), *cnt = NULL;
    char *cseen = st_alloc((size_t)n + 1);
    S->lev = st_alloc(sizeof(int) * ((size_t)m + 1)); S->pos = st_alloc(sizeof(int) * ((size_t)m + 1)); S->rord = st_alloc(sizeof(int) * ((size_t)m + 1));
    if (!rp || !rj || !zp || !q || !lv2 || !cseen || !S->lev || !S->pos || !S->rord) goto fail;
    for (int j = 0; j < n; j++) for (int p = Ap[j]; p < Ap[j + 1]; p++) rp[Ai[p] + 2]++;
    for (int i = 0; i < m; i++) rp[i + 2] += rp[i + 1];
    for (int j = 0; j < n; j++) for (int p = Ap[j]; p < Ap[j + 1]; p++) rj[rp[Ai[p] + 1]++] = j;
    /* the rows' levels: the sweep of the variables' levels with the rows and the columns exchanged */
    for (int i = 0; i < m; i++) S->lev[i] = lv2[i] = -1;
    memset(cseen, 0, (size_t)n);
    int L = 0;
    for (int s = 0; s < m; s++) {
        if (S->lev[s] >= 0) continue;
        int qn = 0; const int l1 = st_bfs(s, 0, rp, rj, Ap, Ai, zp, NULL, lv2, cseen, q, &qn);
        int e = s, ebest = -1;
        for (int t = 0; t < qn; t++) { const int v = q[t]; if (lv2[v] != l1 - 1) continue; const int d = rp[v + 1] - rp[v]; if (ebest < 0 || d < ebest) { ebest = d; e = v; } }
        for (int t = 0; t < qn; t++) for (int p = rp[q[t]]; p < rp[q[t] + 1]; p++) cseen[rj[p]] = 0;
        L += st_bfs(e, L, rp, rj, Ap, Ai, zp, NULL, S->lev, cseen, q, &qn);
    }
    S->L = L;
    S->rst = calloc((size_t)L + 2, sizeof(int)); cnt = calloc((size_t)L + 2, sizeof(int));
    S->oD = st_alloc(sizeof(size_t) * ((size_t)L + 1)); S->oC = st_alloc(sizeof(size_t) * ((size_t)L + 1));
    if (!S->rst || !cnt || !S->oD || !S->oC) goto fail;
    for (int i = 0; i < m; i++) S->rst[S->lev[i] + 1]++;
    for (int k = 0; k < L; k++) S->rst[k + 1] += S->rst[k];
    for (int i = 0; i < m; i++) { const int k = S->lev[i]; S->pos[i] = cnt[k]++; S->rord[S->rst[k] + S->pos[i]] = i; }
    size_t o = 0; double wf = 0, ws = 0, wform = 0; int wmax = 0;
    for (int k = 0; k < L; k++) {
        const double r = S->rst[k + 1] - S->rst[k], r1 = k + 1 < L ? S->rst[k + 2] - S->rst[k + 1] : 0;
        if (r > wmax) wmax = (int)r;
        S->oD[k] = o; o += (size_t)(r * r);
        S->oC[k] = o; o += (size_t)(r1 * r);
        wf += r * r * r / 3 + r * r * r1 + r1 * r1 * r;
        ws += 2 * (r * r + 2 * r * r1);
        S->lnz += r * (r + 1) / 2 + r * r1;
        S->wbar += r * r * r;                 /* (the blocks' width weighted by their work: sum r^3 / sum r^2, below) */
        if (own <= 0 && ((double)o > 0.9 * (double)cap || o > cap)) goto fail;
    }
    for (int j = 0; j < n; j++) { const double c = Ap[j + 1] - Ap[j]; wform += c * (c + 1) / 2; }
    S->nval = o; S->wfact = wf; S->wsolve = ws; S->wform = wform; S->wmax = wmax;
    /* the meeting level: the first whose levels before it carry half of the factorization's work */
    {   double half = 0, run = 0; S->mid = L - 1;
        for (int k = 0; k < L; k++) { const double r = S->rst[k + 1] - S->rst[k]; half += r * r * r; }
        for (int k = 0; k < L; k++) { const double r = S->rst[k + 1] - S->rst[k]; run += r * r * r; if (2 * run >= half) { S->mid = k; break; } }
    }
    {   static int ord = -1; if (ord < 0) { const char *e = getenv("BRISK_QPSTAGEBAND"); ord = !(e && atoi(e) == 0); }
        if (ord) sne_order_levels(S, n, Ap, Ai, S->mid); }
    { double r2 = 0; for (int k = 0; k < L; k++) { const double r = S->rst[k + 1] - S->rst[k]; r2 += r * r; } S->wbar = r2 > 0 ? S->wbar / r2 : 0; }
    /* The automatic rule's three conditions that the levels' sizes decide (own > 0: the entries of the augmented system
     * before any fill) are applied here, before anything is allocated or counted over the blocks: a QP whose levels are
     * huge dense blocks (a lasso or Huber regression with dense columns: one level of 16,896 rows) is turned away after
     * the breadth-first sweeps, at the cost of two passes over A's pattern. info: the factor's entries, the blocks' width
     * by work, the factorization's and the forming's work. */
    if (info) { info[0] = S->lnz; info[1] = S->wbar; info[2] = wf; info[3] = wform; }
    if (own > 0 && (S->lnz > 4.0 * own || S->wbar < 8.0 || wf < wform)) goto fail;
    if ((double)o > 0.9 * (double)cap || o > cap) goto fail;
    S->val = st_alloc(sizeof(double) * (o + 1)); S->d0 = st_alloc(sizeof(double) * ((size_t)m + 1)); S->ys = st_alloc(sizeof(double) * ((size_t)m + 1));
    S->wk = st_alloc(sizeof(double) * (2 * (size_t)wmax * (size_t)wmax + 1));        /* (a work block for each end of the chain) */
    S->fq = st_alloc(sizeof(int) * (na + 1)); S->fps = st_alloc(sizeof(int) * (na + 1)); S->fsp = st_alloc(sizeof(int) * ((size_t)n + 1)); S->flo = st_alloc(sizeof(int) * ((size_t)n + 1));
    S->nth = 1;
#ifdef _OPENMP
    S->nth = omp_get_max_threads(); if (S->nth < 1) S->nth = 1;
#endif
    { int cmax = 0; for (int j = 0; j < n; j++) if (Ap[j + 1] - Ap[j] > cmax) cmax = Ap[j + 1] - Ap[j]; S->fxn = (size_t)cmax + 8; S->fx = st_alloc(sizeof(double) * S->fxn * (size_t)S->nth); }
    if (!S->val || !S->d0 || !S->ys || !S->wk || !S->fq || !S->fps || !S->fsp || !S->flo || !S->fx) goto fail;
    /* (a column's rows lie in one level or in two neighbouring ones; within a level the places grow with the row index,
     * so each group is sorted as A's column is) */
    for (int j = 0; j < n; j++) {
        int lo = L; for (int p = Ap[j]; p < Ap[j + 1]; p++) if (S->lev[Ai[p]] < lo) lo = S->lev[Ai[p]];
        int t = Ap[j];
        for (int p = Ap[j]; p < Ap[j + 1]; p++) if (S->lev[Ai[p]] == lo) { S->fq[t] = p; S->fps[t] = S->pos[Ai[p]]; t++; }
        S->fsp[j] = t - Ap[j]; S->flo[j] = lo;
        for (int p = Ap[j]; p < Ap[j + 1]; p++) if (S->lev[Ai[p]] != lo) { S->fq[t] = p; S->fps[t] = S->pos[Ai[p]]; t++; }
        /* (each group by place: the rows' order within a level is no longer that of their indices) */
        for (int g = 0; g < 2; g++) { const int a0 = g ? Ap[j] + S->fsp[j] : Ap[j], a1 = g ? Ap[j + 1] : Ap[j] + S->fsp[j];
            for (int u = a0 + 1; u < a1; u++) { const int fp = S->fps[u], fqv = S->fq[u]; int w = u; while (w > a0 && S->fps[w - 1] > fp) { S->fps[w] = S->fps[w - 1]; S->fq[w] = S->fq[w - 1]; w--; } S->fps[w] = fp; S->fq[w] = fqv; } }
    }
    /* the runs of columns that share a dense group: the same level, the same first row and length, the rows neighbours
     * in the level (the larger group when both of a column's are dense) */
    S->frun = calloc((size_t)n + 1, sizeof(int)); S->frhi = calloc((size_t)n + 1, 1);
    if (!S->frun || !S->frhi) goto fail;
    {   int gmax = 0, j = 0;
        while (j < n) {
            int sig[4] = { -1, 0, 0, 0 };           /* level, upper group, first row, length */
            int q = 0;
            for (int jj = j; jj < n; jj++) {
                const int p0 = Ap[jj], c = Ap[jj + 1] - p0, c1 = S->fsp[jj]; const int *ps = S->fps + p0;
                const int g0 = c1 >= SNE_PG && ps[c1 - 1] - ps[0] == c1 - 1, g1 = c - c1 >= SNE_PG && ps[c - 1] - ps[c1] == c - c1 - 1;
                int me[4] = { -1, 0, 0, 0 };
                if (g1 && (!g0 || c - c1 >= c1)) { me[0] = S->flo[jj]; me[1] = 1; me[2] = ps[c1]; me[3] = c - c1; }
                else if (g0) { me[0] = S->flo[jj]; me[1] = 0; me[2] = ps[0]; me[3] = c1; }
                if (me[0] < 0) break;
                if (jj == j) memcpy(sig, me, sizeof sig);
                else if (memcmp(sig, me, sizeof sig)) break;
                q++;
            }
            if (q >= 4) { S->frun[j] = q; for (int t = 0; t < q; t++) { S->frhi[j + t] = (char)sig[1]; if (t) S->frun[j + t] = -1; } if (sig[3] > gmax) gmax = sig[3]; j += q; }
            else j += q > 0 ? q : 1;
        }
        if (getenv("BRISK_QPSTAGEDBG")) { long nrun = 0, ncol = 0; for (int t = 0; t < n; t++) { nrun += S->frun[t] > 0; ncol += S->frun[t] != 0; } fprintf(stderr, "[stage forming: %ld runs over %ld of %d columns, the longest dense group %d]\n", nrun, ncol, n, gmax); }
        S->fpn = (size_t)gmax * SNE_PW + 8;
        S->fpan = st_alloc(sizeof(double) * S->fpn * (size_t)S->nth);
        if (!S->fpan) goto fail;
    }
    /* the coupling blocks' bandwidths, from the columns with entries in both of their levels (the entries' places are sorted
     * within a group): C_k's entry (b, a) for a place a in level k and b in level k + 1 */
    S->clb = st_alloc(sizeof(int) * ((size_t)L + 1)); S->cub = st_alloc(sizeof(int) * ((size_t)L + 1));
    for (int k = 0; k <= L; k++) { S->clb[k] = 0; S->cub[k] = 0; }
    {   static int bnd = -1; if (bnd < 0) { const char *e = getenv("BRISK_QPSTAGEBAND"); bnd = !(e && atoi(e) == 0); }     /* (BRISK_QPSTAGEBAND=0: every coupling block as dense, as before) */
        for (int j = 0; j < n; j++) {
            const int p0 = Ap[j], c = Ap[j + 1] - p0, c1 = S->fsp[j], lo = S->flo[j];
            if (c1 <= 0 || c1 >= c) continue;
            int amin = S->fps[p0], amax = amin, bmin = S->fps[p0 + c1], bmax = bmin;
            for (int q = 1; q < c1; q++) { const int a = S->fps[p0 + q]; if (a < amin) amin = a; if (a > amax) amax = a; }
            for (int q = c1 + 1; q < c; q++) { const int b = S->fps[p0 + q]; if (b < bmin) bmin = b; if (b > bmax) bmax = b; }
            if (bmax - amin > S->clb[lo]) S->clb[lo] = bmax - amin;
            if (amax - bmin > S->cub[lo]) S->cub[lo] = amax - bmin;
        }
        if (!bnd) for (int k = 0; k <= L; k++) { S->clb[k] = 1 << 30; S->cub[k] = 1 << 30; }
        if (getenv("BRISK_QPSTAGEDBG")) { int nl = 0, nu = 0; for (int k = 0; k + 1 < L; k++) { if (k < S->mid && S->clb[k] > 8) nl++; if (k >= S->mid && S->cub[k] > 8) nu++; }
            fprintf(stderr, "[stage normal equations: %d levels, widest %d rows, the meeting level %d; coupling blocks wider than 8 below the diagonal on the way up: %d, above it on the way down: %d]\n", L, S->wmax, S->mid, nl, nu);
            if (nl + nu > 0 && L > 6) fprintf(stderr, "[  bandwidths (below, above) of the blocks 0..3: %d %d, %d %d, %d %d, %d %d; of the blocks at the meeting level: %d %d, %d %d]\n", S->clb[0], S->cub[0], S->clb[1], S->cub[1], S->clb[2], S->cub[2], S->clb[3], S->cub[3], S->clb[S->mid - 1], S->cub[S->mid - 1], S->clb[S->mid], S->cub[S->mid]); }
    }
    /* the columns by their lower level (a counting sort: within a level in the order of their indices, so a run stays whole) */
    S->lcol = st_alloc(sizeof(int) * ((size_t)n + 1)); S->lst = calloc((size_t)L + 2, sizeof(int));
    if (!S->lcol || !S->lst) goto fail;
    for (int j = 0; j < n; j++) if (Ap[j + 1] > Ap[j]) S->lst[S->flo[j] + 1]++;
    for (int k = 0; k < L; k++) S->lst[k + 1] += S->lst[k];
    { int *nx = st_alloc(sizeof(int) * ((size_t)L + 1)); if (!nx) goto fail; memcpy(nx, S->lst, sizeof(int) * (size_t)L);
      for (int j = 0; j < n; j++) if (Ap[j + 1] > Ap[j]) S->lcol[nx[S->flo[j]]++] = j;
      free(nx); }
    /* How dense the diagonal blocks are. A diagonal block that is full is a clique of M's graph, and the cliques of
     * different levels share no row: whatever the ordering, a Cholesky factorization of M does at least the work of
     * factoring each of them, r^3 / 3. The sum over the columns of the square of their entries below the diagonal inside
     * the diagonal blocks (that work for a full block, a bound without fill for a sparse one), against the same sum with
     * the blocks full: near 1, M is dense inside its levels and the stage-wise factorization does work no ordering
     * avoids, at the rate of dense kernels and without an analysis; near 0, M is sparse inside wide levels (a grid, a
     * network) and a sparse factorization with its own ordering does far less work. */
    /* The pattern of a diagonal block is the union, over the columns with rows in the level, of the pairs of those rows:
     * marked level by level in a block of bytes. A column whose rows in the level are the previous column's (the same
     * first row and number of neighbouring rows, as in a dense block of dynamics) marks nothing new and is skipped, so
     * the pass costs the square of a dense group once, not once a column. */
    {   char *pat = malloc((size_t)wmax * (size_t)wmax + 1); if (!pat) goto fail;
        double ws_ = 0, wd_ = 0;
        for (int k = 0; k < L; k++) {
            const int r = S->rst[k + 1] - S->rst[k];
            memset(pat, 0, (size_t)r * (size_t)r);
            for (int side = 0; side < 2; side++) {          /* the columns whose upper rows (side 0: lower level k - 1) or lower rows (side 1) lie in level k */
                const int kk = side ? k : k - 1; if (kk < 0) continue;
                int pf = -1, pl = -1;                         /* the last dense group marked: its first row and length */
                for (int t = S->lst[kk]; t < S->lst[kk + 1]; t++) {
                    const int j = S->lcol[t], p0 = Ap[j], c = Ap[j + 1] - p0, c1 = S->fsp[j];
                    const int *ps = S->fps + p0 + (side ? 0 : c1); const int g = side ? c1 : c - c1;
                    if (g <= 0) continue;
                    if (ps[g - 1] - ps[0] == g - 1) {         /* neighbouring rows */
                        if (ps[0] == pf && g == pl) continue;
                        pf = ps[0]; pl = g;
                        for (int b = 0; b < g; b++) memset(pat + (size_t)(pf + b) * r + (size_t)(pf + b), 1, (size_t)(g - b));
                    } else for (int b = 0; b < g; b++) { char *col = pat + (size_t)ps[b] * r; for (int a2 = b; a2 < g; a2++) col[ps[a2]] = 1; }
                }
            }
            for (int c = 0; c < r; c++) {
                int cn = 0; const char *dc = pat + (size_t)c * r;
                for (int i = c + 1; i < r; i++) cn += dc[i];
                const double full = (double)(r - 1 - c);
                ws_ += (double)cn * cn; wd_ += full * full;
            }
        }
        free(pat);
        S->dense = wd_ > 0 ? ws_ / wd_ : 1.0;
    }
    free(rp); free(rj); free(zp); free(q); free(lv2); free(cseen); free(cnt);
    return S;
fail:
    free(rp); free(rj); free(zp); free(q); free(lv2); free(cseen); free(cnt);
    qp_stagene_free(S);
    return NULL;
}

/* one column's part of the blocks (x: a work vector of the column's length, pan: a panel) */
static void sne_form_col(QpStageNe *S, const int *Ap, const double *Ax, const double *w, int j, double *x, double *pan) {
    double *v = S->val;
    {
        const int p0 = Ap[j], c = Ap[j + 1] - p0, c1 = S->fsp[j], lo = S->flo[j]; if (!c) return;
        const int *ps = S->fps + p0, *fq = S->fq + p0; const double wj = w ? w[j] : 1.0;
        const size_t r = (size_t)(S->rst[lo + 1] - S->rst[lo]);
        double *D = v + S->oD[lo];
        /* (a group whose rows are neighbours in their level, as a dense block of dynamics gives: its loops run over
         * consecutive entries of a block's column with no index array) */
        const int g0 = c1 > 0 && ps[c1 - 1] - ps[0] == c1 - 1, g1 = c > c1 && ps[c - 1] - ps[c1] == c - c1 - 1;
        /* A run of columns with the same dense group (the same level, first row and length: the columns of one block of
         * dynamics): the group's part of M is G W G' for the dense matrix G of those columns, added by one level-3
         * product (dsyrk on G W^1/2, at most SNE_PW columns at a time) instead of a rank-one update a column. The run's
         * other parts (the other group, the two groups' cross terms) are added column by column below. pan0, pan1: this
         * column's lower / upper group is covered by a product already. */
        int pan0 = 0, pan1 = 0;
        if (S->frun[j] > 0) {
            const int q = S->frun[j], hi = S->frhi[j];           /* the run's length from j; which group is dense */
            const int g = hi ? c - c1 : c1, off = hi ? c1 : 0, st = ps[off];
            const int rr = hi ? S->rst[lo + 2] - S->rst[lo + 1] : (int)r;
            double *T = (hi ? v + S->oD[lo + 1] : D) + (size_t)st * (size_t)rr + (size_t)st;
            for (int t0 = 0; t0 < q; t0 += SNE_PW) {
                const int nc = q - t0 < SNE_PW ? q - t0 : SNE_PW;
                for (int t = 0; t < nc; t++) {
                    const int jj = j + t0 + t, pp = Ap[jj] + (hi ? S->fsp[jj] : 0); const double sw = sqrt(w ? w[jj] : 1.0); double *pc = pan + (size_t)t * (size_t)g;
                    if (Ax) for (int a = 0; a < g; a++) pc[a] = sw * Ax[S->fq[pp + a]]; else for (int a = 0; a < g; a++) pc[a] = 1.0;
                }
                const double one = 1.0;
                st_dsyrk("L", "N", &g, &nc, &one, pan, &g, &one, T, &rr);
            }
        }
        if (S->frun[j] != 0) { if (S->frhi[j]) pan1 = 1; else pan0 = 1; }      /* (frun < 0: inside a run that began earlier) */
        if (Ax) for (int a = 0; a < c; a++) x[a] = Ax[fq[a]]; else for (int a = 0; a < c; a++) x[a] = 1.0;
        if (!pan0) {
            if (g0) for (int b = 0; b < c1; b++) sk_add1(c1 - b, D + (size_t)(ps[0] + b) * r + (size_t)(ps[0] + b), x + b, wj * x[b]);
            else for (int b = 0; b < c1; b++) { const double t = wj * x[b]; double *col = D + (size_t)ps[b] * r; for (int a = b; a < c1; a++) col[ps[a]] += x[a] * t; }
        }
        if (c1 == c) return;
        const size_t r1 = (size_t)(S->rst[lo + 2] - S->rst[lo + 1]);
        double *D1 = v + S->oD[lo + 1], *C = v + S->oC[lo];
        if (g1 && c - c1 >= 8) {       /* (a short upper group goes through the plain loops below: a kernel call an entry costs more than the entry) */
            const int q0 = ps[c1], ch = c - c1; const double *xh = x + c1;
            if (!pan1) for (int b = 0; b < ch; b++) sk_add1(ch - b, D1 + (size_t)(q0 + b) * r1 + (size_t)(q0 + b), xh + b, wj * xh[b]);
            for (int b = 0; b < c1; b++) sk_add1(ch, C + (size_t)ps[b] * r1 + (size_t)q0, xh, wj * x[b]);
        } else {
            if (!pan1) for (int b = c1; b < c; b++) { const double t = wj * x[b]; double *col = D1 + (size_t)ps[b] * r1; for (int a = b; a < c; a++) col[ps[a]] += x[a] * t; }
            for (int b = 0; b < c1; b++) { const double t = wj * x[b]; double *col = C + (size_t)ps[b] * r1; for (int a = c1; a < c; a++) col[ps[a]] += x[a] * t; }
        }
    }
}
/* the blocks' entries from the columns (Ax NULL: every entry 1 and every weight 1, the pattern). A column with lower level
 * k writes the blocks D_k, C_k and D_{k+1} only, so the columns of the even levels can be taken level by level on the
 * threads, then those of the odd levels: each block receives its two levels' parts in that fixed order and each level's
 * columns in the order of their indices, whatever the number of threads (the same bits). */
static void sne_form(QpStageNe *S, const int *Ap, const double *Ax, const double *w) {
    const int L = S->L;
    /* (15 November 2026) on one thread no parallel construct is entered: a "parallel ... if (condition)" whose condition
     * fails still makes and releases a team of one thread in the runtime, about 2,000 instructions with an allocation,
     * at every factorization and every solve of a small MPC QP (the same in qp_stagene_factor and qp_stagene_solve) */
    int onth = S->wform > 2e6 && S->nth > 1;
#ifdef _OPENMP
    onth = onth && !omp_in_parallel();
#endif
    for (int par = 0; par < 2; par++) {
        if (onth) {
            #pragma omp parallel for schedule(dynamic, 1) num_threads(S->nth)
            for (int k = par; k < L; k += 2) {
                int th = 0;
#ifdef _OPENMP
                th = omp_get_thread_num(); if (th >= S->nth) th = 0;
#endif
                double *x = S->fx + (size_t)th * S->fxn, *pan = S->fpan + (size_t)th * S->fpn;
                for (int t = S->lst[k]; t < S->lst[k + 1]; t++) sne_form_col(S, Ap, Ax, w, S->lcol[t], x, pan);
            }
        } else for (int k = par; k < L; k += 2)
            for (int t = S->lst[k]; t < S->lst[k + 1]; t++) sne_form_col(S, Ap, Ax, w, S->lcol[t], S->fx, S->fpan);
    }
}

/* the Cholesky factorization of a diagonal block with the tiny-pivot rule (d0: the diagonal before any elimination) */
static void sne_potrf_safe(int n, double *A, const double *d0) { sk_potrf_blk(n, A, n, d0); }

/* one level's diagonal block factored: LAPACK first; the loop with the tiny-pivot rule when it stops or leaves a tiny pivot */
static void sne_block(QpStageNe *S, int k, double *wk) {
    const int r = S->rst[k + 1] - S->rst[k]; double *D = S->val + S->oD[k]; const double *d0 = S->d0 + S->rst[k];
    int redo = 1;
    if (r >= ST_POTRF) {
        memcpy(wk, D, sizeof(double) * (size_t)r * (size_t)r);
        const int info = sk_potrf_par(r, D, r); redo = info != 0;
        for (int i = 0; i < r && !redo; i++) { const double l = D[(size_t)i * r + i]; if (!(l * l > 1e-16 * d0[i])) redo = 1; }
        if (redo) memcpy(D, wk, sizeof(double) * (size_t)r * (size_t)r);
    }
    if (redo) sne_potrf_safe(r, D, d0);
}
/* The chain of levels is eliminated from its two ends towards the level c = S->mid: the levels 0 .. c-1 upwards
 * (D_k = L_k L_k', C_k := C_k L_k^-T, D_{k+1} -= C_k C_k') and the levels L-1 .. c+1 downwards, each with its coupling
 * block transposed in place first (T = C_{k-1}', then D_k = L_k L_k', T := T L_k^-T, D_{k-1} -= T T'), so that both ends
 * run the same two kernels; the two updates of D_c are applied after both, in that order, and D_c is factored last. The two halves touch no common entry, so two threads take one each: the arithmetic and its
 * order are the same on one thread, and the factor has the same bits at any thread count. The work is that of the
 * elimination from one end (each level is eliminated once, against one neighbour). */
static int g_sne_tdbg = 0; static double g_sne_t[3];      /* (BRISK_QPSTAGEDBG: the upward sweep's seconds in the blocks' factorizations, the solves with them, the updates) */
static void sne_lower(QpStageNe *S, int c, double *wk) {
    double *v = S->val;
    for (int k = 0; k < c; k++) {
        const int r = S->rst[k + 1] - S->rst[k], r1 = S->rst[k + 2] - S->rst[k + 1];
        const double t0_ = g_sne_tdbg ? st_now() : 0;
        sne_block(S, k, wk);
        const double t1_ = g_sne_tdbg ? st_now() : 0;
        sk_trsm_rlt_par(r1, r, v + S->oD[k], r, v + S->oC[k], r1, S->clb[k]);
        const double t2_ = g_sne_tdbg ? st_now() : 0;
        if (k + 1 < c) sk_syrk_ln_sub_par(r1, r, v + S->oC[k], r1, v + S->oD[k + 1], r1, S->clb[k]);
        if (g_sne_tdbg) { g_sne_t[0] += t1_ - t0_; g_sne_t[1] += t2_ - t1_; g_sne_t[2] += st_now() - t2_; }
    }
}
static void sne_upper(QpStageNe *S, int c, double *wk) {
    double *v = S->val;
    for (int k = S->L - 1; k > c; k--) {
        const int r = S->rst[k + 1] - S->rst[k], r0 = S->rst[k] - S->rst[k - 1];
        double *C = v + S->oC[k - 1];                     /* r x r0 as formed; r0 x r from here on */
        sne_block(S, k, wk);
        memcpy(wk, C, sizeof(double) * (size_t)r * (size_t)r0);
        for (int j = 0; j < r; j++) { double *cj = C + (size_t)j * r0; const double *wj = wk + j; for (int i = 0; i < r0; i++) cj[i] = wj[(size_t)i * r]; }
        sk_trsm_rlt_par(r0, r, v + S->oD[k], r, C, r0, S->cub[k - 1]);      /* (the transposed block's lower bandwidth is the block's upper one) */
        if (k - 1 > c) sk_syrk_ln_sub_par(r0, r, C, r0, v + S->oD[k - 1], r0, S->cub[k - 1]);
    }
}
/* the two ends on two threads: when both halves exist and the work pays for the hand-over. Inside that region the
 * level-3 steps of a wide level hand their fixed chunks to the team as tasks, which the threads without a half take (the
 * same chunks, the same bits) */
static int sne_two(const QpStageNe *S, double work) {
    return S->nth > 1 && S->mid > 0 && S->mid + 1 < S->L && work > 4e6 && !omp_in_parallel();
}

/* M = A diag(w) A' + diag(e) formed in the blocks and factored; always succeeds (the tiny-pivot rule) */
int qp_stagene_factor(QpStageNe *S, const int *Ap, const int *Ai, const double *Ax, const double *w, const double *e) {
    const int m = S->m, L = S->L, c = S->mid; double *v = S->val;
    memset(v, 0, sizeof(double) * S->nval);
    for (int i = 0; i < m; i++) { const int k = S->lev[i]; v[S->oD[k] + (size_t)S->pos[i] * (size_t)(S->rst[k + 1] - S->rst[k] + 1)] = e[i]; }
    (void)Ai;
    static int dbg = -1; if (dbg < 0) dbg = getenv("BRISK_QPSTAGEDBG") != NULL;      /* (a diagnosis: the factorization's time by part) */
    static double tfm = 0, tch = 0; static long ncall = 0; double tq = dbg ? st_now() : 0;
    sne_form(S, Ap, Ax, w);
    if (dbg) { tfm += st_now() - tq; tq = st_now(); }
    /* the diagonal before any elimination, for the tiny-pivot rule */
    for (int k = 0; k < L; k++) { const int r = S->rst[k + 1] - S->rst[k]; const double *D = v + S->oD[k]; double *d0 = S->d0 + S->rst[k];
        for (int i = 0; i < r; i++) d0[i] = D[(size_t)i * r + i]; }
    double *wk2 = S->wk + (size_t)S->wmax * (size_t)S->wmax;
    if (sne_two(S, S->wfact)) {
        #pragma omp parallel sections
        {
            #pragma omp section
            sne_lower(S, c, S->wk);
            #pragma omp section
            sne_upper(S, c, wk2);
        }
    } else { sne_lower(S, c, S->wk); sne_upper(S, c, wk2); }
    {   const int rc = S->rst[c + 1] - S->rst[c];
        if (c > 0) sk_syrk_ln_sub_par(rc, S->rst[c] - S->rst[c - 1], v + S->oC[c - 1], rc, v + S->oD[c], rc, S->clb[c - 1]);
        if (c + 1 < L) sk_syrk_ln_sub_par(rc, S->rst[c + 2] - S->rst[c + 1], v + S->oC[c], rc, v + S->oD[c], rc, S->cub[c]);
        sne_block(S, c, S->wk);
    }
    if (dbg) { tch += st_now() - tq; }
    g_sne_tdbg = dbg;
    if (dbg && ++ncall % 6 == 0) fprintf(stderr, "[stage factorizations so far (%ld): forming %.3f s, the elimination of the levels %.3f s; the upward sweep: the blocks' factorizations %.3f s, the solves %.3f s, the updates %.3f s]\n", ncall, tfm, tch, g_sne_t[0], g_sne_t[1], g_sne_t[2]);
    return 0;
}

/* y := M^-1 y (m): the two ends' forward substitutions (on two threads when the work pays), the meeting level, the two
 * ends' backward substitutions */
static void sne_fwd_lo(const QpStageNe *S, double *ys) {
    const int c = S->mid; const double *v = S->val; const int *rst = S->rst;
    for (int k = 0; k < c; k++) {
        const int r = rst[k + 1] - rst[k], r1 = rst[k + 2] - rst[k + 1];
        sk_trsv_n(r, v + S->oD[k], r, ys + rst[k]);
        if (k + 1 < c) sk_gemv_n(r1, r, -1.0, v + S->oC[k], r1, ys + rst[k], ys + rst[k + 1]);
    }
}
static void sne_fwd_hi(const QpStageNe *S, double *ys) {
    const int c = S->mid, L = S->L; const double *v = S->val; const int *rst = S->rst;
    for (int k = L - 1; k > c; k--) {
        const int r = rst[k + 1] - rst[k], r0 = rst[k] - rst[k - 1];
        sk_trsv_n(r, v + S->oD[k], r, ys + rst[k]);
        if (k - 1 > c) sk_gemv_n(r0, r, -1.0, v + S->oC[k - 1], r0, ys + rst[k], ys + rst[k - 1]);
    }
}
static void sne_meet(const QpStageNe *S, double *ys) {
    const int c = S->mid, L = S->L; const double *v = S->val; const int *rst = S->rst;
    const int rc = rst[c + 1] - rst[c];
    if (c > 0) sk_gemv_n(rc, rst[c] - rst[c - 1], -1.0, v + S->oC[c - 1], rc, ys + rst[c - 1], ys + rst[c]);
    if (c + 1 < L) sk_gemv_n(rc, rst[c + 2] - rst[c + 1], -1.0, v + S->oC[c], rc, ys + rst[c + 1], ys + rst[c]);
    sk_trsv_n(rc, v + S->oD[c], rc, ys + rst[c]);
    sk_trsv_t(rc, v + S->oD[c], rc, ys + rst[c]);
}
static void sne_bwd_lo(const QpStageNe *S, double *ys) {
    const int c = S->mid; const double *v = S->val; const int *rst = S->rst;
    for (int k = c - 1; k >= 0; k--) {
        const int r = rst[k + 1] - rst[k], r1 = rst[k + 2] - rst[k + 1];
        sk_gemv_t(r1, r, -1.0, v + S->oC[k], r1, ys + rst[k + 1], ys + rst[k]);
        sk_trsv_t(r, v + S->oD[k], r, ys + rst[k]);
    }
}
static void sne_bwd_hi(const QpStageNe *S, double *ys) {
    const int c = S->mid, L = S->L; const double *v = S->val; const int *rst = S->rst;
    for (int k = c + 1; k < L; k++) {
        const int r = rst[k + 1] - rst[k], r0 = rst[k] - rst[k - 1];
        sk_gemv_t(r0, r, -1.0, v + S->oC[k - 1], r0, ys + rst[k - 1], ys + rst[k]);
        sk_trsv_t(r, v + S->oD[k], r, ys + rst[k]);
    }
}
void qp_stagene_solve(QpStageNe *S, double *y) {
    const int m = S->m; double *ys = S->ys;
    for (int t = 0; t < m; t++) ys[t] = y[S->rord[t]];
    if (sne_two(S, S->wsolve)) {
        #pragma omp parallel
        {
            #pragma omp sections
            {
                #pragma omp section
                sne_fwd_lo(S, ys);
                #pragma omp section
                sne_fwd_hi(S, ys);
            }
            #pragma omp single
            sne_meet(S, ys);
            #pragma omp sections
            {
                #pragma omp section
                sne_bwd_lo(S, ys);
                #pragma omp section
                sne_bwd_hi(S, ys);
            }
        }
    } else { sne_fwd_lo(S, ys); sne_fwd_hi(S, ys); sne_meet(S, ys); sne_bwd_lo(S, ys); sne_bwd_hi(S, ys); }
    for (int t = 0; t < m; t++) y[S->rord[t]] = ys[t];
}
