/* The Weyl route on Hermitian blocks in their real form: the products in complex arithmetic of half the order
 * (23 October 2026).
 *
 * Quantum state tomography and related problems write a Hermitian d x d matrix (d = 2^q) in its real form
 * R(K) = [[Re K, -Im K], [Im K, Re K]] of order n = 2d, with constraints that are Pauli operators: in the real form
 * signed permutations A_a = wk X^wx Z^wz of Z_2^p (p = q + 1; the block's Weyl route, B->allweyl). When every
 * constraint and C commute with J = X_k Z_k for some index bit k (the bit that separates the real and imaginary parts),
 * each constraint is R(K_a) with K_a = kappa_a D_a, kappa_a = wk (i wk when wx and wz have the bit k), D_u = X^ux Z^uz
 * on the other q bits, and the iterates are R(Xc), R(Wc) (W = Z^-1, Xc and Wc Hermitian). Then
 *
 *   M_ab = tr(A_a X A_b W) = 2 Re tr(K_a Xc K_b Wc) = 2 wa wb Re[i^(ta + tb) tr(D_b Y_a)],   Y_a = Wc D_a Xc,
 *   tr(D_b Y) = sum_j (-1)^(bz . j) Y[j, j ^ bx]:
 *
 * the bx-th XOR diagonal of Y_a, transformed by Walsh-Hadamard, read at bz. D_a Xc is Xc with its rows permuted and
 * signed; Y_a is a complex product of order d, formed as three real products (Wr Br, Wi Bi, (Wr + Wi)(Br + Bi)): 6 d^3
 * flops a constraint against the real form's 2 n^3 = 16 d^3, and the transforms are of length d instead of n.
 * This file holds the detection and the split of the real form; the assembly is in solver.c (schur_weyl_pd).
 * BRISK_WEYLPD=0 keeps the real form's products.
 */
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "brisk.h"

/* Is the block's Weyl data the real form of Pauli constraints? Every constraint must commute with J = X_k Z_k at some
 * bit k of the index (wx and wz agree at k), and so must C: C[r ^ bit, c ^ bit] = s(r) s(c) C[r, c] with
 * s(i) = (-1)^(bit set in i), from J[i ^ bit, i] = (-1)^(i_k). Fills the packed Pauli index of each constraint on the
 * other bits (ax d + az), whether its factor carries i, and the bit k. Returns 1 when it applies. */
static int weylpd_detect(const Block *B, int *kbit, int *ap, unsigned char *ti) {
    const int n = B->n, ns = B->ns;
    if (!B->allweyl || ns < 1 || n < 4 || (n & (n - 1))) return 0;
    int p = 0; while ((1 << p) < n) p++;
    for (int k = p - 1; k >= 0; k--) {
        const int bit = 1 << k;
        int ok = 1;
        for (int b = 0; b < ns && ok; b++) if (((B->wx[b] & bit) != 0) != ((B->wz[b] & bit) != 0)) ok = 0;
        if (!ok) continue;
        const SpSym *C = &B->C;
        if (C->nnz > 0) {
            if (n > 4096) continue;
            double *Cd = calloc((size_t)n * n, sizeof(double));
            if (!Cd) return 0;
            for (int e = 0; e < C->nnz; e++) { Cd[C->row[e] + (size_t)C->col[e] * n] = C->val[e]; Cd[C->col[e] + (size_t)C->row[e] * n] = C->val[e]; }
            for (int e = 0; e < C->nnz && ok; e++) {
                const int r = C->row[e], c = C->col[e];
                const double v = C->val[e], sg = (((r & bit) != 0) ^ ((c & bit) != 0)) ? -1.0 : 1.0;
                if (fabs(Cd[(r ^ bit) + (size_t)(c ^ bit) * n] - sg * v) > 1e-14 * (1 + fabs(v))) ok = 0;
            }
            free(Cd);
        }
        if (!ok) continue;
        const int lo = bit - 1;
        for (int b = 0; b < ns; b++) {
            const int wx = B->wx[b], wz = B->wz[b];
            const int ax = (wx & lo) | ((wx >> 1) & ~lo), az = (wz & lo) | ((wz >> 1) & ~lo);   /* bit k removed */
            ap[b] = ax * (n / 2) + az;
            ti[b] = (wx & bit) ? 1 : 0;
        }
        *kbit = k;
        return 1;
    }
    return 0;
}

void weylpd_free(Block *B) {
    free(B->wpd_ap); free(B->wpd_t);
    B->wpd_ap = NULL; B->wpd_t = NULL; B->wpd = 0;
}

/* After the Weyl route was found (problem.c): the complex form, unless BRISK_WEYLPD=0 */
int weylpd_setup(Block *B) {
    weylpd_free(B);
    const char *e = getenv("BRISK_WEYLPD");
    if ((e && atoi(e) == 0) || !B->allweyl) return 0;
    const int ns = B->ns;
    int *ap = malloc(sizeof(int) * ns); unsigned char *ti = malloc(ns); int k = -1;
    if (!ap || !ti || !weylpd_detect(B, &k, ap, ti)) { free(ap); free(ti); return 0; }
    if (getenv("BRISK_WEYLDBG")) printf("   [Weyl route: the real form of a Hermitian block of order %d (bit %d), %d constraints: complex products]\n", B->n / 2, k, ns);
    B->wpd = 1; B->wpd_k = k; B->wpd_ap = ap; B->wpd_t = ti;
    return 1;
}

/* The complex matrix of the real-form matrix A (n x n, column major) with the complex bit k, projected on the
 * J-invariant matrices: Ac = (A11 + A22)/2 + i (A21 - A12)/2 (column major, d = n / 2). */
void weylpd_split(const double *A, int n, int kbit, double *re, double *im) {
    const int d = n / 2, bit = 1 << kbit, lo = bit - 1;
    for (int c = 0; c < d; c++) {
        const int c0 = (c & lo) | ((c & ~lo) << 1), c1 = c0 | bit;      /* the real index: bit k inserted */
        for (int r = 0; r < d; r++) {
            const int r0 = (r & lo) | ((r & ~lo) << 1), r1 = r0 | bit;
            const double a11 = A[r0 + (size_t)c0 * n], a22 = A[r1 + (size_t)c1 * n];
            const double a21 = A[r1 + (size_t)c0 * n], a12 = A[r0 + (size_t)c1 * n];
            re[r + (size_t)c * d] = 0.5 * (a11 + a22); im[r + (size_t)c * d] = 0.5 * (a21 - a12);
        }
    }
}
