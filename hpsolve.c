/* SPDX-License-Identifier: Apache-2.0
 * hpsolve.c (5.0): high-precision solves. The problem is read as exact decimals into mpx at
 * the target precision (the "master" data); the interior-point method of hpipm.inc then runs
 * up a ladder of precisions (double, double-double, quad-double, mpx), each level starting
 * from the iterate of the one below and running only while it makes healthy progress.
 * The presolve (hppre.inc), the reductions of the double pipeline (hpred.inc) and the dual
 * form are applied in the master precision; the errors are measured in the working
 * precision on the data as read.
 */
#include "brisk.h"
#include <stdarg.h>
#include "hp.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <fenv.h>
#include <float.h>

struct HPChamp { int have; uint64_t **X, **Z, *y, *pobj, *dobj; double m[5]; };
#define HP_NANC 6
typedef struct {
    /* structure */
    int m, nblk, *bs;
    size_t nnzA, nnzC;
    int *ncon, **con; size_t **ptr;        /* per block: constraints present, entry ranges */
    int *ei, *ej;                          /* entries of the A_p (i <= j, 0-based), by block, then constraint */
    size_t *cptr; int *ci, *cj;            /* entries of C by block */
    int **lpptr, **lpe, **lpc;             /* LP blocks: per variable, entry indices and constraints */
    int **kord, **kcolptr, **kcol;         /* SDP blocks: constraints by decreasing column count, their columns */
    /* master data and exchange iterate: mpx with Lm limbs */
    int Lm;
    uint64_t *Av, *Cv, *b;
    uint64_t **X, **Z, *y, *pobj, *dobj;
    int have_iter;
    /* control */
    double tol, level_tol, timelimit, tstart;
    int chol_exit;                         /* the level ended because the Schur complement was not positive definite in its precision */
    int more, sick_exit;                   /* a higher precision can follow this level; the level handed over an earlier iterate */
    double level_best;                     /* the best error of the last level (best_err is that of the iterate handed over) */
    double ray_res;                        /* an infeasibility verdict: the relative residual of the iterate's improving direction */
    double accept;                         /* the error at which the level's end counts as reached (no handover below it) */
    double sick;                           /* > 0: a level ends once <X,Z> is below sick * error twice in a row (off) */
    int maxit, verbose;
    double fast;                           /* the fast end: 1 - gamma = fast * rho (0: off) */
    int method, cgmax;                     /* 0 interior point, 1 ALM / semismooth Newton, 2 low rank */
    double sigma, sigfac;                  /* the ALM penalty (carried between levels), its growth factor */
    /* results */
    int iters, nbis;
    int ncen;             /* centring steps after a fast step */
    /* low rank (hplr.inc): the union pattern of C and the A_p per block (slots i <= j, every
     * diagonal included), the slot of each entry, the pattern by rows, the factor R */
    int *lr_np, **lr_pi, **lr_pj, *lr_es, *lr_cs, **lr_rp, **lr_rc, **lr_rs, **lr_diag;
    uint64_t **lr_R; int *lr_rk, *lr_cap, lr_have;
    double lr_delta;
    double pivreg;        /* Schur pivots below eps^pivreg (relative) are skipped; 0 = never */
    int *mfst;            /* first column of each row of the Schur complement by its structure (NULL: not known to be narrow) */
    int *mpos;            /* row of each constraint in the Schur complement (NULL: as numbered); dep, the factor and its solves are in rows */
    int *dep, dep_known;  /* rows found linearly dependent (left out of the Schur complement) */
    int deponly;          /* the interior-point routine stops after its first factorization (finds the dependent rows) */
    double minpiv;        /* smallest pivot / diagonal entry of the Schur factorizations of a level */
    int ndyn, nwould;     /* rows skipped by the pivot threshold in a level; factorizations where it would have skipped */
    double pinf, dinf, gap, compl, err, best_pinf, best_dinf, best_gap, best_compl, best_err;
    struct HPChamp *anc; int na; double anc_thr;   /* -bound: anchors of the certificate (earlier central iterates), their number, the next error */
    struct HPChamp *spare;                 /* the handover iterate of a level that handed over its best one */
    struct HPChamp *champ;                 /* the best iterate of a level that handed over another one */
} HPJob;

/* ------------------------------------------------------------------ the four instances */
#define HPWMAX 1
#define HPW 1
#define HPT double
#define HPN(x) x##_f64
#define HP_NAME "d"
#define HP_EPS 1.1e-16
#define HP_SETD(r, d) (*(r) = (d))
#define HP_GETD(a) (*(a))
#define HP_COPY(r, a) (*(r) = *(a))
#define HP_ADD(r, a, b) (*(r) = *(a) + *(b))
#define HP_SUB(r, a, b) (*(r) = *(a) - *(b))
#define HP_MUL(r, a, b) (*(r) = *(a) * *(b))
#define HP_DIV(r, a, b) (*(r) = *(a) / *(b))
#define HP_SQRT(r, a) (*(r) = sqrt(*(a)))
#define HP_NEG(r, a) (*(r) = -*(a))
#define HP_SGN(a) (*(a) > 0 ? 1 : *(a) < 0 ? -1 : 0)
#define HP_ADDMUL(r, a, b) (*(r) += *(a) * *(b))
#define HP_SUBMUL(r, a, b) (*(r) -= *(a) * *(b))
#define HP_IMPORT(r, s, J) (*(r) = mpx_getd(s))
#define HP_EXPORT(d, a, J) mpx_setd(d, *(a))
#define HP_OWN_KERNELS
static void axpy_f64(size_t n, const double *a, const double *x, double *y) {
    const double al = *a;
#pragma omp simd
    for (size_t i = 0; i < n; i++) y[i] += al * x[i];
}
static void dot_f64(size_t n, const double *x, const double *y, double *r) {
    double s = 0;
#pragma omp simd reduction(+:s)
    for (size_t i = 0; i < n; i++) s += x[i] * y[i];
    *r = s;
}
#include "hpipm.inc"
#include "hpfom.inc"
#include "hpalg.inc"
#include "hplr.inc"
#undef HP_OWN_KERNELS
#undef HPT
#undef HPN
#undef HP_NAME
#undef HP_EPS
#undef HP_SETD
#undef HP_GETD
#undef HP_COPY
#undef HP_ADD
#undef HP_SUB
#undef HP_MUL
#undef HP_DIV
#undef HP_SQRT
#undef HP_NEG
#undef HP_SGN
#undef HP_ADDMUL
#undef HP_SUBMUL
#undef HP_IMPORT
#undef HP_EXPORT

#define HPT dd_t
#define HPN(x) x##_dd
#define HP_NAME "dd"
#define HP_EPS 1.3e-32
#define HP_SETD(r, d) (*(r) = dd_set(d))
#define HP_GETD(a) dd_get(*(a))
#define HP_COPY(r, a) (*(r) = *(a))
#define HP_ADD(r, a, b) (*(r) = dd_add(*(a), *(b)))
#define HP_SUB(r, a, b) (*(r) = dd_sub(*(a), *(b)))
#define HP_MUL(r, a, b) (*(r) = dd_mul(*(a), *(b)))
#define HP_DIV(r, a, b) (*(r) = dd_div(*(a), *(b)))
#define HP_SQRT(r, a) (*(r) = dd_sqrt(*(a)))
#define HP_NEG(r, a) (*(r) = dd_neg(*(a)))
#define HP_SGN(a) ((a)->h > 0 ? 1 : (a)->h < 0 ? -1 : 0)
#define HP_ADDMUL(r, a, b) (*(r) = dd_add(*(r), dd_mul(*(a), *(b))))
#define HP_SUBMUL(r, a, b) (*(r) = dd_sub(*(r), dd_mul(*(a), *(b))))
#define HP_IMPORT(r, s, J) (*(r) = mpx_get_dd(s))
#define HP_EXPORT(d, a, J) mpx_set_dd(d, *(a))
#define HP_OWN_KERNELS
/* The two loops that carry the double-double solve, written so that the compiler turns them
 * into SIMD code (8 lanes with AVX-512: 0.6-0.8 ns per element against 7 for the scalar dot).
 * Both use the "sloppy" sum (error relative to |a| + |b|), which is the standard model of
 * arithmetic without a guard digit and is enough for the factorizations and products. */
#define DD_NL 8
static void axpy_dd(size_t n, const dd_t *a, const dd_t *x, dd_t *y) {
    const double ah = a->h, al = a->l;
#pragma omp simd
    for (size_t i = 0; i < n; i++) {
        const double xh = x[i].h, xl = x[i].l, yh = y[i].h;
        const double p = ah * xh, e = fma(ah, xh, -p) + (ah * xl + al * xh);
        const double s = yh + p, bb = s - yh;
        const double t = ((yh - (s - bb)) + (p - bb)) + (y[i].l + e);
        const double h = s + t;
        y[i].l = t - (h - s); y[i].h = h;
    }
}
static void dot_dd(size_t n, const dd_t *x, const dd_t *y, dd_t *r) {
    double sh[DD_NL] = { 0 }, sl[DD_NL] = { 0 };
    size_t i = 0;
    for (; i + DD_NL <= n; i += DD_NL) {
#pragma omp simd
        for (int l = 0; l < DD_NL; l++) {
            const double ah = x[i + l].h, al = x[i + l].l, bh = y[i + l].h, bl = y[i + l].l;
            const double p = ah * bh, e = fma(ah, bh, -p) + (ah * bl + al * bh);
            const double s = sh[l] + p, bb = s - sh[l];
            sl[l] += ((sh[l] - (s - bb)) + (p - bb)) + e;
            sh[l] = s;
        }
    }
    if (i < n) {
        /* the remainder as one more vector step on zero-padded copies (5.3: the scalar tail
         * and the scalar sum of the lanes were 60 ns per call, half the time of the dots of
         * a Cholesky factorization of order 100-400) */
        dd_t xb[DD_NL], yb[DD_NL];
        const size_t rem = n - i;
        for (size_t l = 0; l < DD_NL; l++) { if (l < rem) { xb[l] = x[i + l]; yb[l] = y[i + l]; } else { xb[l].h = xb[l].l = 0; yb[l].h = yb[l].l = 0; } }
#pragma omp simd
        for (int l = 0; l < DD_NL; l++) {
            const double ah = xb[l].h, al = xb[l].l, bh = yb[l].h, bl = yb[l].l;
            const double p = ah * bh, e = fma(ah, bh, -p) + (ah * bl + al * bh);
            const double s = sh[l] + p, bb = s - sh[l];
            sl[l] += ((sh[l] - (s - bb)) + (p - bb)) + e;
            sh[l] = s;
        }
    }
    /* the lanes: sloppy pairwise sums 8 -> 4 -> 2 -> 1, one renormalisation */
    for (int w = DD_NL / 2; w >= 1; w /= 2)
        for (int l = 0; l < w; l++) { const double a = sh[l], b = sh[l + w]; const double s = a + b, bb = s - a; sl[l] = ((a - (s - bb)) + (b - bb)) + (sl[l] + sl[l + w]); sh[l] = s; }
    r->h = hp_qts(sh[0], sl[0], &r->l);
}
#define HP_OWN_ROT
/* (x, y) <- (c x - s y, s x + c y): the Jacobi rotation of the eigensolver */
static void rot_dd(size_t n, const dd_t *c, const dd_t *s, dd_t *x, dd_t *y) {
    const double ch = c->h, cl = c->l, sh = s->h, sl = s->l;
#pragma omp simd
    for (size_t i = 0; i < n; i++) {
        const double xh = x[i].h, xl = x[i].l, yh = y[i].h, yl = y[i].l;
        const double p1 = ch * xh, e1 = fma(ch, xh, -p1) + (ch * xl + cl * xh);      /* c x */
        const double p2 = sh * yh, e2 = fma(sh, yh, -p2) + (sh * yl + sl * yh);      /* s y */
        const double p3 = sh * xh, e3 = fma(sh, xh, -p3) + (sh * xl + sl * xh);      /* s x */
        const double p4 = ch * yh, e4 = fma(ch, yh, -p4) + (ch * yl + cl * yh);      /* c y */
        double u = p1 - p2, bb = u - p1, t = ((p1 - (u - bb)) - (p2 + bb)) + (e1 - e2), h = u + t;
        x[i].l = t - (h - u); x[i].h = h;
        u = p3 + p4; bb = u - p3; t = ((p3 - (u - bb)) + (p4 - bb)) + (e3 + e4); h = u + t;
        y[i].l = t - (h - u); y[i].h = h;
    }
}
#include "hpipm.inc"
#include "hpfom.inc"
#include "hpalg.inc"
#include "hplr.inc"
#undef HP_OWN_KERNELS
#undef HP_OWN_ROT
#undef HPT
#undef HPN
#undef HP_NAME
#undef HP_EPS
#undef HP_SETD
#undef HP_GETD
#undef HP_COPY
#undef HP_ADD
#undef HP_SUB
#undef HP_MUL
#undef HP_DIV
#undef HP_SQRT
#undef HP_NEG
#undef HP_SGN
#undef HP_ADDMUL
#undef HP_SUBMUL
#undef HP_IMPORT
#undef HP_EXPORT

#define HPT qd_t
#define HPN(x) x##_qd
#define HP_NAME "qd"
#define HP_EPS 1.3e-64
#define HP_SETD(r, d) (*(r) = qd_set(d))
#define HP_GETD(a) qd_get(*(a))
#define HP_COPY(r, a) (*(r) = *(a))
#define HP_ADD(r, a, b) (*(r) = qd_add(*(a), *(b)))
#define HP_SUB(r, a, b) (*(r) = qd_sub(*(a), *(b)))
#define HP_MUL(r, a, b) (*(r) = qd_mul(*(a), *(b)))
#define HP_DIV(r, a, b) (*(r) = qd_div(*(a), *(b)))
#define HP_SQRT(r, a) (*(r) = qd_sqrt(*(a)))
#define HP_NEG(r, a) (*(r) = qd_neg(*(a)))
#define HP_SGN(a) ((a)->x[0] > 0 ? 1 : (a)->x[0] < 0 ? -1 : 0)
#define HP_ADDMUL(r, a, b) (*(r) = qd_add(*(r), qd_mul(*(a), *(b))))
#define HP_SUBMUL(r, a, b) (*(r) = qd_sub(*(r), qd_mul(*(a), *(b))))
#define HP_IMPORT(r, s, J) (*(r) = mpx_get_qd(s))
#define HP_EXPORT(d, a, J) mpx_set_qd(d, *(a))
#define HP_OWN_KERNELS
#define HP_OWN_ROT
/* Quad-double kernels without branches, so that the compiler makes SIMD code of them: the
 * renormalisation is two passes of quick-two-sums (up, then down) instead of the tests on
 * zero components. A component that is exactly zero in the middle then costs a few bits in
 * that one number, which the sums of a dot product or a factorization do not see. */
#define QTS(s, e, a, b) do { const double a_ = (a), b_ = (b); const double s_ = a_ + b_, bb_ = s_ - a_; (e) = (a_ - (s_ - bb_)) + (b_ - bb_); (s) = s_; } while (0)
#define QTP(p, e, a, b) do { const double a_ = (a), b_ = (b); const double p_ = a_ * b_; (e) = fma(a_, b_, -p_); (p) = p_; } while (0)
#define Q3S(a, b, c) do { double t1_, t2_, t3_; QTS(t1_, t2_, a, b); QTS(a, t3_, c, t1_); QTS(b, c, t2_, t3_); } while (0)
#define Q3S2(a, b, c) do { double t1_, t2_, t3_; QTS(t1_, t2_, a, b); QTS(a, t3_, c, t1_); (b) = t2_ + t3_; } while (0)
#define QRN5(c0, c1, c2, c3, c4, r0, r1, r2, r3) do { \
    double u0_ = (c0), u1_ = (c1), u2_ = (c2), u3_ = (c3), u4_ = (c4), s_, e_; \
    s_ = u3_ + u4_; u4_ = u4_ - (s_ - u3_); u3_ = s_; \
    s_ = u2_ + u3_; u3_ = u3_ - (s_ - u2_); u2_ = s_; \
    s_ = u1_ + u2_; u2_ = u2_ - (s_ - u1_); u1_ = s_; \
    s_ = u0_ + u1_; u1_ = u1_ - (s_ - u0_); u0_ = s_; \
    s_ = u1_ + u2_; e_ = u2_ - (s_ - u1_); u1_ = s_; u2_ = e_; \
    s_ = u2_ + u3_; e_ = u3_ - (s_ - u2_); u2_ = s_; u3_ = e_; \
    (r0) = u0_; (r1) = u1_; (r2) = u2_; (r3) = u3_ + u4_; } while (0)
/* (r0..r3) = (a0..a3) + (b0..b3) */
#define QADD(a0, a1, a2, a3, b0, b1, b2, b3, r0, r1, r2, r3) do { \
    double s0, s1, s2, s3, t0, t1, t2, t3; \
    QTS(s0, t0, a0, b0); QTS(s1, t1, a1, b1); QTS(s2, t2, a2, b2); QTS(s3, t3, a3, b3); \
    QTS(s1, t0, s1, t0); Q3S(s2, t0, t1); Q3S2(s3, t0, t2); t0 = t0 + t1 + t3; \
    QRN5(s0, s1, s2, s3, t0, r0, r1, r2, r3); } while (0)
/* (r0..r3) = (a0..a3) * (b0..b3) */
#define QMUL(a0, a1, a2, a3, b0, b1, b2, b3, r0, r1, r2, r3) do { \
    double p0, p1, p2, p3, p4, p5, q0, q1, q2, q3, q4, q5, t0, t1, s0, s1, s2; \
    QTP(p0, q0, a0, b0); QTP(p1, q1, a0, b1); QTP(p2, q2, a1, b0); \
    QTP(p3, q3, a0, b2); QTP(p4, q4, a1, b1); QTP(p5, q5, a2, b0); \
    Q3S(p1, p2, q0); Q3S(p2, q1, q2); Q3S(p3, p4, p5); \
    QTS(s0, t0, p2, p3); QTS(s1, t1, q1, p4); s2 = q2 + p5; \
    QTS(s1, t0, s1, t0); s2 += (t0 + t1); \
    s1 += (a0) * (b3) + (a1) * (b2) + (a2) * (b1) + (a3) * (b0) + q0 + q3 + q4 + q5; \
    QRN5(p0, p1, s0, s1, s2, r0, r1, r2, r3); } while (0)
#define QD_NL 8
static void axpy_qd(size_t n, const qd_t *a, const qd_t *x, qd_t *y) {
    const double a0 = a->x[0], a1 = a->x[1], a2 = a->x[2], a3 = a->x[3];
#pragma omp simd
    for (size_t i = 0; i < n; i++) {
        const double x0 = x[i].x[0], x1 = x[i].x[1], x2 = x[i].x[2], x3 = x[i].x[3];
        const double y0 = y[i].x[0], y1 = y[i].x[1], y2 = y[i].x[2], y3 = y[i].x[3];
        double m0, m1, m2, m3, r0, r1, r2, r3;
        QMUL(a0, a1, a2, a3, x0, x1, x2, x3, m0, m1, m2, m3);
        QADD(y0, y1, y2, y3, m0, m1, m2, m3, r0, r1, r2, r3);
        y[i].x[0] = r0; y[i].x[1] = r1; y[i].x[2] = r2; y[i].x[3] = r3;
    }
}
static void dot_qd(size_t n, const qd_t *x, const qd_t *y, qd_t *res) {
    double c0[QD_NL] = { 0 }, c1[QD_NL] = { 0 }, c2[QD_NL] = { 0 }, c3[QD_NL] = { 0 };
    size_t i = 0;
    for (; i + QD_NL <= n; i += QD_NL) {
#pragma omp simd
        for (int l = 0; l < QD_NL; l++) {
            const double x0 = x[i + l].x[0], x1 = x[i + l].x[1], x2 = x[i + l].x[2], x3 = x[i + l].x[3];
            const double y0 = y[i + l].x[0], y1 = y[i + l].x[1], y2 = y[i + l].x[2], y3 = y[i + l].x[3];
            double m0, m1, m2, m3, r0, r1, r2, r3;
            QMUL(x0, x1, x2, x3, y0, y1, y2, y3, m0, m1, m2, m3);
            QADD(c0[l], c1[l], c2[l], c3[l], m0, m1, m2, m3, r0, r1, r2, r3);
            c0[l] = r0; c1[l] = r1; c2[l] = r2; c3[l] = r3;
        }
    }
    if (i < n) {
        /* the remainder as one more vector step on zero-padded copies (5.3) */
        qd_t xb[QD_NL], yb[QD_NL];
        const size_t rem = n - i;
        for (size_t l = 0; l < QD_NL; l++) { if (l < rem) { xb[l] = x[i + l]; yb[l] = y[i + l]; } else { xb[l] = qd_set(0.0); yb[l] = qd_set(0.0); } }
#pragma omp simd
        for (int l = 0; l < QD_NL; l++) {
            const double x0 = xb[l].x[0], x1 = xb[l].x[1], x2 = xb[l].x[2], x3 = xb[l].x[3];
            const double y0 = yb[l].x[0], y1 = yb[l].x[1], y2 = yb[l].x[2], y3 = yb[l].x[3];
            double m0, m1, m2, m3, r0, r1, r2, r3;
            QMUL(x0, x1, x2, x3, y0, y1, y2, y3, m0, m1, m2, m3);
            QADD(c0[l], c1[l], c2[l], c3[l], m0, m1, m2, m3, r0, r1, r2, r3);
            c0[l] = r0; c1[l] = r1; c2[l] = r2; c3[l] = r3;
        }
    }
    /* the lanes: pairwise sums 8 -> 4 -> 2 -> 1 with the same vector addition, one
     * renormalisation (5.3: eight scalar renormalisations and additions were most of the cost of
     * the short dots of the Schur assembly) */
    for (int w = QD_NL / 2; w >= 1; w /= 2) {
#pragma omp simd
        for (int l = 0; l < w; l++) {
            double r0, r1, r2, r3;
            QADD(c0[l], c1[l], c2[l], c3[l], c0[l + w], c1[l + w], c2[l + w], c3[l + w], r0, r1, r2, r3);
            c0[l] = r0; c1[l] = r1; c2[l] = r2; c3[l] = r3;
        }
    }
    *res = qd_renorm5(c0[0], c1[0], c2[0], c3[0], 0.0);
}
static void rot_qd(size_t n, const qd_t *c, const qd_t *s, qd_t *x, qd_t *y) {
    const double c0 = c->x[0], c1 = c->x[1], c2 = c->x[2], c3 = c->x[3];
    const double s0_ = s->x[0], s1_ = s->x[1], s2_ = s->x[2], s3_ = s->x[3];
#pragma omp simd
    for (size_t i = 0; i < n; i++) {
        const double x0 = x[i].x[0], x1 = x[i].x[1], x2 = x[i].x[2], x3 = x[i].x[3];
        const double y0 = y[i].x[0], y1 = y[i].x[1], y2 = y[i].x[2], y3 = y[i].x[3];
        double a0, a1, a2, a3, b0, b1, b2, b3, r0, r1, r2, r3;
        QMUL(c0, c1, c2, c3, x0, x1, x2, x3, a0, a1, a2, a3);
        QMUL(s0_, s1_, s2_, s3_, y0, y1, y2, y3, b0, b1, b2, b3);
        QADD(a0, a1, a2, a3, -b0, -b1, -b2, -b3, r0, r1, r2, r3);
        x[i].x[0] = r0; x[i].x[1] = r1; x[i].x[2] = r2; x[i].x[3] = r3;
        QMUL(s0_, s1_, s2_, s3_, x0, x1, x2, x3, a0, a1, a2, a3);
        QMUL(c0, c1, c2, c3, y0, y1, y2, y3, b0, b1, b2, b3);
        QADD(a0, a1, a2, a3, b0, b1, b2, b3, r0, r1, r2, r3);
        y[i].x[0] = r0; y[i].x[1] = r1; y[i].x[2] = r2; y[i].x[3] = r3;
    }
}
#include "hpipm.inc"
#include "hpfom.inc"
#include "hpalg.inc"
#include "hplr.inc"
#undef HP_OWN_KERNELS
#undef HP_OWN_ROT
#undef HPT
#undef HPN
#undef HP_NAME
#undef HP_EPS
#undef HP_SETD
#undef HP_GETD
#undef HP_COPY
#undef HP_ADD
#undef HP_SUB
#undef HP_MUL
#undef HP_DIV
#undef HP_SQRT
#undef HP_NEG
#undef HP_SGN
#undef HP_ADDMUL
#undef HP_SUBMUL
#undef HP_IMPORT
#undef HP_EXPORT
#undef HPW
#undef HPWMAX

#define HPWMAX (MPX_MAXL + 2)
#define HPW mpx_W
#define HPT uint64_t
#define HPN(x) x##_mp
/* own kernels (5.3): the dot product accumulates its terms unnormalised (hpmp.c: mpx_dot),
 * and the matrix product is made of dot products over a transposed copy of B */
#define HP_OWN_KERNELS
#define HP_OWN_GEMM
static void axpy_mp(size_t n, const uint64_t *a, const uint64_t *x, uint64_t *y) {
    const size_t W = (size_t)mpx_W;
    for (size_t i = 0; i < n; i++) mpx_addmul(y + i * W, a, x + i * W);
}
static void dot_mp(size_t n, const uint64_t *x, const uint64_t *y, uint64_t *r) { mpx_dot(n, x, y, r); }
static void gemm_mp(int n, const uint64_t *A, const uint64_t *B, uint64_t *C) {
    const size_t W = (size_t)mpx_W;
    uint64_t *Bt = (uint64_t *)malloc(sizeof(uint64_t) * ((size_t)n * n + 1) * W);
    if (!Bt) { fprintf(stderr, "brisk: out of memory (high precision)\n"); exit(1); }
    for (int i = 0; i < n; i++) for (int j = 0; j < n; j++) memcpy(Bt + ((size_t)j * n + i) * W, B + ((size_t)i * n + j) * W, W * sizeof(uint64_t));
#pragma omp parallel for schedule(static) if (n >= 24)
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++) mpx_dot((size_t)n, A + (size_t)i * n * W, Bt + (size_t)j * n * W, C + ((size_t)i * n + j) * W);
    free(Bt);
}
#define HP_NAME "mp"
#define HP_EPS ldexp(1.0, -mpx_bits())
#define HP_SETD(r, d) mpx_setd(r, d)
#define HP_GETD(a) mpx_getd(a)
#define HP_COPY(r, a) mpx_copy(r, a)
#define HP_ADD(r, a, b) mpx_add(r, a, b)
#define HP_SUB(r, a, b) mpx_sub(r, a, b)
#define HP_MUL(r, a, b) mpx_mul(r, a, b)
#define HP_DIV(r, a, b) mpx_div(r, a, b)
#define HP_SQRT(r, a) mpx_sqrt(r, a)
#define HP_NEG(r, a) mpx_neg(r, a)
#define HP_SGN(a) mpx_sgn(a)
#define HP_ADDMUL(r, a, b) mpx_addmul(r, a, b)
#define HP_SUBMUL(r, a, b) mpx_submul(r, a, b)
#define HP_IMPORT(r, s, J) mpx_conv(r, mpx_L, s, (J)->Lm)
#define HP_EXPORT(d, a, J) mpx_conv(d, (J)->Lm, a, mpx_L)
#include "hpipm.inc"
#include "hpfom.inc"
#include "hpalg.inc"
#include "hplr.inc"
#undef HP_OWN_KERNELS
#undef HP_OWN_GEMM

/* ------------------------------------------------------------------ reading */
typedef struct { int con, blk, i, j; size_t idx; } HEnt;
static int cmp_hent(const void *a, const void *b) {
    const HEnt *x = (const HEnt *)a, *y = (const HEnt *)b;
    if (x->blk != y->blk) return x->blk < y->blk ? -1 : 1;
    if (x->con != y->con) return x->con < y->con ? -1 : 1;
    if (x->i != y->i) return x->i < y->i ? -1 : 1;
    return x->j < y->j ? -1 : x->j > y->j;
}
static void hp_fail(const char *msg) { fprintf(stderr, "brisk: high precision: %s\n", msg); }

/* builds the grouped structure of J from the entry list (con = -1 for C) */
/* Order of the constraints in the Schur complement (5.3).  Two constraints are coupled when
 * they share an SDP block or an LP variable, so with many small blocks (sparse polynomial
 * optimization) the matrix is sparse, and the envelope Cholesky factorization of the template
 * (cholr) costs the sum of the squared row widths.  A breadth-first order from a far vertex,
 * reversed (reverse Cuthill-McKee without the degree sort), is taken when its envelope is less
 * than half that of the order as read; otherwise NULL (nothing changes).
 * Returns pos[p], the row of constraint p. */
static double hp_envelope(const HPJob *J, const int *pos, int **rowfst) {
    const int m = J->m;
    int *fst = (int *)malloc(sizeof(int) * (size_t)(m + 1));
    for (int p = 0; p < m; p++) fst[p] = pos ? pos[p] : p;
#define HP_CLQ(cnt, mem) do { int mn = m; for (int a_ = 0; a_ < (cnt); a_++) { const int r_ = pos ? pos[mem] : (mem); if (r_ < mn) mn = r_; } \
        for (int a_ = 0; a_ < (cnt); a_++) if (mn < fst[mem]) fst[mem] = mn; } while (0)
    for (int k = 0; k < J->nblk; k++) {
        if (J->bs[k] < 0) { for (int i = 0; i < -J->bs[k]; i++) HP_CLQ(J->lpptr[k][i + 1] - J->lpptr[k][i], J->lpc[k][J->lpptr[k][i] + a_]); }
        else HP_CLQ(J->ncon[k], J->con[k][a_]);
    }
#undef HP_CLQ
    double env = 0.0;
    for (int p = 0; p < m; p++) env += (double)((pos ? pos[p] : p) - fst[p]);
    if (rowfst) {                                                 /* first column of every row */
        int *rf = (int *)malloc(sizeof(int) * (size_t)(m + 1));
        for (int p = 0; p < m; p++) rf[pos ? pos[p] : p] = fst[p];
        *rowfst = rf;
    }
    free(fst);
    return env;
}
static int *hp_schur_order(HPJob *J) {
    const int m = J->m;
    J->mfst = NULL;
    if (m < 64 || getenv("BRISK_HPNOORDER")) return NULL;
    const double env0 = hp_envelope(J, NULL, NULL);
    if (env0 < 0.05 * (double)m * m) { (void)hp_envelope(J, NULL, &J->mfst); return NULL; }   /* narrow as it is */
    /* cliques: SDP blocks and LP variables; members by clique, cliques by constraint */
    int ncl = 0; size_t tot = 0;
    for (int k = 0; k < J->nblk; k++) {
        if (J->bs[k] < 0) { ncl += -J->bs[k]; tot += (size_t)J->lpptr[k][-J->bs[k]]; }
        else { ncl++; tot += (size_t)J->ncon[k]; }
    }
    size_t *cp = (size_t *)calloc((size_t)ncl + 2, sizeof(size_t)), *vp = (size_t *)calloc((size_t)m + 2, sizeof(size_t));
    int *cm = (int *)malloc(sizeof(int) * (tot + 1)), *vc = (int *)malloc(sizeof(int) * (tot + 1));
    int c = 0; size_t w = 0;
    for (int k = 0; k < J->nblk; k++) {
        if (J->bs[k] < 0) for (int i = 0; i < -J->bs[k]; i++) { cp[c++] = w; for (int a = J->lpptr[k][i]; a < J->lpptr[k][i + 1]; a++) cm[w++] = J->lpc[k][a]; }
        else { cp[c++] = w; for (int a = 0; a < J->ncon[k]; a++) cm[w++] = J->con[k][a]; }
    }
    cp[ncl] = w;
    for (size_t q = 0; q < tot; q++) vp[cm[q] + 2]++;
    for (int p = 0; p < m; p++) vp[p + 2] += vp[p + 1];
    for (int q = 0; q < ncl; q++) for (size_t a = cp[q]; a < cp[q + 1]; a++) vc[vp[cm[a] + 1]++] = q;
    int *ord = (int *)malloc(sizeof(int) * (size_t)(m + 1)), *pos = (int *)malloc(sizeof(int) * (size_t)(m + 1));
    char *seen = (char *)malloc((size_t)m + 1), *cdone = (char *)malloc((size_t)ncl + 1);
    int start = 0;
    for (int pass = 0; pass < 3; pass++) {                        /* twice to find a far vertex, then the order */
        memset(seen, 0, (size_t)m); memset(cdone, 0, (size_t)ncl);
        int head = 0, tail = 0, next = 0;
        while (tail < m) {
            if (head == tail) {                                   /* a new component */
                int s = pass == 0 || tail > 0 ? -1 : start;
                if (s < 0 || seen[s]) { while (seen[next]) next++; s = next; }
                seen[s] = 1; ord[tail++] = s;
            }
            const int p = ord[head++];
            for (size_t a = vp[p]; a < vp[p + 1]; a++) {
                const int q = vc[a];
                if (cdone[q]) continue;
                cdone[q] = 1;
                for (size_t t = cp[q]; t < cp[q + 1]; t++) if (!seen[cm[t]]) { seen[cm[t]] = 1; ord[tail++] = cm[t]; }
            }
        }
        start = ord[m - 1];
    }
    for (int r = 0; r < m; r++) pos[ord[r]] = m - 1 - r;
    free(cp); free(vp); free(cm); free(vc); free(ord); free(seen); free(cdone);
    const double env1 = hp_envelope(J, pos, NULL);
    if (getenv("BRISK_HPDBG")) printf("       [Schur order: envelope %.3g as read, %.3g breadth-first, of %.3g]\n", env0, env1, 0.5 * (double)m * m);
    if (env1 < 0.5 * env0) {
        /* the first columns are kept when the envelope is narrow: the Schur matrix is then
         * zeroed, factored and solved within it (nothing of order m^2 per iteration) */
        if (env1 < 0.25 * (double)m * m) (void)hp_envelope(J, pos, &J->mfst);
        return pos;
    }
    free(pos);
    return NULL;
}

static int hp_build(HPJob *J, HEnt *E, size_t ne, const uint64_t *vals) {
    const size_t Wm = (size_t)J->Lm + 2;
    const int nb = J->nblk;
    J->mpos = NULL; J->mfst = NULL;
    qsort(E, ne, sizeof(HEnt), cmp_hent);
    size_t nA = 0, nC = 0;
    for (size_t e = 0; e < ne; e++) { if (E[e].con < 0) nC++; else nA++; }
    J->nnzA = nA; J->nnzC = nC;
    J->Av = (uint64_t *)calloc((nA ? nA : 1) * Wm, sizeof(uint64_t)); J->Cv = (uint64_t *)calloc((nC ? nC : 1) * Wm, sizeof(uint64_t));
    J->ei = (int *)malloc(sizeof(int) * (nA + 1)); J->ej = (int *)malloc(sizeof(int) * (nA + 1));
    J->ci = (int *)malloc(sizeof(int) * (nC + 1)); J->cj = (int *)malloc(sizeof(int) * (nC + 1));
    J->cptr = (size_t *)calloc((size_t)nb + 1, sizeof(size_t));
    J->ncon = (int *)calloc((size_t)nb, sizeof(int)); J->con = (int **)calloc((size_t)nb, sizeof(int *)); J->ptr = (size_t **)calloc((size_t)nb, sizeof(size_t *));
    J->lpptr = (int **)calloc((size_t)nb, sizeof(int *)); J->lpe = (int **)calloc((size_t)nb, sizeof(int *)); J->lpc = (int **)calloc((size_t)nb, sizeof(int *));
    J->kord = (int **)calloc((size_t)nb, sizeof(int *)); J->kcolptr = (int **)calloc((size_t)nb, sizeof(int *)); J->kcol = (int **)calloc((size_t)nb, sizeof(int *));
    size_t a = 0, c = 0, e = 0;
    for (int k = 0; k < nb; k++) {
        const int n = abs(J->bs[k]), lp = J->bs[k] < 0;
        J->cptr[k] = c;
        size_t e1 = e;
        while (e1 < ne && E[e1].blk == k) e1++;
        int nc = 0;
        for (size_t q = e; q < e1; q++) if (E[q].con >= 0 && (q == e || E[q].con != E[q - 1].con)) nc++;
        J->ncon[k] = nc;
        J->con[k] = (int *)malloc(sizeof(int) * (size_t)(nc + 1)); J->ptr[k] = (size_t *)malloc(sizeof(size_t) * (size_t)(nc + 1));
        int t = -1;
        for (size_t q = e; q < e1; q++) {
            if (E[q].i < 0 || E[q].j >= n || E[q].i > E[q].j || (lp && E[q].i != E[q].j)) { hp_fail("an entry is outside its block"); return 1; }
            if (q > e && cmp_hent(&E[q], &E[q - 1]) == 0) { hp_fail("a matrix entry is given twice"); return 1; }
            if (E[q].con < 0) { memcpy(J->Cv + c * Wm, vals + E[q].idx * Wm, Wm * sizeof(uint64_t)); J->ci[c] = E[q].i; J->cj[c] = E[q].j; c++; continue; }
            if (t < 0 || E[q].con != J->con[k][t]) { t++; J->con[k][t] = E[q].con; J->ptr[k][t] = a; }
            memcpy(J->Av + a * Wm, vals + E[q].idx * Wm, Wm * sizeof(uint64_t)); J->ei[a] = E[q].i; J->ej[a] = E[q].j; a++;
        }
        J->ptr[k][nc] = a;
        e = e1;
        if (lp) {
            int *cnt = (int *)calloc((size_t)n + 1, sizeof(int));
            for (size_t q = J->ptr[k][0]; nc > 0 && q < J->ptr[k][nc]; q++) cnt[J->ei[q] + 1]++;
            for (int i = 0; i < n; i++) cnt[i + 1] += cnt[i];
            const int tot = cnt[n];
            J->lpptr[k] = cnt; J->lpe[k] = (int *)malloc(sizeof(int) * (size_t)(tot + 1)); J->lpc[k] = (int *)malloc(sizeof(int) * (size_t)(tot + 1));
            int *pos = (int *)malloc(sizeof(int) * (size_t)(n + 1));
            memcpy(pos, cnt, sizeof(int) * (size_t)(n + 1));
            for (int tt = 0; tt < nc; tt++)
                for (size_t q = J->ptr[k][tt]; q < J->ptr[k][tt + 1]; q++) { const int i = J->ei[q]; J->lpe[k][pos[i]] = (int)q; J->lpc[k][pos[i]] = J->con[k][tt]; pos[i]++; }
            free(pos);
        } else {
            int *mark = (int *)malloc(sizeof(int) * (size_t)n), *cp = (int *)calloc((size_t)nc + 1, sizeof(int));
            for (int i = 0; i < n; i++) mark[i] = -1;
            size_t tot = 0;
            for (int tt = 0; tt < nc; tt++) tot += 2 * (J->ptr[k][tt + 1] - J->ptr[k][tt]);
            int *cl = (int *)malloc(sizeof(int) * (tot + 1));
            int w = 0;
            for (int tt = 0; tt < nc; tt++) {
                cp[tt] = w;
                for (size_t q = J->ptr[k][tt]; q < J->ptr[k][tt + 1]; q++) {
                    if (mark[J->ei[q]] != tt) { mark[J->ei[q]] = tt; cl[w++] = J->ei[q]; }
                    if (mark[J->ej[q]] != tt) { mark[J->ej[q]] = tt; cl[w++] = J->ej[q]; }
                }
            }
            cp[nc] = w;
            int *ord = (int *)malloc(sizeof(int) * (size_t)(nc + 1));
            for (int tt = 0; tt < nc; tt++) ord[tt] = tt;
            /* by decreasing column count (insertion by counting: counts are at most n) */
            int *bucket = (int *)calloc((size_t)n + 2, sizeof(int));
            for (int tt = 0; tt < nc; tt++) bucket[n - (cp[tt + 1] - cp[tt]) + 1]++;
            for (int i = 0; i <= n; i++) bucket[i + 1] += bucket[i];
            for (int tt = 0; tt < nc; tt++) ord[bucket[n - (cp[tt + 1] - cp[tt])]++] = tt;
            free(bucket); free(mark);
            J->kord[k] = ord; J->kcolptr[k] = cp; J->kcol[k] = cl;
        }
    }
    J->cptr[nb] = c;
    J->mpos = hp_schur_order(J);
    return 0;
}

/* reads an SDPA sparse file into J (master precision already set in mpx_L = J->Lm) */
static int hp_read(const char *fname, HPJob *J) {
    FILE *f = fopen(fname, "rb");
    if (!f) { fprintf(stderr, "brisk: cannot open %s\n", fname); return 1; }
    fseek(f, 0, SEEK_END); const long sz = ftell(f); fseek(f, 0, SEEK_SET);
    char *buf = (char *)malloc((size_t)sz + 2);
    if (!buf || fread(buf, 1, (size_t)sz, f) != (size_t)sz) { fclose(f); free(buf); hp_fail("cannot read the file"); return 1; }
    fclose(f); buf[sz] = '\n'; buf[sz + 1] = 0;
    const size_t Wm = (size_t)J->Lm + 2;
    char *p = buf;
    /* comment lines */
    for (;;) {
        char *q = p; while (*q == ' ' || *q == '\t') q++;
        if (*q == '"' || *q == '*') { while (*p && *p != '\n') p++; if (*p) p++; } else break;
    }
    char *end;
    J->m = (int)strtol(p, &end, 10); if (end == p) { hp_fail("no number of constraints"); free(buf); return 1; }
    p = end; while (*p && *p != '\n') p++;
    J->nblk = (int)strtol(p, &end, 10); if (end == p || J->nblk < 1) { hp_fail("no number of blocks"); free(buf); return 1; }
    p = end; while (*p && *p != '\n') p++;
    if (J->m < 1) { hp_fail("no constraints"); free(buf); return 1; }
    for (char *q = p; *q; q++) if (*q == ',' || *q == '(' || *q == ')' || *q == '{' || *q == '}') *q = ' ';
    J->bs = (int *)malloc(sizeof(int) * (size_t)J->nblk);
    for (int k = 0; k < J->nblk; k++) {
        J->bs[k] = (int)strtol(p, &end, 10);
        if (end == p || J->bs[k] == 0) { hp_fail("bad block sizes"); free(buf); return 1; }
        p = end;
    }
    while (*p && *p != '\n') p++;                 /* text after the block sizes ("= bLOCKsTRUCT"), as on the two lines above */
    J->b = (uint64_t *)calloc((size_t)J->m * Wm, sizeof(uint64_t));
    for (int i = 0; i < J->m; i++) {
        const char *e2;
        if (mpx_from_str(J->b + (size_t)i * Wm, p, &e2)) { hp_fail("bad objective vector"); free(buf); return 1; }
        p = (char *)e2;
    }
    size_t cap = 1024, ne = 0;
    HEnt *E = (HEnt *)malloc(sizeof(HEnt) * cap);
    uint64_t *vals = (uint64_t *)malloc(sizeof(uint64_t) * cap * Wm);
    for (;;) {
        while (*p && isspace((unsigned char)*p)) p++;
        if (!*p) break;
        long q[4];
        int okk = 1;
        for (int a = 0; a < 4; a++) { q[a] = strtol(p, &end, 10); if (end == p) { okk = 0; break; } p = end; }
        if (!okk) break;
        if (ne == cap) { cap *= 2; E = (HEnt *)realloc(E, sizeof(HEnt) * cap); vals = (uint64_t *)realloc(vals, sizeof(uint64_t) * cap * Wm); if (!E || !vals) { hp_fail("out of memory"); return 1; } }
        const char *e2;
        if (mpx_from_str(vals + ne * Wm, p, &e2)) { hp_fail("bad matrix entry"); free(buf); return 1; }
        p = (char *)e2;
        if (q[0] < 0 || q[0] > J->m || q[1] < 1 || q[1] > J->nblk) { hp_fail("matrix or block number out of range"); free(buf); return 1; }
        if (mpx_sgn(vals + ne * Wm) == 0) continue;
        HEnt *h = &E[ne];
        h->con = (int)q[0] - 1; h->blk = (int)q[1] - 1;
        h->i = (int)(q[2] < q[3] ? q[2] : q[3]) - 1; h->j = (int)(q[2] < q[3] ? q[3] : q[2]) - 1;
        h->idx = ne;
        if (h->con < 0) mpx_neg(vals + ne * Wm, vals + ne * Wm);       /* C = -F0 */
        ne++;
    }
    free(buf);
    const int rc = hp_build(J, E, ne, vals);
    free(E); free(vals);
    return rc;
}

/* the same from the numbers given in memory (doubles: converted exactly) */
static int hp_from_data(const BriskData *d, HPJob *J) {
    const size_t Wm = (size_t)J->Lm + 2;
    if (d->m < 1 || d->nblk < 1) { hp_fail("no constraints or no blocks"); return 1; }
    J->m = d->m; J->nblk = d->nblk;
    J->bs = (int *)malloc(sizeof(int) * (size_t)d->nblk);
    for (int k = 0; k < d->nblk; k++) { J->bs[k] = d->bs[k]; if (!J->bs[k]) { hp_fail("bad block sizes"); return 1; } }
    J->b = (uint64_t *)calloc((size_t)J->m * Wm, sizeof(uint64_t));
    for (int i = 0; i < J->m; i++) mpx_setd(J->b + (size_t)i * Wm, d->c[i]);
    HEnt *E = (HEnt *)malloc(sizeof(HEnt) * (d->nnz + 1));
    uint64_t *vals = (uint64_t *)malloc(sizeof(uint64_t) * (d->nnz + 1) * Wm);
    size_t ne = 0;
    for (size_t q = 0; q < d->nnz; q++) {
        if (d->v[q] == 0.0) continue;
        if (d->mat[q] < 0 || d->mat[q] > J->m || d->blk[q] < 1 || d->blk[q] > J->nblk || !isfinite(d->v[q])) { hp_fail("matrix or block number out of range"); free(E); free(vals); return 1; }
        HEnt *h = &E[ne];
        h->con = d->mat[q] - 1; h->blk = d->blk[q] - 1;
        h->i = (d->i[q] < d->j[q] ? d->i[q] : d->j[q]) - 1; h->j = (d->i[q] < d->j[q] ? d->j[q] : d->i[q]) - 1;
        h->idx = ne;
        mpx_setd(vals + ne * Wm, h->con < 0 ? -d->v[q] : d->v[q]);
        ne++;
    }
    const int rc = hp_build(J, E, ne, vals);
    free(E); free(vals);
    return rc;
}

static void hp_free(HPJob *J) {
    for (int k = 0; k < J->nblk; k++) {
        if (J->con) free(J->con[k]);
        if (J->ptr) free(J->ptr[k]);
        if (J->lpptr) { free(J->lpptr[k]); free(J->lpe[k]); free(J->lpc[k]); }
        if (J->kord) { free(J->kord[k]); free(J->kcolptr[k]); free(J->kcol[k]); }
        if (J->X) { free(J->X[k]); free(J->Z[k]); }
    }
    free(J->con); free(J->ptr); free(J->lpptr); free(J->lpe); free(J->lpc); free(J->kord); free(J->kcolptr); free(J->kcol);
    free(J->mpos); J->mpos = NULL; free(J->mfst); J->mfst = NULL;
    free(J->ncon); free(J->bs); free(J->ei); free(J->ej); free(J->ci); free(J->cj); free(J->cptr);
    free(J->Av); free(J->Cv); free(J->b); free(J->X); free(J->Z); free(J->y); free(J->pobj); free(J->dobj);
}

/* writes the exchange iterate: y one value per line, X and Z as "block i j value" (upper triangle) */
static void hp_write(const HPJob *J, const char *yfile, const char *xfile, const char *zfile, int nd) {
    const size_t Wm = (size_t)J->Lm + 2;
    char *s = (char *)malloc((size_t)nd + 64);
    if (yfile) {
        FILE *f = fopen(yfile, "w");
        if (f) { for (int p = 0; p < J->m; p++) { mpx_to_str(s, J->y + (size_t)p * Wm, nd); fprintf(f, "%s\n", s); } fclose(f); }
        else fprintf(stderr, "brisk: cannot write %s\n", yfile);
    }
    for (int w = 0; w < 2; w++) {
        const char *fn = w ? zfile : xfile;
        if (!fn) continue;
        FILE *f = fopen(fn, "w");
        if (!f) { fprintf(stderr, "brisk: cannot write %s\n", fn); continue; }
        uint64_t *const *V = w ? J->Z : J->X;
        for (int k = 0; k < J->nblk; k++) {
            const int n = abs(J->bs[k]);
            if (J->bs[k] < 0) { for (int i = 0; i < n; i++) { mpx_to_str(s, V[k] + (size_t)i * Wm, nd); fprintf(f, "%d %d %d %s\n", k + 1, i + 1, i + 1, s); } continue; }
            for (int i = 0; i < n; i++)
                for (int j = i; j < n; j++) {
                    const uint64_t *v = V[k] + ((size_t)i * n + j) * Wm;
                    if (mpx_sgn(v) == 0) continue;
                    mpx_to_str(s, v, nd); fprintf(f, "%d %d %d %s\n", k + 1, i + 1, j + 1, s);
                }
        }
        fclose(f);
    }
    free(s);
}

/* kind: 1 dd, 2 qd, 3 mpx with `digits` decimal digits. Returns the exit code. */
#include "hppre.inc"
#include "hpcert.inc"

/* 1 when lambda_min(M) >= need is proved for the symmetric n x n double matrix M (column-major,
 * taken exactly as given) in mpx arithmetic of at least `bits` bits: the double checks of the
 * bound mode (boundcert.c) when their rounding allowance (of the order of n u tr M in double) is
 * what fails them. The proof is hpc_lammin_ge's: the Cholesky factorization of
 * W = fl(M - c I) runs to the end with c = need + 2 (g tr(W) + u max(|M_ii|, |c|)),
 * g = gamma_{3n+10}(u), u the model unit of the arithmetic (hpcert.inc), and
 * tr(W) <= sum |M_ii| + n |c|; then lambda_min(M) >= need. Returns -1 when the matrices do
 * not fit in the memory (nothing was proved). */
int brisk_lammin_ge_mp(int n, const double *M, double need, int bits) {
    if (n <= 0) return 1;
    if (!isfinite(need)) return 0;
    if (bits < 128) return -1;                   /* (19 October 2026: at 64 bits and below HPC_U is 1 and the allowance below
                                                  * went negative: a "proof" of anything) */
    const int L0 = mpx_L, W0 = mpx_W;
    mpx_set_prec(bits);
    const size_t Wm = (size_t)mpx_W;
    if (2.0 * (double)n * n * Wm * 8.0 > 0.3 * brisk_mem_limit()) { mpx_L = L0; mpx_W = W0; return -1; }
    HPJob Jd;
    memset(&Jd, 0, sizeof(Jd));
    Jd.Lm = mpx_L;
    const double u = HPC_U(&Jd);
    double tr = 0, dmax = 0;
    for (int i = 0; i < n; i++) { const double a = fabs(M[i + (size_t)i * n]); tr += a; if (a > dmax) dmax = a; }
    tr *= 1.01;
    if (!isfinite(tr)) { mpx_L = L0; mpx_W = W0; return 0; }
    if (!((3.0 * n + 10.0) * u < 0.1)) { mpx_L = L0; mpx_W = W0; return -1; }   /* the error analysis needs (3n+10)u < 1 */
    const double g = 1.01 * (3.0 * n + 10.0) * u / (1 - (3.0 * n + 10.0) * u);
    /* the shift c >= need + the allowance, computed with upward rounding (in double, need + a tiny allowance rounded to need
     * and the proof covered only lambda_min >= need - allowance) */
    const int rm0 = fegetround();
    fesetround(FE_UPWARD);
    double c = need;
    for (int it = 0; it < 4; it++) c = need + (2.02 * (g * (tr + n * fabs(c)) + u * fmax(dmax, fabs(c))) + 1e-290);
    fesetround(rm0);
    if (!(c > need)) { mpx_L = L0; mpx_W = W0; return -1; }
    uint64_t *W = (uint64_t *)calloc((size_t)n * n * Wm + 1, sizeof(uint64_t));
    if (!W) { mpx_L = L0; mpx_W = W0; return -1; }
    uint64_t cm[MPX_MAXL + 2], t[MPX_MAXL + 2];
    mpx_setd(cm, c);
    for (int i = 0; i < n; i++)
        for (int j = 0; j <= i; j++) {
            /* the lower triangle of W, row-major as cholr_mp reads it */
            if (i == j) { mpx_setd(t, M[i + (size_t)i * n]); mpx_sub(W + ((size_t)i * n + i) * Wm, t, cm); }
            else mpx_setd(W + ((size_t)i * n + j) * Wm, M[i + (size_t)j * n]);
        }
    const int ok = cholr_mp(n, W, 0.0, NULL, NULL, 0, NULL, NULL, 0.0) == 0;
    free(W);
    mpx_L = L0; mpx_W = W0;
    return ok;
}
#include "hpred.inc"
static int g_hp_nored = 0;   /* 5.9: the solve is repeated without the reductions of hpred.inc */
/* 20 October 2026 (the owner): the preprocessing for the Summary at the end of the log, in the words of the standard
 * solver's Summary (main.c); filled once the presolve has run */
static char g_hpsum[2048]; static size_t g_hpsum_n;
static void hp_sum(const char *fmt, ...) {
    if (g_hpsum_n + 8 >= sizeof g_hpsum) return;
    va_list ap; va_start(ap, fmt);
    const int w = vsnprintf(g_hpsum + g_hpsum_n, sizeof g_hpsum - g_hpsum_n, fmt, ap);
    va_end(ap);
    if (w > 0) g_hpsum_n = g_hpsum_n + (size_t)w < sizeof g_hpsum ? g_hpsum_n + (size_t)w : sizeof g_hpsum - 1;
}

/* the pattern structures of the low-rank method */
typedef struct { long long key; int src; } LrKey;
static int lr_cmp(const void *a, const void *b) { const long long x = ((const LrKey *)a)->key, y = ((const LrKey *)b)->key; return x < y ? -1 : x > y ? 1 : 0; }
static void lr_build(HPJob *J) {
    const int nb = J->nblk;
    J->lr_np = (int *)calloc((size_t)nb, sizeof(int));
    J->lr_pi = (int **)calloc((size_t)nb, sizeof(int *)); J->lr_pj = (int **)calloc((size_t)nb, sizeof(int *));
    J->lr_rp = (int **)calloc((size_t)nb, sizeof(int *)); J->lr_rc = (int **)calloc((size_t)nb, sizeof(int *)); J->lr_rs = (int **)calloc((size_t)nb, sizeof(int *));
    J->lr_diag = (int **)calloc((size_t)nb, sizeof(int *));
    J->lr_es = (int *)malloc(sizeof(int) * (J->nnzA + 1)); J->lr_cs = (int *)malloc(sizeof(int) * (J->nnzC + 1));
    J->lr_R = (uint64_t **)calloc((size_t)nb, sizeof(uint64_t *));
    J->lr_rk = (int *)calloc((size_t)nb, sizeof(int)); J->lr_cap = (int *)calloc((size_t)nb, sizeof(int));
    for (int k = 0; k < nb; k++) {
        const int n = abs(J->bs[k]), lp = J->bs[k] < 0;
        const size_t a0 = J->ptr[k][0], a1 = J->ptr[k][J->ncon[k]], c0 = J->cptr[k], c1 = J->cptr[k + 1];
        const size_t tot = (a1 - a0) + (c1 - c0) + (size_t)n;
        LrKey *K = (LrKey *)malloc(sizeof(LrKey) * (tot + 1));
        size_t q = 0;
        for (size_t e = a0; e < a1; e++) { const int i = J->ei[e], j = lp ? i : J->ej[e]; K[q].key = (long long)(i < j ? i : j) * n + (i < j ? j : i); K[q].src = (int)(e - a0); q++; }
        for (size_t e = c0; e < c1; e++) { const int i = J->ci[e], j = lp ? i : J->cj[e]; K[q].key = (long long)(i < j ? i : j) * n + (i < j ? j : i); K[q].src = (int)((a1 - a0) + (e - c0)); q++; }
        for (int i = 0; i < n; i++) { K[q].key = (long long)i * n + i; K[q].src = -1 - i; q++; }
        qsort(K, tot, sizeof(LrKey), lr_cmp);
        int np = 0;
        for (size_t x = 0; x < tot; x++) if (x == 0 || K[x].key != K[x - 1].key) np++;
        J->lr_np[k] = np;
        J->lr_pi[k] = (int *)malloc(sizeof(int) * (size_t)(np + 1)); J->lr_pj[k] = (int *)malloc(sizeof(int) * (size_t)(np + 1));
        J->lr_diag[k] = (int *)malloc(sizeof(int) * (size_t)(n + 1));
        J->lr_rp[k] = (int *)calloc((size_t)n + 2, sizeof(int));
        int s = -1;
        for (size_t x = 0; x < tot; x++) {
            if (x == 0 || K[x].key != K[x - 1].key) { s++; J->lr_pi[k][s] = (int)(K[x].key / n); J->lr_pj[k][s] = (int)(K[x].key % n); }
            const int src = K[x].src;
            if (src < 0) J->lr_diag[k][-1 - src] = s;
            else if ((size_t)src < a1 - a0) J->lr_es[a0 + (size_t)src] = s;
            else J->lr_cs[c0 + ((size_t)src - (a1 - a0))] = s;
        }
        free(K);
        int *cnt = J->lr_rp[k];
        for (s = 0; s < np; s++) { cnt[J->lr_pi[k][s] + 1]++; if (J->lr_pi[k][s] != J->lr_pj[k][s]) cnt[J->lr_pj[k][s] + 1]++; }
        for (int i = 0; i < n; i++) cnt[i + 1] += cnt[i];
        J->lr_rc[k] = (int *)malloc(sizeof(int) * (size_t)(cnt[n] + 1)); J->lr_rs[k] = (int *)malloc(sizeof(int) * (size_t)(cnt[n] + 1));
        int *pos = (int *)malloc(sizeof(int) * (size_t)(n + 1));
        memcpy(pos, cnt, sizeof(int) * (size_t)n);
        for (s = 0; s < np; s++) {
            const int i = J->lr_pi[k][s], j = J->lr_pj[k][s];
            J->lr_rc[k][pos[i]] = j; J->lr_rs[k][pos[i]++] = s;
            if (i != j) { J->lr_rc[k][pos[j]] = i; J->lr_rs[k][pos[j]++] = s; }
        }
        free(pos);
    }
}
static void lr_free(HPJob *J) {
    if (!J->lr_np) return;
    for (int k = 0; k < J->nblk; k++) { free(J->lr_pi[k]); free(J->lr_pj[k]); free(J->lr_rp[k]); free(J->lr_rc[k]); free(J->lr_rs[k]); free(J->lr_diag[k]); free(J->lr_R[k]); }
    free(J->lr_np); free(J->lr_pi); free(J->lr_pj); free(J->lr_rp); free(J->lr_rc); free(J->lr_rs); free(J->lr_diag); free(J->lr_es); free(J->lr_cs); free(J->lr_R); free(J->lr_rk); free(J->lr_cap);
}

/* An interior, centred starting point from an approximate solution in double (X in J->X,
 * Z = C - A*(y) in J->Z, both possibly on the boundary or slightly outside): with
 * W = X - s Z = V diag(lambda) V' (s balances the two norms; at a solution X and Z are the
 * positive and negative parts of W), X0 = V diag(xi) V' and Z0 = mu X0^-1, where
 * xi = (lambda + sqrt(lambda^2 + 4 s mu)) / 2. Then X0 Z0 = mu I exactly and X0 - s Z0 = W:
 * the point is on the central path of a nearby problem, at the distance of the double
 * solution's error. mu is set from relerr, the error of the double solution. */
static double hp_interior1(HPJob *J, double relerr, double sfac, double fixmu, double kacc, double *const *Xin, double *const *Zin, double *perr) {
    const size_t Wm = (size_t)J->Lm + 2;
    const int nb = J->nblk;
    double nx = 0, nz = 0, N = 0;
    /* the spectral decompositions of W = X - s Z, once */
    double **Vs = (double **)calloc((size_t)nb, sizeof(double *)), **ws = (double **)calloc((size_t)nb, sizeof(double *));
    uint64_t **Zt = (uint64_t **)calloc((size_t)nb, sizeof(uint64_t *));
    int nmax = 1;
    for (int k = 0; k < nb; k++) {
        const int n = abs(J->bs[k]);
        const size_t len = J->bs[k] < 0 ? (size_t)n : (size_t)n * n;
        N += n; if (J->bs[k] > nmax) nmax = J->bs[k];
        Zt[k] = (uint64_t *)malloc(sizeof(uint64_t) * len * Wm);
        for (size_t i = 0; i < len; i++) { const double x = Xin[k][i], z = Zin[k][i]; nx += x * x; nz += z * z; }
    }
    const double s = sfac * (nx > 0 && nz > 0 ? sqrt(nx / nz) : 1.0);
    const double obj = 1 + fabs(mpx_getd(J->pobj)) + fabs(mpx_getd(J->dobj));
    int lw = 3 * nmax + 64;
    double *work = (double *)malloc(sizeof(double) * (size_t)lw);
    for (int k = 0; k < nb; k++) {
        const int n = abs(J->bs[k]);
        ws[k] = (double *)malloc(sizeof(double) * (size_t)n);
        if (J->bs[k] < 0) { for (int i = 0; i < n; i++) ws[k][i] = Xin[k][i] - s * Zin[k][i]; continue; }
        Vs[k] = (double *)malloc(sizeof(double) * (size_t)n * n);
        for (int i = 0; i < n; i++)
            for (int j = 0; j < n; j++)
                Vs[k][(size_t)i * n + j] = 0.5 * ((Xin[k][(size_t)i * n + j] + Xin[k][(size_t)j * n + i])
                                                - s * (Zin[k][(size_t)i * n + j] + Zin[k][(size_t)j * n + i]));
        int info = 0;
        BL(dsyev_)("V", "U", &n, Vs[k], &n, ws[k], work, &lw, &info);
    }
    free(work);
    double *Xd = (double *)malloc(sizeof(double) * (size_t)nmax * nmax), *Zd = (double *)malloc(sizeof(double) * (size_t)nmax * nmax);
    double murel = fixmu > 0 ? fixmu : fmin(fmax(10.0 * relerr, 1e-9), 1e-1);
    for (int trial = 0; ; trial++, murel *= 10.0) {
        const double mu = murel * obj / N;
        double dz2 = 0;
        for (int k = 0; k < nb; k++) {
            const int n = abs(J->bs[k]);
            if (J->bs[k] < 0) {
                for (int i = 0; i < n; i++) {
                    const double l = ws[k][i];
                    const double xi = l > 0 ? 0.5 * (l + sqrt(l * l + 4 * s * mu)) : 2 * s * mu / (sqrt(l * l + 4 * s * mu) - l);
                    mpx_setd(J->X[k] + (size_t)i * Wm, xi); mpx_setd(J->Z[k] + (size_t)i * Wm, mu / xi);
                    dz2 += (mu / xi - Zin[k][i]) * (mu / xi - Zin[k][i]);
                }
                continue;
            }
            memset(Xd, 0, sizeof(double) * (size_t)n * n); memset(Zd, 0, sizeof(double) * (size_t)n * n);
            for (int q = 0; q < n; q++) {
                const double l = ws[k][q];
                const double xi = l > 0 ? 0.5 * (l + sqrt(l * l + 4 * s * mu)) : 2 * s * mu / (sqrt(l * l + 4 * s * mu) - l);
                const double zi = mu / xi;
                const double *v = Vs[k] + (size_t)q * n;
                for (int i = 0; i < n; i++) { const double a = xi * v[i], c = zi * v[i]; for (int j = 0; j <= i; j++) { Xd[(size_t)i * n + j] += a * v[j]; Zd[(size_t)i * n + j] += c * v[j]; } }
            }
            for (int i = 0; i < n; i++)
                for (int j = 0; j <= i; j++) {
                    mpx_setd(J->X[k] + ((size_t)i * n + j) * Wm, Xd[(size_t)i * n + j]); mpx_setd(J->X[k] + ((size_t)j * n + i) * Wm, Xd[(size_t)i * n + j]);
                    mpx_setd(J->Z[k] + ((size_t)i * n + j) * Wm, Zd[(size_t)i * n + j]); mpx_setd(J->Z[k] + ((size_t)j * n + i) * Wm, Zd[(size_t)i * n + j]);
                    const double d = Zd[(size_t)i * n + j] - 0.5 * (Zin[k][(size_t)i * n + j] + Zin[k][(size_t)j * n + i]);
                    dz2 += (i == j ? 1.0 : 2.0) * d * d;
                }
        }
        /* the infeasibilities of the point: accepted when they do not exceed mu (relative) by
         * much; otherwise mu is raised (near-zero eigenvalues of W, i.e. no strict
         * complementarity, are lifted to sqrt(s mu), which costs feasibility) */
        /* (and both matrices must be safely positive definite as doubles: xi within 1e12) */
        double ximin = 1e300, ximax = 0;
        for (int k = 0; k < nb; k++) for (int q = 0; q < abs(J->bs[k]); q++) {
            const double l = ws[k][q];
            const double xi = l > 0 ? 0.5 * (l + sqrt(l * l + 4 * s * mu)) : 2 * s * mu / (sqrt(l * l + 4 * s * mu) - l);
            if (xi < ximin) ximin = xi;
            if (xi > ximax) ximax = xi;
        }
        hp_measure(J, J->X, J->y, Zt, -1.0);
        const double pinf = J->best_pinf, dinf = sqrt(dz2) / (1 + sqrt(nz));
        *perr = fmax(pinf, dinf);
        if (fixmu > 0) break;
        if (!(ximin > (getenv("BRISK_HPXI") ? atof(getenv("BRISK_HPXI")) : 1e-12) * ximax) && murel < 1e-1 && trial < 9) continue;
        if (fmax(pinf, dinf) <= (getenv("BRISK_HPSTARTK") ? atof(getenv("BRISK_HPSTARTK")) : kacc) * murel || murel >= 1e-1 || trial >= 9) break;
    }
    for (int k = 0; k < nb; k++) { free(Vs[k]); free(ws[k]); free(Zt[k]); }
    free(Vs); free(ws); free(Zt); free(Xd); free(Zd);
    return murel;
}
/* The balance s. With the norms' ratio the eigenvalues of W near zero (no strict
 * complementarity) are lifted to sqrt(s mu) in X; where A(X) is sensitive to that (Mittelmann's
 * vibra4: primal infeasibility 1.2 at the first mu that passed the test, from a double
 * solution with error 5e-8) a smaller s moves the lift to Z. When the first balance ends
 * 1000 times above its first mu or with an infeasibility above 1e-2, the balances 1e-3,
 * 1e3 and 1e-6 times it are tried with a stricter test (infeasibility at most 100 mu: a
 * point closer to the solution whose infeasibility is 1000 mu stalls, roa_univar_d16_K), and
 * one that passes it with a mu at least 10 times smaller, or with the same mu and 10 times
 * less infeasibility, is taken (5.3). */
static double hp_interior(HPJob *J, double relerr) {
    const size_t Wm = (size_t)J->Lm + 2;
    const int nb = J->nblk;
    double **Xin = (double **)calloc((size_t)nb, sizeof(double *)), **Zin = (double **)calloc((size_t)nb, sizeof(double *));
    for (int k = 0; k < nb; k++) {
        const int n = abs(J->bs[k]);
        const size_t len = J->bs[k] < 0 ? (size_t)n : (size_t)n * n;
        Xin[k] = (double *)malloc(sizeof(double) * len); Zin[k] = (double *)malloc(sizeof(double) * len);
        for (size_t i = 0; i < len; i++) { Xin[k][i] = mpx_getd(J->X[k] + i * Wm); Zin[k][i] = mpx_getd(J->Z[k] + i * Wm); }
    }
    static const double sf[4] = { 1.0, 1e-3, 1e3, 1e-6 };
    double e = 0, murel = hp_interior1(J, relerr, 1.0, 0.0, 3000.0, Xin, Zin, &e);
    const double mu0 = fmin(fmax(10.0 * relerr, 1e-9), 1e-1);
    if ((murel > 999.0 * mu0 || e > 1e-2) && !getenv("BRISK_HPNOBAL")) {
        int best = 0, last = 0; double bmu = murel, be = e;
        for (int c = 1; c < 4; c++) {
            double ec = 0; const double mc = hp_interior1(J, relerr, sf[c], 0.0, 100.0, Xin, Zin, &ec);
            last = c;
            if (ec <= 100.0 * mc && (mc <= 0.1 * murel || (mc <= murel && ec <= 0.1 * e)) && (mc < bmu || (mc == bmu && ec < be))) { best = c; bmu = mc; be = ec; }
        }
        if (best != last) { double e2; (void)hp_interior1(J, relerr, sf[best], bmu, best ? 100.0 : 3000.0, Xin, Zin, &e2); }
        if (J->verbose > 1 && best) printf("     start: balance %g times the ratio of the norms (mu %.0e -> %.0e, infeasibility %.1e -> %.1e)\n", sf[best], murel, bmu, e, be);
        murel = bmu;
    }
    for (int k = 0; k < nb; k++) { free(Xin[k]); free(Zin[k]); }
    free(Xin); free(Zin);
    return murel;
}

/* ------------------------------------------------------------------ all the digits, for the interfaces
 * The last high-precision solution is kept (until the next one) and handed out as decimal text:
 *   brisk_hp_digits()                  decimal digits of the last solve (0: none)
 *   brisk_hp_nblk(), brisk_hp_blocksize(k)   the blocks (size negative: an LP block)
 *   brisk_hp_count(what, block)        number of values of an item
 *   brisk_hp_text(what, block, buf, len)  the values, separated by blanks; returns the length
 *                                      written, or -(needed length) when buf is too short
 * what: 0 primal objective <C,X>, 1 dual objective b'y, 2 y, 3 X of a block, 4 Z of a block,
 * 5 the rigorous bound of a certificate (-bound p|d: rounded up for p, down for d; with
 * -bound pd that of p), 6 the rigorous bound of side p, 7 that of side d, 8 the certificate X
 * of side p (a block), 9 the certificate y of side d (the certificates with all the digits of
 * their check, brisk_hp_cert_digits(); the solution with those of -prec)
 * (blocks from 0; a matrix block row by row, n*n values; an LP block its n values). */
static struct { int digits, cdigits, Lm, m, nb, *bs; uint64_t **X, **Z, *y, *pobj, *dobj; int bside; uint64_t *bound, *bnd[2], **certX, *certy; } g_hp_last;
static void hp_last_free(void) {
    for (int k = 0; k < g_hp_last.nb; k++) { if (g_hp_last.X) free(g_hp_last.X[k]); if (g_hp_last.Z) free(g_hp_last.Z[k]); if (g_hp_last.certX) free(g_hp_last.certX[k]); }
    free(g_hp_last.X); free(g_hp_last.Z); free(g_hp_last.y); free(g_hp_last.pobj); free(g_hp_last.dobj); free(g_hp_last.bs); free(g_hp_last.bound);
    free(g_hp_last.bnd[0]); free(g_hp_last.bnd[1]); free(g_hp_last.certX); free(g_hp_last.certy);
    memset(&g_hp_last, 0, sizeof(g_hp_last));
}
#if defined(__GNUC__)
#define HP_API __attribute__((visibility("default")))
#else
#define HP_API
#endif
void brisk_hp_clear(void) { hp_last_free(); }
HP_API int brisk_hp_digits(void) { return g_hp_last.digits; }
HP_API int brisk_hp_cert_digits(void) { return g_hp_last.cdigits ? g_hp_last.cdigits : g_hp_last.digits; }
HP_API int brisk_hp_nblk(void) { return g_hp_last.nb; }
HP_API int brisk_hp_blocksize(int block) { return block >= 0 && block < g_hp_last.nb ? g_hp_last.bs[block] : 0; }   /* negative: LP */
HP_API long long brisk_hp_count(int what, int block) {
    if (!g_hp_last.digits) return 0;
    if (what == 0 || what == 1) return g_hp_last.pobj ? 1 : 0;
    if (what == 5) return g_hp_last.bound ? 1 : 0;                 /* the rigorous bound of -bound / -certify-x / -certify-y */
    if (what == 6 || what == 7) return g_hp_last.bnd[what - 6] ? 1 : 0;
    if (what == 8 && g_hp_last.certX && block >= 0 && block < g_hp_last.nb) { const long long n = g_hp_last.bs[block]; return n < 0 ? -n : n * n; }
    if (what == 9) return g_hp_last.certy ? g_hp_last.m : 0;
    if (what == 2) return g_hp_last.y ? g_hp_last.m : 0;
    if ((what == 3 || what == 4) && block >= 0 && block < g_hp_last.nb) { const long long n = g_hp_last.bs[block]; return n < 0 ? -n : n * n; }
    return 0;
}
HP_API long long brisk_hp_text(int what, int block, char *buf, long long len) {
    const long long cnt = brisk_hp_count(what, block);
    if (cnt <= 0) return 0;
    const int nd = what >= 8 ? brisk_hp_cert_digits() : g_hp_last.digits;
    const long long need = cnt * (long long)(nd + 40) + 1;
    if (!buf || len < need) return -need;
    const uint64_t *src = what == 0 ? g_hp_last.pobj : what == 1 ? g_hp_last.dobj : what == 2 ? g_hp_last.y : what == 3 ? g_hp_last.X[block] : what == 5 ? g_hp_last.bound
                        : what == 6 || what == 7 ? g_hp_last.bnd[what - 6] : what == 8 ? g_hp_last.certX[block] : what == 9 ? g_hp_last.certy : g_hp_last.Z[block];
    const int L0 = mpx_L, W0 = mpx_W;
    mpx_L = g_hp_last.Lm; mpx_W = mpx_L + 2;
    const size_t Wm = (size_t)mpx_W;
    long long pos = 0;
    for (long long i = 0; i < cnt; i++) {
        if (what == 5) hpc_bound_str(buf + pos, src, nd, g_hp_last.bside == 1 ? +1 : -1);
        else if (what == 6 || what == 7) hpc_bound_str(buf + pos, src, nd, what == 6 ? +1 : -1);
        else mpx_to_str(buf + pos, src + (size_t)i * Wm, nd);
        pos += (long long)strlen(buf + pos);
        buf[pos++] = i + 1 < cnt ? ' ' : 0;
    }
    mpx_L = L0; mpx_W = W0;
    return pos - 1;
}

/* 5.16: the tolerances and the measured errors are doubles, so below 1e-290 they underflow (ISSUES 90: at
 * -prec 500 the tolerance read 0 and the last level iterated 42 times until the errors did too); the floor
 * caps the tolerance at 1e-290, which the log says */
static double hp_tolf(double v) { return v < 1e-290 ? 1e-290 : v; }
/* 23 October 2026 (ISSUES 256): an infeasibility verdict of the iterates checked by a certificate on the problem as read
 * (O, the double data the measures use). The certificate is the solution of the ray problem of the side in question,
 * solved in double precision and, when that does not certify, in the working precision:
 *   DUAL INFEASIBLE, an X >= 0 with A(X) = 0 and <C,X> < 0:   min <C,D> over D >= 0 with A(D) = 0 and tr D = 1;
 *   PRIMAL INFEASIBLE, a y with b'y > 0 and A*(y) <= 0:       max b'y over y with A*(y) <= 0 and -1 <= y_i <= 1
 *                                                             (the bounds as an LP block; b'y = 1 alone leaves the
 *                                                             problem unbounded when a strict certificate exists).
 * Rounded to doubles, the solution is checked by the double solve's own test (ps_infeas_check: the violation of the ray
 * normalized to <C,X> = -1 or b'y = 1 at most 1e-6). On clique_60_k15_4_4 (SDP, the owner's benchmark) the first problem
 * has the value 9.1e-8, a weak direction; its double solution still certifies (violation 5.3e-8).
 * Returns 1 (certified), 0 (not), -1 (not run); *viol the violation. */
static int g_hp_inray = 0;
int brisk_hp_run(const char *fname, const BriskData *data, BriskResult *res, const BriskResult *warm, const Params *par, int kind, int digits, double tol, const char *yfile, const char *xfile, const char *zfile);
static int hp_ray_check(const PSOrig *O, int dinf, const Params *par, double tleft, int kind, int digits, double *viol, int *prec) {
    *viol = INFINITY;
    const int m = O->m, nb = O->nblk;
    size_t nid = 0;
    for (int k = 0; k < nb; k++) nid += (size_t)abs(O->bs[k]);
    const size_t cap = O->nnz + nid + 4 * (size_t)m + 4;
    int *mat = (int *)malloc(sizeof(int) * cap), *blk = (int *)malloc(sizeof(int) * cap), *ii = (int *)malloc(sizeof(int) * cap), *jj = (int *)malloc(sizeof(int) * cap);
    double *v = (double *)malloc(sizeof(double) * cap), *c = (double *)calloc((size_t)m + 2, sizeof(double));
    int *bs = (int *)malloc(sizeof(int) * (size_t)(nb + 2));
    memcpy(bs, O->bs, sizeof(int) * (size_t)nb);
    size_t q = 0;
    /* (the SDPA numbers, 1-based: matrix 0 is F0 = -C, matrix i the constraint i, the objective c = b of the ray problem) */
    #define HP_PUT(mt, k, i, j, x) do { mat[q] = (mt); blk[q] = (k) + 1; ii[q] = (i) + 1; jj[q] = (j) + 1; v[q] = (x); q++; } while (0)
    for (size_t e = 0; e < O->nnz; e++) {
        if (O->v[e] == 0) continue;
        if (O->con[e] < 0) { if (dinf) HP_PUT(0, O->blk[e], O->ii[e], O->jj[e], -O->v[e]); }
        else HP_PUT(O->con[e] + 1, O->blk[e], O->ii[e], O->jj[e], O->v[e]);
    }
    int nb2 = nb, m2 = m;
    if (dinf) {
        for (int k = 0; k < nb; k++) for (int i = 0; i < abs(O->bs[k]); i++) HP_PUT(m + 1, k, i, i, 1.0);
        m2 = m + 1; c[m] = 1.0;
    } else {
        /* the bounds -1 <= y_i <= 1 as an LP block of 2m entries, 1 - y_i and 1 + y_i (F0 = -1 there) */
        bs[nb] = -2 * m; nb2 = nb + 1;
        for (int i = 0; i < m; i++) {
            c[i] = O->b[i];
            HP_PUT(0, nb, i, i, -1.0); HP_PUT(0, nb, m + i, m + i, -1.0);
            HP_PUT(i + 1, nb, i, i, 1.0); HP_PUT(i + 1, nb, m + i, m + i, -1.0);
        }
    }
    #undef HP_PUT
    BriskData d = { m2, nb2, bs, c, q, mat, blk, ii, jj, v };
    Params p2 = *par;
    p2.verbose = -1; p2.bound_side = 0; p2.bound_xfile = NULL; p2.bound_yfile = NULL;
    if (par->timelimit > 0) p2.timelimit = fmax(1.0, tleft);
    p2.t_start = wtime();          /* (its limit is the time left, from now: brisk_hp_run counts from t_start) */
    char sum[sizeof g_hpsum]; const size_t sum_n = g_hpsum_n;
    memcpy(sum, g_hpsum, sizeof sum);
    /* first in double precision by the default pipeline, as the double solve that warms this solver (most rays are not
     * weak: in double-double alone the ray problem of infeas_d_3x150_m1500 took 48 s after a verdict of 1.4 s); in the
     * working precision only when that does not certify, from the double point */
    int ok = 0; double obj = 0;
    BriskResult r0;
    (void)brisk_double_solve_data(&d, &r0);
    const int have0 = r0.status >= 0 && r0.status != 1 && r0.status != 2 && r0.have_x && r0.X && r0.y;
    if (have0) ok = ps_infeas_check(O, dinf ? r0.X : NULL, dinf ? NULL : r0.y, dinf ? 2 : 1, &obj, viol, NULL) == 1;
    *prec = 0;
    if (!ok) {
        BriskResult r; memset(&r, 0, sizeof r);
        g_hp_inray = 1;
        const int rr = brisk_hp_run(NULL, &d, &r, have0 ? &r0 : NULL, &p2, kind, digits, 0.0, NULL, NULL, NULL);
        g_hp_inray = 0;
        *prec = 1;
        if (r.have_x && (rr == 0 || rr == 10)) ok = ps_infeas_check(O, dinf ? r.X : NULL, dinf ? NULL : r.y, dinf ? 2 : 1, &obj, viol, NULL) == 1;
        brisk_result_free(&r);
    }
    brisk_result_free(&r0);
    memcpy(g_hpsum, sum, sizeof sum); g_hpsum_n = sum_n;      /* (the Summary is the outer solve's) */
    free(mat); free(blk); free(ii); free(jj); free(v); free(c); free(bs);
    return ok;
}

/* How far the blocks M of J are from positive semidefinite, from their eigenvalues in double precision, over scale:
 * lo and hi bound max(0, -lambda_min) below and above by the eigenvalues' error, (n + 1) eps |M|_2 for a block of order
 * n (the rounding of M to doubles and the backward-stable eigensolver; an LP block: its most negative entry, eps |m|).
 * For comparing two points (24 October 2026, ISSUES 278: the double solution against the result of a solve that ended
 * short of its tolerance): on control problems |Z| is 1e7 times |C| or more, and the bounds are then 1e-7 apart. */
static void hp_psd_viol_d(const HPJob *J, uint64_t *const *M, double scale, double *lo, double *hi) {
    const size_t Wm = (size_t)mpx_W;
    double vlo = 0, vhi = 0;
    for (int k = 0; k < J->nblk; k++) {
        const int n = abs(J->bs[k]);
        if (J->bs[k] < 0) {
            for (int i = 0; i < n; i++) { const double z = mpx_getd(M[k] + (size_t)i * Wm); vlo = fmax(vlo, -z - DBL_EPSILON * fabs(z)); vhi = fmax(vhi, -z + DBL_EPSILON * fabs(z)); }
            continue;
        }
        double *A = (double *)malloc(sizeof(double) * (size_t)n * n), *ev = (double *)malloc(sizeof(double) * (size_t)n), wq; int iwq, lw = -1, liw = -1, info = 0;
        for (size_t i = 0; i < (size_t)n * n; i++) A[i] = mpx_getd(M[k] + i * Wm);
        BL(dsyevd_)("N", "L", &n, A, &n, ev, &wq, &lw, &iwq, &liw, &info);
        lw = (int)wq; liw = iwq;
        double *wk = (double *)malloc(sizeof(double) * (size_t)(lw > 1 ? lw : 1)); int *iw = (int *)malloc(sizeof(int) * (size_t)(liw > 1 ? liw : 1));
        BL(dsyevd_)("N", "L", &n, A, &n, ev, wk, &lw, iw, &liw, &info);
        if (info == 0 && n > 0) {
            const double d = (n + 1) * DBL_EPSILON * fmax(fabs(ev[0]), fabs(ev[n - 1]));
            vlo = fmax(vlo, -ev[0] - d); vhi = fmax(vhi, -ev[0] + d);
        } else if (info != 0) vhi = INFINITY;
        free(wk); free(iw); free(A); free(ev);
    }
    *lo = vlo / scale; *hi = vhi / scale;
}

int brisk_hp_run(const char *fname, const BriskData *data, BriskResult *res, const BriskResult *warm, const Params *par, int kind, int digits, double tol, const char *yfile, const char *xfile, const char *zfile) {
    HPJob J_, *J = &J_;
    memset(J, 0, sizeof(*J));
    const BriskResult *const warm_arg = warm;   /* (the double solution on the problem as read: a candidate for the result) */
    /* 24 October 2026 (ISSUES 278): the limit counts from the run's start (par->t_start, set before the double solve that
     * warms this one; main.c), so that the run as a whole keeps it: from this function's own start the method had the
     * whole limit again after the double solve had used it (-timelimit 3: 3.0 s of double solve, then 6.8 s here). The
     * totals printed and returned count from it too, the double solve included (until the evening of 24 October they
     * added its time once more) */
    J->tstart = par->t_start > 0 ? par->t_start : wtime();
    brisk_threads_busy(0);            /* only the cores that are free (solver.c) */
    const int target_bits = kind == 1 ? 106 : kind == 2 ? 212 : (int)ceil(digits * 3.3219280949) + 8;
    /* the master precision: the target and one guard limb; two more with the presolve, whose
     * eliminations then leave the reduced problem exact to the working precision */
    const int use_pre = !getenv("BRISK_HPNOPRE");
    /* (and the room for the extension levels: twice the target) */
    const int max_ext = getenv("BRISK_HPEXT") ? atoi(getenv("BRISK_HPEXT")) : par->hp_ext;
    { const int g = use_pre ? 192 : 64; mpx_set_prec(target_bits + (max_ext > 0 && target_bits + 64 > g ? target_bits + 64 : g)); }
    J->Lm = mpx_L;
    if (data ? hp_from_data(data, J) : hp_read(fname, J)) { hp_free(J); return 2; }
    const size_t Wm = (size_t)J->Lm + 2;
    /* the problem as read, for the Summary */
    int rd_m = J->m, rd_sdp = 0, rd_maxn = 0, rd_lp = 0;
    for (int k = 0; k < J->nblk; k++) { if (J->bs[k] > 0) { rd_sdp++; if (J->bs[k] > rd_maxn) rd_maxn = J->bs[k]; } else rd_lp += -J->bs[k]; }
    g_hpsum_n = 0; g_hpsum[0] = 0;
    /* blocks split by their aggregate sparsity pattern (hppre.inc): from here on J is the
     * split problem; the results are put back into the layout as read at the end */
    HPSplit sp_; int split = 0;
    BriskResult warm2_;
    memset(&sp_, 0, sizeof(sp_)); memset(&warm2_, 0, sizeof(warm2_));
    if (use_pre && !getenv("BRISK_HPNOSPLIT")) {
        HPJob Js_;
        if (hp_split(J, &Js_, &sp_)) {
            if (par->verbose >= 0) printf("presolve: %d SDP block(s) split by their sparsity pattern: %d block(s) as read, %d now (%d LP variable(s) from diagonal entries%s)\n",
                                          sp_.nsplit, sp_.nb0, Js_.nblk, sp_.nlp, sp_.ndrop ? "; indices without entries dropped" : "");
            const double ts = J->tstart;
            hp_free(J); *J = Js_; J->tstart = ts;
            split = 1;
            if (warm && warm->nblk == sp_.nb0 && warm->m == J->m && warm->X && warm->y) {
                warm2_ = *warm; warm2_.nblk = J->nblk;
                warm2_.X = hp_split_fwd_d(&sp_, J, warm->X);
                warm2_.Z = warm->Z ? hp_split_fwd_d(&sp_, J, warm->Z) : NULL;
                warm = &warm2_;
            } else warm = NULL;
        }
    }
    /* 5.9: the reductions of the double pipeline (hpred.inc): sign symmetry, orbits of the
     * constraints, the block structure of the data. From here on J is the reduced problem; the
     * solution is put back and measured on the problem before them (red_.Jfull) at the end.
     * (Interior-point method; not with -bound, whose certificate works on the layout as read.) */
    HPRed red_; int reduced = 0;
    BriskResult warm3_;
    const BriskResult *warm_in = warm;
    memset(&warm3_, 0, sizeof(warm3_));
    memset(&red_, 0, sizeof(red_));
    if (use_pre && !g_hp_nored && par->bound_side == 0 && par->fom <= 0 && par->lralm <= 0) {
        { extern double g_sym_est_cap; extern int g_sym_cap_bound; g_sym_est_cap = 0; g_sym_cap_bound = 0; }
        g_hpred_bits = target_bits;
        reduced = hp_reduce(J, &red_, par);
        if (reduced) {
            /* the double solution in the reduced layout (none after a chordal decomposition:
             * the reduced problem then starts in its own double level) */
            double **X3 = NULL, **Z3 = NULL, *y3 = NULL;
            const int wok = warm && warm->nblk == red_.Jfull.nblk && warm->m == red_.Jfull.m && warm->X && warm->y && !getenv("BRISK_HPREDCOLD")
                            && hp_red_fwd(&red_, warm->X, warm->Z, warm->y, &X3, &Z3, &y3);
            if (warm == &warm2_) { for (int k = 0; k < warm2_.nblk; k++) { free(warm2_.X[k]); if (warm2_.Z) free(warm2_.Z[k]); } free(warm2_.X); free(warm2_.Z); }
            if (wok) { warm3_ = *warm_in; warm3_.nblk = J->nblk; warm3_.m = J->m; warm3_.X = X3; warm3_.Z = Z3; warm3_.y = y3; warm = &warm3_; }
            else warm = NULL;
            memset(&warm2_, 0, sizeof(warm2_));
        }
    }
    HPJob Jr_, Jd_;
    HPJob *JB = J;                    /* the problem the presolve works on: as read, or its dual form */
    int dualized = 0;
    const int nb_orig = J->nblk, m_orig = J->m;
    if (use_pre) {
        /* the dual form when it leaves clearly fewer rows: positions - m <= 0.6 m */
        double npos = 0;
        for (int k = 0; k < nb_orig; k++) { const double n = abs(J->bs[k]); npos += J->bs[k] < 0 ? n : n * (n + 1) / 2; }
        const char *ed = getenv("BRISK_HPDUAL");
        const int want = ed ? atoi(ed) : (par->fom <= 0 && par->lralm <= 0 && npos - m_orig <= 0.6 * m_orig && npos < 2e5 && (!warm || warm->Z));   /* (interior-point method only) */
        if (want && hp_dualize(J, &Jd_) == 0) { JB = &Jd_; dualized = 1; }
    }
    HPPre *pre = use_pre ? hp_presolve(JB, &Jr_, par->verbose >= 0, !dualized || !getenv("BRISK_HPNODUALFACE")) : NULL;
    if (dualized && !pre) { hp_free(&Jd_); JB = J; dualized = 0; pre = hp_presolve(JB, &Jr_, par->verbose >= 0, 1); }
    if (dualized && par->verbose >= 0) printf("presolve: the dual form (%d rows over the positions of the cone, %d after the elimination of y; m was %d)\n", JB->m, Jr_.m, m_orig);
    const int nb_base = JB->nblk, m_base = JB->m;
    const int hp_nfree_read = pre && !dualized ? pre->nfree : 0;   /* (for the Summary; pre is freed before it) */
    /* the Summary's preprocessing lines (one per step, as the standard solver's) */
    if (split) hp_sum("%d semidefinite block%s split by their sparsity pattern (blocks %d -> %d%s)\n", sp_.nsplit, sp_.nsplit > 1 ? "s" : "", sp_.nb0, nb_orig,
                      sp_.nlp ? "; diagonal entries alone in their pattern as nonnegative variables" : "");
    if (reduced) for (int q = 0; q < red_.n; q++) {
        const HPStage *St = &red_.st[q];
        if (St->kind == 1) hp_sum("sign symmetry: equality constraints %d -> %d, blocks %d -> %d\n", St->m0, St->m1, St->nb0, St->nb1);
        else if (St->kind == 2) hp_sum("symmetry reduction: equality constraints %d -> %d, blocks %d -> %d\n", St->m0, St->m1, St->nb0, St->nb1);
        else if (St->kind == 3) hp_sum("block structure shared by the data: blocks %d -> %d\n", St->nb0, St->nb1);
        else if (St->kind == 4) hp_sum("chordal decomposition: blocks %d -> %d, equality constraints %d -> %d\n", St->nb0, St->nb1, St->m0, St->m1);
        else if (St->kind == 5) hp_sum("%d equality constraint%s left without entries removed\n", St->m0 - St->m1, St->m0 - St->m1 > 1 ? "s" : "");
    }
    if (g_hp_nored) hp_sum("the symmetry and structure reductions did not hold in the working precision: solved without them\n");
    if (dualized) hp_sum("the dual form, its free variables eliminated (equality constraints %d -> %d)\n", m_orig, Jr_.m);
    if (pre && !dualized) {
        if (pre->nfree) hp_sum("%d free variable%s eliminated by substitution\n", pre->nfree, pre->nfree > 1 ? "s" : "");
        if (pre->nface) hp_sum("facial reduction: %d constraint%s removed\n", pre->nface, pre->nface > 1 ? "s" : "");
        if (pre->ndrop) hp_sum("%d equality constraint%s made redundant by these eliminations removed\n", pre->ndrop, pre->ndrop > 1 ? "s" : "");
    } else if (pre && pre->nface) hp_sum("facial reduction of the dual form: %d reduction%s\n", pre->nface, pre->nface > 1 ? "s" : "");
    if (pre) { Jr_.tstart = J->tstart; J = &Jr_; }
    int nb = J->nblk, m = J->m;
    J->X = (uint64_t **)calloc((size_t)nb, sizeof(uint64_t *)); J->Z = (uint64_t **)calloc((size_t)nb, sizeof(uint64_t *));
    int maxn = 0, nsdp = 0;
    for (int k = 0; k < nb; k++) {
        const size_t len = J->bs[k] < 0 ? (size_t)(-J->bs[k]) : (size_t)J->bs[k] * J->bs[k];
        J->X[k] = (uint64_t *)calloc(len * Wm, sizeof(uint64_t)); J->Z[k] = (uint64_t *)calloc(len * Wm, sizeof(uint64_t));
        if (J->bs[k] > 0) { nsdp++; if (J->bs[k] > maxn) maxn = J->bs[k]; }
    }
    J->y = (uint64_t *)calloc((size_t)m * Wm, sizeof(uint64_t));
    J->pobj = (uint64_t *)calloc(Wm, sizeof(uint64_t)); J->dobj = (uint64_t *)calloc(Wm, sizeof(uint64_t));
    J->verbose = par->verbose >= 0 ? par->verbose + 1 : 0;
    J->maxit = par->maxit > 0 ? 10 * par->maxit : 1000;
    J->method = par->lralm > 0 ? 2 : par->fom > 0 ? 1 : 0;
    if (J->method == 2) lr_build(J);
    J->cgmax = getenv("BRISK_HPCG") ? atoi(getenv("BRISK_HPCG")) : 0; J->sigfac = getenv("BRISK_HPSIG") ? atof(getenv("BRISK_HPSIG")) : 0;
    { const char *e = getenv("BRISK_HPREG"); J->pivreg = e ? atof(e) : 0.0; }
    J->dep = getenv("BRISK_HPNODEP") ? NULL : (int *)calloc((size_t)m + 1, sizeof(int));
    J->ncen = getenv("BRISK_HPCEN") ? atoi(getenv("BRISK_HPCEN")) : 6;
    J->fast = getenv("BRISK_HPFAST") ? atof(getenv("BRISK_HPFAST")) : 1.0;
    J->timelimit = par->timelimit;
    /* the ladder: double, dd, [qd], [mpx at doubling precisions up to the target] */
    int lk[40], lb[40], nl = 0;
    lk[nl] = 0; lb[nl++] = 53;
    lk[nl] = 1; lb[nl++] = 106;
    if (kind >= 2) { lk[nl] = 2; lb[nl++] = 212; }
    if (kind == 3) { int b2 = 424; while (b2 < target_bits) { lk[nl] = 3; lb[nl++] = b2; b2 *= 2; } lk[nl] = 3; lb[nl++] = target_bits; }
    if (kind == 3 && target_bits <= 212) nl = target_bits <= 106 ? 2 : 3;
    const int final_bits = lb[nl - 1];
    J->tol = tol > 0 ? tol : hp_tolf(ldexp(1.0, -(int)((J->method ? 0.60 : 0.75) * final_bits)));   /* the first-order methods: 0.6 */
    const char *lname[4] = { "double", "double-double", "quad-double", "variable precision" };
    if (J->verbose) {
        printf("BRISK %s high precision: %s\n", BRISK_VERSION, J->method == 1 ? "augmented Lagrangian method with semismooth Newton-CG" : J->method == 2 ? "low-rank augmented Lagrangian method" : "interior-point method");
        brisk_log_threads(1);
        printf("problem %s: m = %d, %d SDP blocks (max n = %d), %d blocks in all, read %.2fs\n", fname, m, nsdp, maxn, nb, wtime() - J->tstart);
        printf("precision: %s", lname[lk[nl - 1]]);
        if (lk[nl - 1] == 3) printf(" (%d bits, about %d digits)", final_bits, (int)(final_bits * 0.30103));
        printf(", tolerance %.1e%s; ladder:", J->tol, J->tol == 1e-290 && !(tol > 0) ? " (limited to 1e-290: the measures are double-precision numbers)" : "");
        for (int l = 0; l < nl; l++) { if (lk[l] == 3) printf(" mp%d", lb[l]); else printf(" %s", l == 0 ? "d" : lk[l] == 1 ? "dd" : "qd"); }
        printf("\n");
        if (J->verbose > 1) printf("  it prec      pobj                  dobj              pinf     dinf      gap       mu      ap     ad    sigma    time\n");
    }
    int rc = 3, lev, lev0 = 0, skip_pref = 0, lr_rmax = 0, face_dual = 0, levs = 0, next = 0, start_fail = 0;
    struct HPChamp champ_, spare_;
    memset(&champ_, 0, sizeof(champ_)); memset(&spare_, 0, sizeof(spare_));
    spare_.X = (uint64_t **)calloc((size_t)nb, sizeof(uint64_t *)); spare_.Z = (uint64_t **)calloc((size_t)nb, sizeof(uint64_t *));
    J->spare = &spare_;
    champ_.X = (uint64_t **)calloc((size_t)nb, sizeof(uint64_t *)); champ_.Z = (uint64_t **)calloc((size_t)nb, sizeof(uint64_t *));
    J->champ = &champ_;
    /* -bound: the anchors of the certificate, recorded by the interior-point method */
    const int bside = par->bound_side;
    struct HPChamp anc_[HP_NANC];
    memset(anc_, 0, sizeof(anc_));
    if (bside) { J->anc = anc_; J->na = 0; J->anc_thr = 1e-4; }
    double hp_start_err = 1e300;
    /* the double solution in the coordinates of the base problem */
    double **WX = NULL, *Wy = NULL;
    const int warm_ok = warm && warm->nblk == nb_orig && warm->m == m_orig;
    if (warm_ok && dualized) {
        /* dual form: its X is (Z, y+ , y-), its y the negated entries of X (twice off the diagonal) */
        WX = (double **)calloc((size_t)nb_base, sizeof(double *)); Wy = (double *)calloc((size_t)m_base + 1, sizeof(double));
        size_t r = 0;
        for (int k = 0; k < nb_orig; k++) {
            const int n = abs(J_.bs[k]);
            const size_t len = J_.bs[k] < 0 ? (size_t)n : (size_t)n * n;
            WX[k] = (double *)malloc(sizeof(double) * len);
            memcpy(WX[k], warm->Z[k], sizeof(double) * len);
            if (J_.bs[k] < 0) { for (int i = 0; i < n; i++) Wy[r++] = -warm->X[k][i]; continue; }
            for (int i = 0; i < n; i++) for (int j = i; j < n; j++) Wy[r++] = i == j ? -warm->X[k][(size_t)i * n + i] : -(warm->X[k][(size_t)i * n + j] + warm->X[k][(size_t)j * n + i]);
        }
        WX[nb_orig] = (double *)malloc(sizeof(double) * (size_t)(2 * m_orig + 1));
        for (int p2 = 0; p2 < m_orig; p2++) { WX[nb_orig][2 * p2] = warm->y[p2] > 0 ? warm->y[p2] : 0.0; WX[nb_orig][2 * p2 + 1] = warm->y[p2] < 0 ? -warm->y[p2] : 0.0; }
    } else if (warm_ok) { WX = warm->X; Wy = warm->y; }
    if (warm_ok && pre) {
        /* the double solution of the problem as read, carried to the reduced problem: X on the
         * positions that survive (after a face reduction X = V W V' has W there), y on the rows
         * that survive (the eliminations leave their multipliers unchanged), Z = C - A*(y) */
        double nx = 0, nz = 0;
        for (int k = 0; k < nb_base; k++) {
            const int kr = pre->blkmap[k], n0 = abs(JB->bs[k]);
            if (kr < 0) continue;
            const int nr = abs(J->bs[kr]);
            for (int i = 0; i < n0; i++) {
                const int ir = pre->idxmap[k][i];
                if (ir < 0) continue;
                if (JB->bs[k] < 0) { mpx_setd(J->X[kr] + (size_t)ir * Wm, WX[k][i]); nx += WX[k][i] * WX[k][i]; continue; }
                for (int j = 0; j < n0; j++) {
                    const int jr = pre->idxmap[k][j];
                    if (jr < 0) continue;
                    const double v = WX[k][(size_t)i * n0 + j];
                    mpx_setd(J->X[kr] + ((size_t)ir * nr + jr) * Wm, v); nx += v * v;
                }
            }
        }
        for (int r = 0; r < pre->mr; r++) mpx_setd(J->y + (size_t)r * Wm, Wy[pre->rowmap[r]]);
        hp_measure(J, J->X, J->y, J->Z, 1e-8);
        for (int k = 0; k < nb; k++) {
            const size_t len = J->bs[k] < 0 ? (size_t)(-J->bs[k]) : (size_t)J->bs[k] * J->bs[k];
            for (size_t i = 0; i < len; i++) { const double z = mpx_getd(J->Z[k] + i * Wm); nz += z * z; }
        }
        J->best_err = 1e300;
        J->have_iter = 1; lev0 = 1;
        if (dualized) { for (int k = 0; k < nb_base; k++) free(WX[k]); free(WX); free(Wy); }
        if (nx > 0 && nz > 0) J->sigma = sqrt(nx / nz) * (getenv("BRISK_HPSIG0") ? atof(getenv("BRISK_HPSIG0")) : 1e6);
    } else if (warm_ok && !dualized) {
        /* the double solution as the starting point: X, y exactly; sigma from the sizes of X and Z */
        double nx = 0, nz = 0;
        for (int k = 0; k < nb; k++) {
            const size_t len = J->bs[k] < 0 ? (size_t)(-J->bs[k]) : (size_t)J->bs[k] * J->bs[k];
            for (size_t i = 0; i < len; i++) {
                mpx_setd(J->X[k] + i * Wm, warm->X[k][i]); nx += warm->X[k][i] * warm->X[k][i];
                if (warm->Z && warm->Z[k]) { mpx_setd(J->Z[k] + i * Wm, warm->Z[k][i]); nz += warm->Z[k][i] * warm->Z[k][i]; }
            }
        }
        for (int p = 0; p < m; p++) mpx_setd(J->y + (size_t)p * Wm, warm->y[p]);
        J->have_iter = 1; lev0 = 1;
        if (nx > 0 && nz > 0) J->sigma = sqrt(nx / nz) * (getenv("BRISK_HPSIG0") ? atof(getenv("BRISK_HPSIG0")) : 1e6);
    }
    if (lev0 == 1 && J->method == 0) {
        /* interior-point method from a double solution: an interior, centred point near it
         * (hp_interior), and the dependent rows from one factorization at the cold start */
        hp_measure(J, J->X, J->y, J->Z, -1.0);
        const double relerr = J->best_err;
        hp_start_err = relerr;
        const double murel = hp_interior(J, relerr);
        if (J->dep) { J->have_iter = 0; J->deponly = 1; (void)ipm_f64(J); J->deponly = 0; J->have_iter = 1; J->iters = 0; }
        J->best_err = 1e300;
        if (J->verbose > 1) printf("     start: the double solution (error %.1e on this problem), moved to the central path at mu = %.0e (relative)\n", relerr, murel);
    }
    double tlev[40];
    int itlev[40];
    for (int l = 0; l < lev0; l++) { tlev[l] = warm ? warm->time : 0; itlev[l] = warm ? warm->iters : 0; }   /* the double solve (default method) */
    double start_err = 1e300, keep_t[40], ext_err = 1e300;
    const int it_warm = lev0 && warm ? warm->iters : 0;
    HPJob keep; int have_keep = 0, keep_lev = 0, keep_it[40];
    if (lev0 == 1 && J->method == 0) start_err = hp_start_err;
    levs = lev0;
    /* experiment (14 October, a user's suggestion: "start with the selected high precision right from the
     * first iteration"): BRISK_HPLEV0=<k> starts the ladder cold at level k (1 dd, 2 qd, 3 the first mp level)
     * instead of from the double solution */
    if (getenv("BRISK_HPLEV0")) { const int k = atoi(getenv("BRISK_HPLEV0")); if (k >= 0 && k < nl) { J->have_iter = 0; lev0 = 0; levs = k; if (J->verbose > 1) printf("     cold start at level %d\n", k); } }
/* BRISK_HPDBG: the iterate remeasured (Z = C - A*(y) formed again) at the bookkeeping points after the ladder */
#define HP_DBG_MEASURE(tag) do { if (getenv("BRISK_HPDBG") && J->have_iter) { const double e0 = J->best_err, p0 = J->best_pinf, d0 = J->best_dinf; hp_measure(J, J->X, J->y, J->Z, J->tol); printf("       [%s: remeasured err %.1e pinf %.1e dinf %.1e (recorded err %.1e pinf %.1e dinf %.1e)]\n", tag, J->best_err, J->best_pinf, J->best_dinf, e0, p0, d0); J->best_err = e0; J->best_pinf = p0; J->best_dinf = d0; } } while (0)
hp_again:
    for (lev = levs; lev < nl; lev++) {
        /* 24 October 2026 (ISSUES 278): no level is started after the time limit (or an interrupt): a level's start
         * converts and measures the point in its precision before its first limit check (about 2 s in double-double on
         * 9,300 constraints); the run ends as at a level's own limit stop, at the last level that ran */
        if (lev > levs && ((J->timelimit > 0 && wtime() - J->tstart > J->timelimit) || brisk_stop_flag)) { rc = 2; lev--; break; }
        const double t0 = wtime(); const int it0 = J->iters;
        J->level_tol = lev == nl - 1 ? 0.0 : hp_tolf(ldexp(1.0, -(int)(0.58 * lb[lev])));
        {   /* can a higher precision follow? (the next level, or an extension: see below) */
            const int nbx = lk[nl - 1] == 1 ? 212 : 2 * lb[nl - 1] > 64 * (J->Lm - 1) ? 64 * (J->Lm - 1) : 2 * lb[nl - 1];
            J->more = J->method == 0 && lev >= 1 && (lev < nl - 1 || (next < max_ext && nl < 39 && nbx > lb[nl - 1] + 32));
            J->sick = getenv("BRISK_HPSICK") ? atof(getenv("BRISK_HPSICK")) : 0.0;
            if (getenv("BRISK_HPNOHAND")) J->more = 0;
            J->accept = J->level_tol > J->tol ? J->level_tol : tol > 0 ? tol : hp_tolf(ldexp(1.0, -(int)(0.60 * final_bits)));
        }
        if (lk[lev] == 3) { mpx_set_prec(lb[lev]); if (mpx_L > J->Lm) { mpx_L = J->Lm; mpx_W = mpx_L + 2; } }
        else { mpx_L = J->Lm; mpx_W = mpx_L + 2; }
        if (J->method == 2) rc = lk[lev] == 0 ? lr_f64(J) : lk[lev] == 1 ? lr_dd(J) : lk[lev] == 2 ? lr_qd(J) : lr_mp(J);
        else if (J->method == 1) rc = lk[lev] == 0 ? alm_f64(J) : lk[lev] == 1 ? alm_dd(J) : lk[lev] == 2 ? alm_qd(J) : alm_mp(J);
        else {
            /* Interior point. When the double-double level ends exhausted short of its target
             * it is run a second time from the same start with small Schur pivots skipped
             * (below eps^0.9 of their diagonal entry), and the better end is taken: on problems
             * whose Schur complement becomes singular in the limit (no interior: moment
             * relaxations, gpp) skipping gains many digits; on others (hinf2) it loses them.
             * The higher levels use the winner (the decision is made where it is cheap). */
            const int retry = lev == 1 && J->have_iter && !getenv("BRISK_HPNORETRY") && !getenv("BRISK_HPREG");
            const int have0 = J->have_iter;
            uint64_t **sX = NULL, **sZ = NULL, *sy = NULL;
            const size_t Wm_ = (size_t)J->Lm + 2;
            if (retry) {
                sX = (uint64_t **)malloc(sizeof(uint64_t *) * (size_t)nb); sZ = (uint64_t **)malloc(sizeof(uint64_t *) * (size_t)nb);
                for (int k = 0; k < nb; k++) {
                    const size_t len = (J->bs[k] < 0 ? (size_t)(-J->bs[k]) : (size_t)J->bs[k] * J->bs[k]) * Wm_ * sizeof(uint64_t);
                    sX[k] = (uint64_t *)malloc(len); memcpy(sX[k], J->X[k], len);
                    sZ[k] = (uint64_t *)malloc(len); memcpy(sZ[k], J->Z[k], len);
                }
                sy = (uint64_t *)malloc((size_t)m * Wm_ * sizeof(uint64_t)); memcpy(sy, J->y, (size_t)m * Wm_ * sizeof(uint64_t));
            }
#define HP_IPM_LEVEL() (lk[lev] == 0 ? ipm_f64(J) : lk[lev] == 1 ? ipm_dd(J) : lk[lev] == 2 ? ipm_qd(J) : ipm_mp(J))
            if (!getenv("BRISK_HPREG")) J->pivreg = skip_pref ? 0.9 : 0.0;
            const double pm[5] = { J->best_pinf, J->best_dinf, J->best_gap, J->best_compl, J->best_err };
            const int upper = lev > levs || next > 0;       /* a level above another one (an extension level starts at levs) */
            const int have_spare = spare_.have && upper;
            spare_.have = 0;
            J->best_err = 1e300; J->minpiv = 1e300; J->ndyn = 0; J->nwould = 0;
            rc = HP_IPM_LEVEL();
            if (rc == 3 && upper && have0 && have_spare) {
                /* the best iterate of the level below is not positive definite in this precision
                 * (it was at the rounding level of the one below: hinf9); its handover iterate
                 * is tried, and the best one stays the result if this level does not do better */
                if (J->verbose > 1) printf("     the level starts from the handover iterate of the one below (error %.1e)\n", spare_.m[4]);
                if (!champ_.have || pm[4] < champ_.m[4]) {
                    for (int k = 0; k < nb; k++) { uint64_t *t1 = champ_.X[k]; champ_.X[k] = J->X[k]; J->X[k] = t1; t1 = champ_.Z[k]; champ_.Z[k] = J->Z[k]; J->Z[k] = t1; }
                    { uint64_t *t1 = champ_.y; champ_.y = J->y; J->y = t1; t1 = champ_.pobj; champ_.pobj = J->pobj; J->pobj = t1; t1 = champ_.dobj; champ_.dobj = J->dobj; J->dobj = t1; }
                    for (int q = 0; q < 5; q++) champ_.m[q] = pm[q];
                    champ_.have = 1;
                }
                for (int k = 0; k < nb; k++) {
                    const size_t len = (J->bs[k] < 0 ? (size_t)(-J->bs[k]) : (size_t)J->bs[k] * J->bs[k]) * Wm_ * sizeof(uint64_t);
                    if (!J->X[k]) { J->X[k] = (uint64_t *)malloc(len); J->Z[k] = (uint64_t *)malloc(len); }
                    memcpy(J->X[k], spare_.X[k], len); memcpy(J->Z[k], spare_.Z[k], len);
                }
                if (!J->y) { J->y = (uint64_t *)malloc((size_t)m * Wm_ * sizeof(uint64_t) + 8); J->pobj = (uint64_t *)calloc(Wm_, sizeof(uint64_t)); J->dobj = (uint64_t *)calloc(Wm_, sizeof(uint64_t)); }
                memcpy(J->y, spare_.y, (size_t)m * Wm_ * sizeof(uint64_t)); memcpy(J->pobj, spare_.pobj, Wm_ * sizeof(uint64_t)); memcpy(J->dobj, spare_.dobj, Wm_ * sizeof(uint64_t));
                J->best_pinf = spare_.m[0]; J->best_dinf = spare_.m[1]; J->best_gap = spare_.m[2]; J->best_compl = spare_.m[3];
                J->best_err = 1e300;
                if (lk[lev] == 3) { mpx_set_prec(lb[lev]); if (mpx_L > J->Lm) { mpx_L = J->Lm; mpx_W = mpx_L + 2; } }
                rc = HP_IPM_LEVEL();
                if (rc == 3) { J->best_err = spare_.m[4]; rc = 1; start_fail = 1; J->level_best = spare_.m[4]; }
            } else if (rc == 3 && upper && have0) {
                /* the iterate of the level below is not positive definite in this precision (it
                 * was at the rounding level of the one below: hinf9): the solve ends with it */
                J->best_pinf = pm[0]; J->best_dinf = pm[1]; J->best_gap = pm[2]; J->best_compl = pm[3]; J->best_err = pm[4];
                rc = 1; start_fail = 1; J->level_best = pm[4];
            }
            if (!(rc == 0 || rc == 1) || !J->have_iter) J->level_best = 1e300;
            /* (the second run would be the same run when no pivot is, or would be, skipped) */
            const double accept = tol > 0 ? tol : hp_tolf(ldexp(1.0, -(int)(0.60 * final_bits)));
            const double target = J->level_tol > J->tol ? J->level_tol : accept;
            const int differs = skip_pref ? J->ndyn > 0 : J->nwould > 0;
            if (retry && rc == 1 && !start_fail && J->level_best > target && differs) {
                /* keep the first end, restore the start, run the other variant */
                HPJob A = *J;
                uint64_t **aX = J->X, **aZ = J->Z, *ay = J->y, *ap_ = J->pobj, *ad_ = J->dobj;
                J->X = sX; J->Z = sZ; J->y = sy;
                J->pobj = (uint64_t *)calloc(Wm_, sizeof(uint64_t)); J->dobj = (uint64_t *)calloc(Wm_, sizeof(uint64_t));
                J->have_iter = have0; J->best_err = 1e300; J->minpiv = 1e300; J->ndyn = 0; J->nwould = 0;
                J->pivreg = skip_pref ? 0.0 : 0.9;
                if (lk[lev] == 3) { mpx_set_prec(lb[lev]); if (mpx_L > J->Lm) { mpx_L = J->Lm; mpx_W = mpx_L + 2; } }
                const int rc2 = HP_IPM_LEVEL();
                if (!(rc2 == 0 || rc2 == 1) || !J->have_iter) J->level_best = 1e300;
                const int better = (rc2 == 0 || rc2 == 1) && J->have_iter && J->level_best < A.level_best;
                if (J->verbose > 1) printf("     second run of this level with small Schur pivots %s: error %.1e against %.1e\n", skip_pref ? "kept" : "skipped", J->level_best, A.level_best);
                if (better) {
                    for (int k = 0; k < nb; k++) { free(aX[k]); free(aZ[k]); }
                    free(aX); free(aZ); free(ay); free(ap_); free(ad_);
                    rc = rc2; skip_pref = !skip_pref;
                } else {
                    const int its = J->iters, nbis = J->nbis;
                    for (int k = 0; k < nb; k++) { free(J->X[k]); free(J->Z[k]); }
                    free(J->X); free(J->Z); free(J->y); free(J->pobj); free(J->dobj);
                    *J = A; J->iters = its; J->nbis = nbis;
                }
                sX = NULL; sZ = NULL; sy = NULL;
            }
            if (sX) { for (int k = 0; k < nb; k++) { free(sX[k]); free(sZ[k]); } free(sX); free(sZ); free(sy); }
#undef HP_IPM_LEVEL
        }
        mpx_L = J->Lm; mpx_W = mpx_L + 2;
        tlev[lev] = wtime() - t0; itlev[lev] = J->iters - it0;
        if (rc != 1 || start_fail) break;
    }
    if (lev0 == 1 && next == 0 && J->method == 0 && (rc == 3 || (rc == 1 && fmin(J->best_err, champ_.have ? champ_.m[4] : 1e300) > start_err)) && !brisk_stop_flag
        && !(J->timelimit > 0 && wtime() - J->tstart > J->timelimit)) {     /* (24 October 2026, ISSUES 278: not after the limit) */
        /* the start from the double solution did not work (not positive definite as doubles,
         * or the solve ended worse than it began): the cold start with its own double level */
        if (J->verbose > 1) printf("     the start from the double solution failed (error %.1e): cold start\n", J->best_err);
        lev0 = 0; skip_pref = 0; J->dep_known = 0;
        if (J->dep) memset(J->dep, 0, sizeof(int) * (size_t)m);
        /* (the end of the first attempt is kept: the cold start may end worse still) */
        if (rc == 1 && J->have_iter) {
            keep = *J; have_keep = 1; keep_lev = lev == nl ? nl - 1 : lev;
            for (int l = 0; l < nl; l++) { keep_t[l] = tlev[l]; keep_it[l] = itlev[l]; }
            J->X = (uint64_t **)calloc((size_t)nb, sizeof(uint64_t *)); J->Z = (uint64_t **)calloc((size_t)nb, sizeof(uint64_t *));
            for (int k = 0; k < nb; k++) {
                const size_t len = J->bs[k] < 0 ? (size_t)(-J->bs[k]) : (size_t)J->bs[k] * J->bs[k];
                J->X[k] = (uint64_t *)calloc(len * Wm, sizeof(uint64_t)); J->Z[k] = (uint64_t *)calloc(len * Wm, sizeof(uint64_t));
            }
            J->y = (uint64_t *)calloc((size_t)m * Wm, sizeof(uint64_t));
            J->pobj = (uint64_t *)calloc(Wm, sizeof(uint64_t)); J->dobj = (uint64_t *)calloc(Wm, sizeof(uint64_t));
        }
        J->have_iter = 0; J->best_err = 1e300;
        levs = 0;
        goto hp_again;
    }
    HP_DBG_MEASURE("after the ladder loop");
    if (have_keep) {
        const int cold_better = (rc == 0 || rc == 1) && J->have_iter && J->best_err < keep.best_err;
        if (J->verbose > 1) printf("     cold start: error %.1e against %.1e of the first attempt\n", J->best_err, keep.best_err);
        if (rc == 2 || cold_better) {
            for (int k = 0; k < nb; k++) { free(keep.X[k]); free(keep.Z[k]); }
            free(keep.X); free(keep.Z); free(keep.y); free(keep.pobj); free(keep.dobj);
        } else {
            const int its = J->iters, nbis = J->nbis;
            for (int k = 0; k < nb; k++) { free(J->X[k]); free(J->Z[k]); }
            free(J->X); free(J->Z); free(J->y); free(J->pobj); free(J->dobj);
            *J = keep; J->iters = its; J->nbis = nbis; rc = 1; lev = keep_lev == nl - 1 ? nl : keep_lev;
            for (int l = 1; l < nl; l++) { tlev[l] += keep_t[l]; itlev[l] += keep_it[l]; }
            tlev[0] = keep_t[0]; itlev[0] = keep_it[0];
        }
        have_keep = 0;
    }
    if (J->method == 0 && rc == 1 && lev == nl && next < max_ext && nl < 39 && !brisk_stop_flag
        && (next == 0 || J->sick_exit || J->chol_exit || J->best_err < 0.1 * ext_err)      /* (a further one only while the last gained a digit, or when it ended on a factorization that needs more digits) */
        && J->best_err > (tol > 0 ? tol : hp_tolf(ldexp(1.0, -(int)(0.60 * final_bits)))) && J->best_err < 1e-3) {
        /* Extension: the precision asked for is exhausted short of the tolerance (a problem
         * without an interior ends near eps^0.45). The solve continues from its last iterate in
         * the next precision (dd -> qd -> twice the bits ...), with the tolerance unchanged. */
        const int kb = lk[nl - 1], maxbits = 64 * (J->Lm - 1);
        int nbits = kb == 1 ? 212 : 2 * lb[nl - 1];
        if (nbits > maxbits) nbits = maxbits;
        if (nbits > lb[nl - 1] + 32) {
            if (J->verbose > 1) printf("     the precision is exhausted at the error %.1e: continuing in %d bits\n", J->best_err, nbits);
            lk[nl] = kb == 1 ? 2 : 3; lb[nl] = nbits; nl++; next++; ext_err = J->best_err;
            levs = nl - 1;
            goto hp_again;
        }
    }
    if (lev == nl) lev = nl - 1;
    /* a level that handed over an earlier iterate kept its best one: taken back when the
     * levels after it ended worse */
    HP_DBG_MEASURE("after the keep decision");
    if (champ_.have && (rc == 1 || rc == 2 || rc == 3) && (!J->have_iter || rc == 3 || champ_.m[4] < J->best_err)) {
        if (rc == 3) rc = 1;
        if (J->verbose > 1) printf("     the best iterate is the one before the handover (error %.1e against %.1e)\n", champ_.m[4], J->best_err);
        for (int k = 0; k < nb; k++) { uint64_t *t1 = J->X[k]; J->X[k] = champ_.X[k]; champ_.X[k] = t1; t1 = J->Z[k]; J->Z[k] = champ_.Z[k]; champ_.Z[k] = t1; }
        { uint64_t *t1 = J->y; J->y = champ_.y; champ_.y = t1; t1 = J->pobj; J->pobj = champ_.pobj; champ_.pobj = t1; t1 = J->dobj; J->dobj = champ_.dobj; champ_.dobj = t1; }
        J->best_pinf = champ_.m[0]; J->best_dinf = champ_.m[1]; J->best_gap = champ_.m[2]; J->best_compl = champ_.m[3]; J->best_err = champ_.m[4];
        J->have_iter = 1;
    }
    HP_DBG_MEASURE("after the champion decision");
    for (int k = 0; k < nb; k++) { free(champ_.X[k]); free(champ_.Z[k]); free(spare_.X[k]); free(spare_.Z[k]); }
    free(champ_.X); free(champ_.Z); free(champ_.y); free(champ_.pobj); free(champ_.dobj);
    free(spare_.X); free(spare_.Z); free(spare_.y); free(spare_.pobj); free(spare_.dobj);
    J->champ = NULL; J->spare = NULL;
    /* Without -hptol the solve aims at 2^(-0.75 bits) and a result within 2^(-0.6 bits) counts
     * as optimal: ill-conditioned problems (control) end between the two, as accurate as the
     * precision allows. */
    if (rc == 1 && !(tol > 0) && J->best_err <= hp_tolf(ldexp(1.0, -(int)(0.60 * final_bits)))) rc = 0;
    /* -bound: the anchors and the direction of the face multiplier on the problem as read */
    HPPoint ap[HP_NANC]; int nap = 0;
    uint64_t **DirX = NULL, *diry = NULL;
    J->anc = NULL;
    if (bside && J->have_iter && rc != 4 && rc != 5) {
        for (int a = 0; a < HP_NANC; a++) {
            if (!anc_[a].have || !anc_[a].y) continue;
            ap[nap].X = hpc_alloc_blocks(&J_); ap[nap].y = (uint64_t *)calloc(((size_t)m_orig + 1) * Wm, sizeof(uint64_t)); ap[nap].err = anc_[a].m[4];
            hp_map_back(pre, JB, J, dualized, &J_, anc_[a].X, anc_[a].y, 0.0, ap[nap].X, ap[nap].y);
            nap++;
        }
        for (int a = 1; a < nap; a++) { const HPPoint v = ap[a]; int b = a; while (b > 0 && ap[b - 1].err < v.err) { ap[b] = ap[b - 1]; b--; } ap[b] = v; }   /* older (larger error) first */
        const int sface = dualized ? 1 : 2;       /* the side whose certificate takes the face reduction's direction */
        if (pre && pre->nface && (bside & sface)) {
            uint64_t **X1 = hpc_alloc_blocks(&J_), **X2 = hpc_alloc_blocks(&J_);
            uint64_t *y1 = (uint64_t *)calloc(((size_t)m_orig + 1) * Wm, sizeof(uint64_t)), *y2 = (uint64_t *)calloc(((size_t)m_orig + 1) * Wm, sizeof(uint64_t));
            hp_map_back(pre, JB, J, dualized, &J_, J->X, J->y, 0.0, X1, y1);
            hp_map_back(pre, JB, J, dualized, &J_, J->X, J->y, 1.0, X2, y2);
            if (sface == 1) {
                for (int k = 0; k < nb_orig; k++) { const size_t len = J_.bs[k] < 0 ? (size_t)(-J_.bs[k]) : (size_t)J_.bs[k] * J_.bs[k]; for (size_t i = 0; i < len; i++) mpx_sub(X2[k] + i * Wm, X2[k] + i * Wm, X1[k] + i * Wm); }
                DirX = X2; hpc_free_blocks(X1, nb_orig); free(y1); free(y2);
            } else {
                for (int c = 0; c < m_orig; c++) mpx_sub(y2 + (size_t)c * Wm, y2 + (size_t)c * Wm, y1 + (size_t)c * Wm);
                diry = y2; hpc_free_blocks(X1, nb_orig); hpc_free_blocks(X2, nb_orig); free(y1);
            }
        }
    }
    for (int a = 0; a < HP_NANC; a++) { if (anc_[a].X) for (int k = 0; k < nb; k++) free(anc_[a].X[k]); free(anc_[a].X); free(anc_[a].y); }
    if (pre) {
        /* back to the problem as read: X and y through the records of the presolve,
         * Z = C - A*(y) and the errors on the original data, in the master precision */
        HPJob *JR = J;
        J = JB;
        nb = nb_base; m = m_base;
        J->X = (uint64_t **)calloc((size_t)nb, sizeof(uint64_t *)); J->Z = (uint64_t **)calloc((size_t)nb, sizeof(uint64_t *));
        for (int k = 0; k < nb; k++) {
            const size_t len = J->bs[k] < 0 ? (size_t)(-J->bs[k]) : (size_t)J->bs[k] * J->bs[k];
            J->X[k] = (uint64_t *)calloc(len * Wm, sizeof(uint64_t)); J->Z[k] = (uint64_t *)calloc(len * Wm, sizeof(uint64_t));
        }
        J->y = (uint64_t *)calloc((size_t)m * Wm, sizeof(uint64_t));
        J->pobj = (uint64_t *)calloc(Wm, sizeof(uint64_t)); J->dobj = (uint64_t *)calloc(Wm, sizeof(uint64_t));
        J->iters = JR->iters; J->maxit = JR->maxit; J->verbose = JR->verbose; J->method = JR->method; J->tol = JR->tol;
        J->best_pinf = JR->best_pinf; J->best_dinf = JR->best_dinf; J->best_gap = JR->best_gap; J->best_compl = JR->best_compl; J->best_err = JR->best_err;
        J->have_iter = JR->have_iter; J->ray_res = JR->ray_res;
        if (JR->have_iter) {
            const double red_err = JR->best_err;
            const double accept0 = tol > 0 ? tol : hp_tolf(ldexp(1.0, -(int)(0.60 * final_bits)));
            if (getenv("BRISK_HPDBG")) { const double e0 = JR->best_err, p0 = JR->best_pinf; hp_measure(JR, JR->X, JR->y, JR->Z, JR->tol); printf("       [reduced problem remeasured: err %.1e pinf %.1e dinf %.1e gap %.1e (solver: err %.1e pinf %.1e)]\n", JR->best_err, JR->best_pinf, JR->best_dinf, JR->best_gap, e0, p0); }
            hp_postsolve(pre, J, JR, JR->X, JR->y, J->X, J->y, 0.0);
            hp_measure(J, J->X, J->y, J->Z, JR->tol);
            /* After a face reduction the dual of the problem as read is in general not attained.
             * The multipliers of the removed constraints stay zero: Z = C - A*(y) is then
             * positive semidefinite on the face (V' Z V is the Z of the reduced problem), which
             * with A_p X = 0 for the certificates is all that weak duality needs: b'y is a lower
             * bound and <X, Z> the gap. Its semidefiniteness is therefore measured on the face
             * (the reduced problem's value). BRISK_HPFACEDUAL: instead a multiplier large enough
             * for Z itself to be positive semidefinite to the tolerance (of the order 1 / tol). */
            if (pre->nface) {
                double tface = 0;
                for (int tr = 0; getenv("BRISK_HPFACEDUAL") && J->best_dinf > accept0 && tr < 5; tr++) {
                    tface = (tr == 0 ? 1.0 : tface * accept0 * 1e4) / accept0;
                    hp_postsolve(pre, J, JR, JR->X, JR->y, J->X, J->y, tface);
                    hp_measure(J, J->X, J->y, J->Z, JR->tol);
                }
                if (tface > 0) { if (J->verbose) printf("presolve: the constraints removed by face reductions have the multiplier %.0e\n", tface); }
                else {
                    J->best_dinf = JR->best_dinf; face_dual = 1;
                    J->best_err = fmax(fmax(J->best_pinf, J->best_dinf), fmax(fabs(J->best_gap), fabs(J->best_compl)));
                }
            }
            if (J->verbose > 1) printf("     postsolve: error %.1e on the reduced problem, %.1e on the problem as read\n", red_err, J->best_err);
            const double accept = tol > 0 ? tol : hp_tolf(ldexp(1.0, -(int)(0.60 * final_bits)));
            if (rc == 0 && J->best_err > 10.0 * accept) rc = 1;
        }
        if (JR->method == 2) { lr_rmax = 0; for (int k = 0; k < JR->nblk; k++) if (JR->bs[k] > 0 && JR->lr_rk[k] > lr_rmax) lr_rmax = JR->lr_rk[k]; J->lr_delta = JR->lr_delta; }
        free(JR->dep); JR->dep = NULL;
        lr_free(JR); hp_free(JR);
        hp_pre_free(pre, nb_base);
    }
    double hp_xpsd = 0;     /* (how far X is from positive semidefinite where it was measured: the dual form's slack; ISSUES 281) */
    if (dualized) {
        /* from the dual form back to the problem as read: X = the dual slack of the dual form
         * (formed by the measure above), y = y+ - y-; then the errors on the data as read */
        HPJob *JD = J;
        J = &J_;
        nb = nb_orig; m = m_orig;
        J->X = (uint64_t **)calloc((size_t)nb, sizeof(uint64_t *)); J->Z = (uint64_t **)calloc((size_t)nb, sizeof(uint64_t *));
        for (int k = 0; k < nb; k++) {
            const size_t len = J->bs[k] < 0 ? (size_t)(-J->bs[k]) : (size_t)J->bs[k] * J->bs[k];
            J->X[k] = (uint64_t *)calloc(len * Wm, sizeof(uint64_t)); J->Z[k] = (uint64_t *)calloc(len * Wm, sizeof(uint64_t));
            if (JD->have_iter) memcpy(J->X[k], JD->Z[k], len * Wm * sizeof(uint64_t));
        }
        J->y = (uint64_t *)calloc((size_t)m * Wm, sizeof(uint64_t));
        J->pobj = (uint64_t *)calloc(Wm, sizeof(uint64_t)); J->dobj = (uint64_t *)calloc(Wm, sizeof(uint64_t));
        J->iters = JD->iters; J->maxit = JD->maxit; J->verbose = JD->verbose; J->method = JD->method; J->tol = JD->tol;
        J->best_pinf = JD->best_pinf; J->best_dinf = JD->best_dinf; J->best_gap = JD->best_gap; J->best_compl = JD->best_compl; J->best_err = JD->best_err;
        J->have_iter = JD->have_iter; J->lr_delta = JD->lr_delta; J->ray_res = JD->ray_res;
        /* 23 October 2026: an infeasibility verdict of the dual form names its own sides: its primal is the dual of the
         * problem as read and its dual the primal, so the verdict changes sides here (until now PRIMAL INFEASIBLE was
         * reported for a problem whose dual is infeasible and the other way round, exit codes 11 and 12 swapped) */
        if (rc == 4 || rc == 5) rc = 9 - rc;
        if (JD->have_iter) {
            const double xpsd = JD->best_dinf;                 /* how far X (the dual form's slack) is from positive semidefinite */
            hp_xpsd = xpsd;
            if (face_dual) face_dual = 2;                      /* a face reduction in the dual form: X is positive semidefinite on the face of Z only */
            for (int p2 = 0; p2 < m; p2++) mpx_sub(J->y + (size_t)p2 * Wm, JD->X[nb] + (size_t)(2 * p2) * Wm, JD->X[nb] + (size_t)(2 * p2 + 1) * Wm);
            hp_measure(J, J->X, J->y, J->Z, JD->tol);
            if (xpsd > J->best_err) J->best_err = xpsd;
            if (J->verbose > 1) printf("     dual form undone: error %.1e on the problem as read (X positive semidefinite to %.1e)\n", J->best_err, xpsd);
            const double accept = tol > 0 ? tol : hp_tolf(ldexp(1.0, -(int)(0.60 * final_bits)));
            if (rc == 0 && J->best_err > 10.0 * accept) rc = 1;
            if (rc == 1 && !(tol > 0) && J->best_err <= accept) rc = 0;
        }
        hp_free(JD);
    }
    if (reduced) {
        /* 5.9: back through the reductions; the errors on the problem before them */
        uint64_t **Xf = NULL, *yf = NULL;
        const int have = J->have_iter;
        const double red_err = J->best_err;
        if (have) hp_red_back(&red_, J->X, J->y, &Xf, &yf);
        HPJob Jf = red_.Jfull;
        Jf.tstart = J->tstart; Jf.iters = J->iters; Jf.maxit = J->maxit; Jf.verbose = J->verbose; Jf.method = J->method; Jf.tol = J->tol;
        Jf.best_pinf = J->best_pinf; Jf.best_dinf = J->best_dinf; Jf.best_gap = J->best_gap; Jf.best_compl = J->best_compl; Jf.best_err = J->best_err;
        Jf.have_iter = have; Jf.lr_delta = J->lr_delta; Jf.ray_res = J->ray_res;
        if (warm3_.X) { for (int k = 0; k < J->nblk; k++) { free(warm3_.X[k]); if (warm3_.Z) free(warm3_.Z[k]); } free(warm3_.X); free(warm3_.Z); free(warm3_.y); memset(&warm3_, 0, sizeof(warm3_)); }
        free(J->dep); J->dep = NULL; lr_free(J); hp_free(J);
        J_ = Jf; J = &J_;
        nb = J->nblk; m = J->m;
        J->Z = (uint64_t **)calloc((size_t)nb, sizeof(uint64_t *));
        if (!have) { Xf = (uint64_t **)calloc((size_t)nb, sizeof(uint64_t *)); yf = (uint64_t *)calloc(((size_t)m + 1) * Wm, sizeof(uint64_t)); }
        for (int k = 0; k < nb; k++) {
            const size_t len = J->bs[k] < 0 ? (size_t)(-J->bs[k]) : (size_t)J->bs[k] * J->bs[k];
            J->Z[k] = (uint64_t *)calloc(len * Wm, sizeof(uint64_t));
            if (!have) Xf[k] = (uint64_t *)calloc(len * Wm, sizeof(uint64_t));
        }
        J->X = Xf; J->y = yf;
        J->pobj = (uint64_t *)calloc(Wm, sizeof(uint64_t)); J->dobj = (uint64_t *)calloc(Wm, sizeof(uint64_t));
        hp_red_free(&red_);
        if (have) {
            hp_measure(J, J->X, J->y, J->Z, J->tol);
            if (J->verbose > 1 || getenv("BRISK_HPDBG")) printf("     reductions undone: error %.1e on the reduced problem, %.1e on the problem before them\n", red_err, J->best_err);
            const double accept = tol > 0 ? tol : hp_tolf(ldexp(1.0, -(int)(0.60 * final_bits)));
            if (rc <= 1 && J->best_err > 10.0 * J->tol && J->best_err > 100.0 * red_err) {
                /* a reduction that does not hold in the working precision (a symmetry of the
                 * data rounded to doubles only): the solve is repeated without them */
                if (J->verbose) printf("presolve: the reductions do not hold in the working precision (error %.1e on the problem as read against %.1e): solving again without them\n", J->best_err, red_err);
                hp_free(J);
                if (split) hp_split_free(&sp_);
                g_hp_nored = 1;
                const int rc2 = brisk_hp_run(fname, data, res, warm_in == &warm2_ ? NULL : warm_in, par, kind, digits, tol, yfile, xfile, zfile);
                g_hp_nored = 0;
                return rc2;
            }
            if (rc == 0 && J->best_err > 10.0 * accept) rc = 1;
        }
    }
    /* 24 October 2026 (ISSUES 278): the double solution this solve started from is the result when the solve ended
     * worse than it (a time limit reached in the first iterations of a level returned the iterate after one iteration,
     * error 2e3, instead of the double solution, 2.3e-4). Both are measured alike on the problem as read: the residuals
     * and the gap of hp_measure in the working precision, and how far Z = C - A*(y) and X are from positive
     * semidefinite by their eigenvalues in double precision, with bounds for the eigenvalues' error (hp_psd_viol_d; a
     * grid of Cholesky factorizations in the working precision costs seconds on a block of order 250, more than a time
     * limit may have left; the solve's X is interior only in its own coordinates). The double solution is returned
     * only when its error's upper bound is below the result's lower bound: on hinf11 (SDPLIB, SDP), where |Z| is about
     * 1e7 times |C|, the bounds are about 1.5e-7 wide, the two points cannot be told apart, and the result (5.6e-8 by
     * its own measure) stays. Only when the double
     * solve's own errors are within ten times the result's (the measures' norms differ), so that a solve that went
     * past the double solution pays nothing. */
    double res_double = 0;          /* (> 0: the result is the double solution; the method's own result's error) */
    if (warm_arg && warm_arg->X && warm_arg->y && (rc == 1 || rc == 2 || rc == 3) && warm_arg->m == J->m
        && warm_arg->nblk == (split ? sp_.nb0 : J->nblk)) {
        double werr = 0;
        for (int e = 1; e <= 6; e++) werr = fmax(werr, fabs(warm_arg->err[e]));
        if (!J->have_iter || werr < 10.0 * J->best_err) {
            double nb2 = 0, nc2 = 0;
            for (int c = 0; c < J->m; c++) { const double bc = mpx_getd(J->b + (size_t)c * Wm); nb2 += bc * bc; }
            for (int k = 0; k < J->nblk; k++)
                for (size_t e = J->cptr[k]; e < J->cptr[k + 1]; e++) { const double v = mpx_getd(J->Cv + e * Wm); nc2 += (J->bs[k] > 0 && J->ci[e] != J->cj[e] ? 2 : 1) * v * v; }
            const double xscale = 1 + sqrt(nb2), zscale = 1 + sqrt(nc2);
            double **WXd = split ? hp_split_fwd_d(&sp_, J, warm_arg->X) : warm_arg->X;
            uint64_t **WX2 = hpc_alloc_blocks(J), **WZ2 = hpc_alloc_blocks(J), **Zt = hpc_alloc_blocks(J);
            uint64_t *Wy2 = (uint64_t *)calloc(((size_t)J->m + 1) * Wm, sizeof(uint64_t));
            for (int k = 0; k < J->nblk; k++) {
                const size_t len = J->bs[k] < 0 ? (size_t)(-J->bs[k]) : (size_t)J->bs[k] * J->bs[k];
                for (size_t i = 0; i < len; i++) mpx_setd(WX2[k] + i * Wm, WXd[k][i]);
            }
            for (int p = 0; p < J->m; p++) mpx_setd(Wy2 + (size_t)p * Wm, warm_arg->y[p]);
            if (split) { for (int k = 0; k < J->nblk; k++) free(WXd[k]); free(WXd); }
            const double own[5] = { J->best_pinf, J->best_dinf, J->best_gap, J->best_compl, J->best_err };
            if (!J->pobj) J->pobj = (uint64_t *)calloc(Wm, sizeof(uint64_t));      /* (hp_measure writes them; a solve without a point may not have them) */
            if (!J->dobj) J->dobj = (uint64_t *)calloc(Wm, sizeof(uint64_t));
            uint64_t po[MPX_MAXL + 2], dob[MPX_MAXL + 2];
            mpx_copy(po, J->pobj); mpx_copy(dob, J->dobj);
            double own_err = INFINITY, lo1, hi1, lo2, hi2;
            if (J->have_iter) {
                hp_measure(J, J->X, J->y, Zt, -1.0);
                hp_psd_viol_d(J, Zt, zscale, &lo1, &hi1); hp_psd_viol_d(J, J->X, xscale, &lo2, &hi2);
                own_err = fmax(J->best_err, fmax(lo1, lo2));          /* (a lower bound) */
            }
            hp_measure(J, WX2, Wy2, WZ2, -1.0);
            hp_psd_viol_d(J, WZ2, zscale, &lo1, &hi1); hp_psd_viol_d(J, WX2, xscale, &lo2, &hi2);
            const double w_dinf = hi1, w_err = fmax(J->best_err, fmax(hi1, hi2));   /* (an upper bound) */
            if (w_err < own_err) {
                if (J->verbose) printf("high precision: the double solution is returned: its error on the problem as read is %.1e, that of the method's own result %.1e (both relative to the 2-norms of b and C)\n", w_err, own_err);
                hpc_free_blocks(J->X, J->nblk); hpc_free_blocks(J->Z, J->nblk); free(J->y);
                J->X = WX2; J->Z = WZ2; J->y = Wy2; WX2 = WZ2 = NULL; Wy2 = NULL;
                J->best_dinf = w_dinf; J->best_err = w_err; J->have_iter = 1; res_double = own_err; hp_xpsd = hi2;
                if (rc == 3) rc = 1;
            } else {
                J->best_pinf = own[0]; J->best_dinf = own[1]; J->best_gap = own[2]; J->best_compl = own[3]; J->best_err = own[4];
                mpx_copy(J->pobj, po); mpx_copy(J->dobj, dob);
            }
            hpc_free_blocks(WX2, J->nblk); hpc_free_blocks(WZ2, J->nblk); hpc_free_blocks(Zt, J->nblk); free(Wy2);
        }
    }
    /* -bound: the certificate of each side asked for on the problem as read (hpcert.inc),
     * beside the returned point (which stays the solve's own; up to 1.5.1 the certificate
     * replaced its side of it) */
    HPCertOut cert[2]; int have_cert[2] = { 0, 0 }, tried_cert[2] = { 0, 0 }; char cert_src[2][200]; double t_cert[2] = { 0, 0 };
    uint64_t **certX = NULL, *certy = NULL;
    int prior_[2] = { 0, 0 };                   /* certified by a-priori bounds */
    memset(cert, 0, sizeof(cert)); cert_src[0][0] = cert_src[1][0] = 0;
    for (int sd = 1; sd <= 2; sd++) {
        if (!(bside & sd) || !J->have_iter || rc == 4 || rc == 5) continue;
        const double tc0 = wtime();
        PSOrig *Oc = hp_psorig(J);
        tried_cert[sd - 1] = 1;
        uint64_t **Xc = sd == 1 ? hpc_alloc_blocks(J) : NULL;
        uint64_t *yc = sd == 2 ? (uint64_t *)calloc(((size_t)J->m + 1) * Wm, sizeof(uint64_t)) : NULL;
        have_cert[sd - 1] = hp_bound(J, Oc, data != NULL, sd, nap, ap, sd == 1 ? DirX : NULL, sd == 2 ? diry : NULL, J->verbose, &cert[sd - 1],
                                     cert_src[sd - 1], sizeof(cert_src[0]), Xc, yc);
        if (!have_cert[sd - 1] && par->bound_prior) {
            /* no margin: a-priori bounds of the other side (boundcert.c), in double precision, for
             * the returned point rounded to doubles (valid for any point; the bound has double accuracy) */
            double **Xd = NULL, *yd = NULL, B; char w2[160];
            if (sd == 1) {
                Xd = bound_alloc_blocks(Oc);
                for (int k = 0; k < J->nblk; k++) { const size_t len = J->bs[k] < 0 ? (size_t)(-J->bs[k]) : (size_t)J->bs[k] * J->bs[k]; for (size_t i = 0; i < len; i++) Xd[k][i] = mpx_getd(J->X[k] + i * Wm); }
            } else { yd = (double *)malloc(sizeof(double) * ((size_t)J->m + 1)); for (int p = 0; p < J->m; p++) yd[p] = mpx_getd(J->y + (size_t)p * Wm); }
            if (brisk_prior_bound(Oc, sd, Xd, NULL, yd, NULL, data != NULL, J->verbose, &B, NULL, w2, sizeof(w2))) {
                have_cert[sd - 1] = 1; prior_[sd - 1] = 1;
                mpx_setd(cert[sd - 1].bound, B);
                if (sd == 1) { for (int k = 0; k < J->nblk; k++) { const size_t len = J->bs[k] < 0 ? (size_t)(-J->bs[k]) : (size_t)J->bs[k] * J->bs[k]; for (size_t i = 0; i < len; i++) mpx_setd(Xc[k] + i * Wm, Xd[k][i]); } }
                else for (int p = 0; p < J->m; p++) mpx_setd(yc + (size_t)p * Wm, yd[p]);
                if (sd == 1) mpx_copy(cert[sd - 1].value, J->pobj); else mpx_copy(cert[sd - 1].value, J->dobj);
                snprintf(cert_src[sd - 1], sizeof(cert_src[0]), "a-priori bounds of %s (double precision; the margin check failed: %.90s); the returned point",
                         sd == 1 ? "y" : "the diagonal of X", cert[sd - 1].why);
            } else { const size_t l = strlen(cert[sd - 1].why); snprintf(cert[sd - 1].why + l, sizeof(cert[0].why) - l, "; a-priori bounds: %.100s", w2); }
            if (Xd) bound_free_blocks(Xd, Oc->nblk);
            free(yd);
        }
        ps_orig_free(Oc);
        if (have_cert[sd - 1] && sd == 1) certX = Xc; else hpc_free_blocks(Xc, sd == 1 ? J->nblk : 0);
        if (have_cert[sd - 1] && sd == 2) certy = yc; else free(yc);
        t_cert[sd - 1] = wtime() - tc0;
    }
    for (int a = 0; a < nap; a++) { hpc_free_blocks(ap[a].X, nb_orig); free(ap[a].y); }
    hpc_free_blocks(DirX, nb_orig); free(diry);
    /* 23 October 2026 (ISSUES 256): an infeasibility verdict checked by the solution of its ray problem (hp_ray_check;
     * BRISK_HPRAYCHECK=0: not) */
    int ray_cert = -1, ray_prec = 0; double ray_viol = INFINITY, t_ray = 0;
    if ((rc == 4 || rc == 5) && !g_hp_inray && !bside && !(getenv("BRISK_HPRAYCHECK") && atoi(getenv("BRISK_HPRAYCHECK")) == 0)) {
        const double tr0 = wtime();
        PSOrig *Oc = hp_psorig(J);
        ray_cert = hp_ray_check(Oc, rc == 5, par, par->timelimit > 0 ? par->timelimit - (tr0 - J->tstart) : 0, kind, digits, &ray_viol, &ray_prec);
        ps_orig_free(Oc);
        t_ray = wtime() - tr0;
    }
    const char *status = rc == 0 ? "OPTIMAL" : rc == 1 ? "PRECISION LIMIT" : rc == 2 ? (brisk_stop_flag ? "INTERRUPTED" : J->iters >= J->maxit ? "ITERATION LIMIT" : "TIME LIMIT")
                       : rc == 4 ? "PRIMAL INFEASIBLE" : rc == 5 ? "DUAL INFEASIBLE" : "FAILED";
    /* 24 October 2026 (ISSUES 281): the six DIMACS errors of the result, in the working precision, from the measures
     * above. Those are relative to the 2-norms of b and C; the DIMACS definitions (the standard solver's ps_measure) to
     * 1 + |b|_1 and 1 + |C|_1, the entrywise sums with the off-diagonal entries twice: the same point read 250 times apart
     * on a theta function of 250 vertices. Z = C - A*(y) is formed exactly (err3 = 0), its distance from positive
     * semidefinite is err4 (the measure's shift, resolved to a factor 2), and err2 is X's where it was measured (the dual
     * form's slack, a returned double solution), else 0 (X is positive definite in the method's coordinates). The
     * status line and the library's errors (the interfaces' DIMACS errors) are these; until now the library carried
     * the 2-norm measure under the DIMACS name, Z's semidefiniteness in the dual residual's place. The ladder's own
     * decisions keep its measure. */
    double dimacs[7] = { 0 }, dim_max = J->best_err;
    if (J->have_iter && J->b && J->cptr && rc != 4 && rc != 5) {
        double nb1 = 0, nb2 = 0, nc1 = 0, nc2 = 0;
        for (int c = 0; c < J->m; c++) { const double bc = mpx_getd(J->b + (size_t)c * Wm); nb1 += fabs(bc); nb2 += bc * bc; }
        for (int k = 0; k < J->nblk; k++)
            for (size_t e = J->cptr[k]; e < J->cptr[k + 1]; e++) {
                const double v = mpx_getd(J->Cv + e * Wm), w = J->bs[k] > 0 && J->ci[e] != J->cj[e] ? 2.0 : 1.0;
                nc1 += w * fabs(v); nc2 += w * v * v;
            }
        const double rb = (1 + sqrt(nb2)) / (1 + nb1), rcz = (1 + sqrt(nc2)) / (1 + nc1);
        dimacs[1] = J->best_pinf * rb; dimacs[2] = hp_xpsd * rb; dimacs[3] = 0; dimacs[4] = J->best_dinf * rcz;
        dimacs[5] = J->best_gap; dimacs[6] = J->best_compl;
        dim_max = fmax(fmax(dimacs[1], dimacs[2]), fmax(dimacs[4], fmax(fabs(dimacs[5]), fabs(dimacs[6]))));
    }
    const int nd = (int)(final_bits * 0.30103);
    if (J->verbose) {
        char *s1 = (char *)malloc((size_t)nd + 64), *s2 = (char *)malloc((size_t)nd + 64);
        uint64_t t[MPX_MAXL + 2];
        printf("status: %s (max rel error %.1e)   iterations: %d\n", status, dim_max, J->iters + it_warm);
        /* 23 October 2026 (ISSUES 256): the verdict comes from the iterates (they run along an improving direction), not
         * from a certificate checked on the problem as read, which the double solver gives; said so */
        if ((rc == 4 || rc == 5) && ray_cert == 1)
            printf("  certificate: the solution of the ray problem (solved in %s precision, %.2fs) is a ray of the data as read\n"
                   "  (%s; violation %.1e with %s)\n", ray_prec ? "this" : "double", t_ray, rc == 4 ? "b'y > 0, A'y <= 0" : "<C,X> < 0, A(X) = 0, X >= 0", ray_viol, rc == 4 ? "b'y = 1" : "<C,X> = -1");
        else if (rc == 4 || rc == 5) {
            printf("  no certificate: the verdict comes from the iterates, which run along an improving direction");
            if (J->ray_res > 0) printf(" (relative residual %.1e)", J->ray_res);
            if (ray_cert == 0 && isinf(ray_viol)) printf(";\n  its ray problem did not end at a solution (%.2fs)\n", t_ray);
            else if (ray_cert == 0) printf(";\n  the solution of the ray problem (%.2fs) is not a ray of the data as read (violation %.1e with %s): the problem may\n"
                                           "  be weakly infeasible (no ray exists) or the verdict may be wrong\n", t_ray, ray_viol, rc == 4 ? "b'y = 1" : "<C,X> = -1");
            else printf("\n");
        }
        /* 20 October 2026 (the owner): the values to ten significant digits in the log (all of the working precision's
         * digits with -v, verbosity 2 and more) */
        const int ndp = J->verbose > 2 || nd < 10 ? nd : 10;   /* (J->verbose is the verbosity + 1) */
        mpx_neg(t, J->pobj); mpx_to_str(s1, t, ndp);
        /* (23 October 2026: after an infeasibility verdict there is no optimal value; the values are those of the last point) */
        if (rc == 4 || rc == 5) printf("the last point's values (no optimal value: the verdict above)\n");
        else printf("optimal value (SDPA/SDPLIB convention, max <F0,Y>): %s\n", s1);
        mpx_to_str(s1, J->pobj, ndp); mpx_to_str(s2, J->dobj, ndp);
        printf("  primal obj <C,X> = %s\n  dual obj b'y     = %s\n", s1, s2);
        printf("  errors (in the working precision, on the data as read (before presolve)): pinf %.1e  dinf %.1e  gap %.1e  <X,Z> %.1e\n", J->best_pinf, J->best_dinf, J->best_gap, J->best_compl);
        if ((rc == 1 || rc == 0) && J->have_iter) {
            /* 5.16: the double solver's diagnosis of a side without an attained optimum (main.c, the same rule), which
             * no precision changes: a user's SOS problem crept at every precision with tr X = 6e7 (ISSUES 88). Also on
             * OPTIMAL (15 October 2026, the owner's benchmark, ISSUES 101b): hinf12 at -prec dd meets 1e-24 in the working
             * precision with |y|_1 = 5e29, a dual growing like 1/mu; evaluated in double precision that point is not usable */
            const size_t Wm_ = (size_t)mpx_W;
            double trX = 0, nb1 = 0, nC1 = 0, y1 = 0;
            for (int k = 0; k < nb; k++) { const int n = abs(J->bs[k]); for (int i = 0; i < n; i++) trX += mpx_getd(J->X[k] + (J->bs[k] < 0 ? (size_t)i : (size_t)i * n + i) * Wm_); for (size_t e = J->cptr[k]; e < J->cptr[k + 1]; e++) nC1 += fabs(mpx_getd(J->Cv + e * Wm_)); }
            for (int c = 0; c < m; c++) { nb1 += fabs(mpx_getd(J->b + (size_t)c * Wm_)); y1 += fabs(mpx_getd(J->y + (size_t)c * Wm_)); }
            const double po = mpx_getd(J->pobj), dob = mpx_getd(J->dobj), obj = fabs(po) > fabs(dob) ? po : dob;
            if (trX > 1e5 * (1 + nb1) && trX > 1e4 * (1 + fabs(obj)))
                printf("  note: tr X = %.1e is large against the data: either the optimum is not attained ((D) without a strictly feasible\n  point) or the solution's scale is that large;\n  <C,X> is still an upper bound (-bound p)%s\n", trX,
                       rc == 0 ? "; the status holds in the working precision, in double precision the primal is not usable" : "");
            else if (y1 > 1e5 * (1 + nC1) && y1 > 1e4 * (1 + fabs(obj)))
                printf("  note: |y|_1 = %.1e is large against the data: either the optimum is not attained ((P) without a strictly feasible\n  point) or the solution's scale is that large; b'y is still a lower bound (-bound d)%s\n", y1,
                       rc == 0 ? "; the status holds in the working precision, in double precision the dual is not usable" : "");
        }
        if (J->method == 2) {
            int rmax = lr_rmax; if (J->lr_rk) for (int k = 0; k < nb; k++) if (J->bs[k] > 0 && J->lr_rk[k] > rmax) rmax = J->lr_rk[k];
            printf("  X = R R' (largest rank %d); Z + d I is positive definite for d = %.1e (Cholesky in the working precision)\n", rmax, J->lr_delta);
        } else if (face_dual == 2) printf("  Z lies in the face found by the presolve (dual form); X is positive semidefinite on that face only: the primal of the\n  problem as read is in general not attained; <C,X> is an upper bound of the dual value by weak duality on the face\n");
        else if (face_dual) printf("  X lies in the face found by the presolve; Z = C - A*(y) is positive definite on that face (dinf is measured there:\n  the dual of the problem as read is in general not attained; b'y is a lower bound by weak duality on the face)\n");
        else printf("  X and Z are positive definite (Cholesky in the working precision)\n");
        printf("time: total %.3fs;", wtime() - J->tstart);
        for (int l = 0; l <= lev; l++) { if (lk[l] == 3) printf(" mp%d", lb[l]); else printf(" %s", l == 0 ? "d" : lk[l] == 1 ? "dd" : "qd"); printf(" %d it %.3fs%s", itlev[l], tlev[l], l < lev ? "," : "\n"); }
        for (int sd = 1; sd <= 2; sd++) {
            if (!tried_cert[sd - 1]) continue;
            const char *SD = sd == 1 ? "P" : "D", *ul = sd == 1 ? "upper" : "lower";
            const HPCertOut *ce = &cert[sd - 1];
            if (have_cert[sd - 1]) {
                mpx_to_str(s1, ce->value, nd);
                printf("BOUND (%s): %s bound %s = %s (certificate: %s)\n", SD, ul, sd == 1 ? "<C,X>" : "b'y", s1, cert_src[sd - 1]);
                hpc_bound_str(s1, ce->bound, nd, sd == 1 ? +1 : -1);
                mpx_neg(t, ce->bound); hpc_bound_str(s2, t, nd, sd == 1 ? -1 : +1);
                if (prior_[sd - 1])
                    printf("CERTIFIED (%s): rigorous %s bound %s on the optimal value of %s\n           (SDPA convention: %s bound %s; by a-priori bounds, in double precision; %.2fs)\n",
                           SD, ul, s1, sd == 1 ? "(D)" : "(P)", sd == 1 ? "a lower" : "an upper", s2, t_cert[sd - 1]);
                else
                printf("CERTIFIED (%s): rigorous %s bound %s on the optimal value of (P)\n           (SDPA convention: %s bound %s; %d-bit arithmetic, distance to the feasible point %.1e; %.2fs)\n",
                       SD, ul, s1, sd == 1 ? "a lower" : "an upper", s2, 64 * (J->Lm - 1), ce->delta, t_cert[sd - 1]);
            } else printf("NOT CERTIFIED (%s): %s: no guaranteed %s bound (%.2fs)\n", SD, ce->why[0] ? ce->why : "no certificate", ul, t_cert[sd - 1]);
        }
        /* 20 October 2026 (the owner): a Summary as the standard solver's: the problem as read, the preprocessing, the
         * solve, the time and the errors */
        if (!getenv("BRISK_NOSUMMARY")) {
            /* (a free variable passed as the difference of two nonnegative ones is counted as free, as the standard
             * Summary does: the pairs the presolve eliminated) */
            const int rd_free = hp_nfree_read, rd_nn = rd_lp - 2 * rd_free;
            printf("Summary\n  Problem as read:   %d equality constraints, %d semidefinite block%s (largest of order %d)", rd_m, rd_sdp, rd_sdp == 1 ? "" : "s", rd_maxn);
            if (rd_nn > 0) printf(", %d nonnegative variable%s", rd_nn, rd_nn > 1 ? "s" : "");
            if (rd_free > 0) printf(", %d free variable%s", rd_free, rd_free > 1 ? "s" : "");
            printf("\n");
            const char *lab = "  Preprocessing:     ";
            if (!g_hpsum_n) printf("%snone applied\n", lab);
            else { const char *p0 = g_hpsum; int first = 1;
                   while (*p0) { const char *e = strchr(p0, '\n'); const int len = e ? (int)(e - p0) : (int)strlen(p0);
                                 printf("%s%.*s\n", first ? lab : "                     ", len, p0); first = 0; p0 += len + (e ? 1 : 0); } }
            char pn[64];
            if (kind == 1) snprintf(pn, sizeof pn, "double-double (about 31 digits)");
            else if (kind == 2) snprintf(pn, sizeof pn, "quad-double (about 63 digits)");
            else snprintf(pn, sizeof pn, "%d bits (about %d digits)", final_bits, nd);
            printf("  Solve:             %s in %s, %d iterations (", J->method == 2 ? "low-rank augmented Lagrangian method" : J->method == 1 ? "augmented Lagrangian method with semismooth Newton steps" : "interior-point method",
                   pn, J->iters + it_warm);
            int first = 1;
            for (int l = 0; l <= lev; l++) {
                if (itlev[l] == 0) continue;
                if (lk[l] == 3) printf("%s%d bits %d", first ? "" : ", ", lb[l], itlev[l]);
                else printf("%s%s %d", first ? "" : ", ", l == 0 ? "double precision" : lk[l] == 1 ? "double-double" : "quad-double", itlev[l]);
                first = 0;
            }
            printf(")\n");
            if (res_double > 0 && isfinite(res_double)) printf("                     its result is the double-precision solve's (the high-precision method's own ended at %.1e)\n", res_double);
            else if (res_double > 0) printf("                     its result is the double-precision solve's (the high-precision method ended without a point)\n");
            printf("  Time:              %.2f s\n", wtime() - J->tstart);
            printf("  Errors:            pinf %.1e  dinf %.1e  gap %.1e  <X,Z> %.1e (on the problem as read, in the working precision)\n", J->best_pinf, J->best_dinf, J->best_gap, J->best_compl);
            if (J->have_iter && rc != 4 && rc != 5)
                printf("  DIMACS errors:     pinf %.1e  X>=0 %.1e  dinf %.1e  Z>=0 %.1e  gap %.1e  compl %.1e (in the working precision)\n",
                       dimacs[1], dimacs[2], dimacs[3], dimacs[4], dimacs[5], dimacs[6]);
        }
        free(s1); free(s2);
    }
    /* a certificate is written (and kept as text) with all the digits of the master precision:
     * rounded to the digits of -prec it would no longer be feasible to the accuracy of its check */
    /* a split problem: X and Z back in the blocks as read (block diagonal) */
    HPJob Jo_;
    if (split) {
        memset(&Jo_, 0, sizeof(Jo_));
        Jo_.Lm = J->Lm; Jo_.m = J->m; Jo_.nblk = sp_.nb0;
        Jo_.bs = (int *)malloc(sizeof(int) * (size_t)(sp_.nb0 + 1)); memcpy(Jo_.bs, sp_.bs0, sizeof(int) * (size_t)sp_.nb0);
        Jo_.X = hp_split_back(&sp_, J, J->X); Jo_.Z = hp_split_back(&sp_, J, J->Z);
        if (certX) { uint64_t **cb = hp_split_back(&sp_, J, certX); hpc_free_blocks(certX, J->nblk); certX = cb; }
        Jo_.y = J->y; Jo_.pobj = J->pobj; Jo_.dobj = J->dobj; J->y = NULL; J->pobj = NULL; J->dobj = NULL;
        Jo_.iters = J->iters; Jo_.maxit = J->maxit; Jo_.have_iter = J->have_iter; Jo_.verbose = J->verbose; Jo_.tstart = J->tstart;
        Jo_.best_pinf = J->best_pinf; Jo_.best_dinf = J->best_dinf; Jo_.best_gap = J->best_gap; Jo_.best_compl = J->best_compl; Jo_.best_err = J->best_err;
        J_.iters = J->iters; J_.maxit = J->maxit;
        free(J->dep); J->dep = NULL;
        lr_free(J); hp_free(J);
        for (int k = 0; warm2_.X && k < warm2_.nblk; k++) { free(warm2_.X[k]); if (warm2_.Z) free(warm2_.Z[k]); }
        if (warm == &warm2_) { free(warm2_.X); free(warm2_.Z); }
        J = &Jo_; nb = sp_.nb0;
        hp_split_free(&sp_);
    }
    const int ndw = (have_cert[0] || have_cert[1]) ? (int)(64.0 * (J->Lm - 1) * 0.30103) - 1 : nd;
    hp_write(J, yfile, xfile, zfile, nd);
    if (par->bound_xfile || par->bound_yfile) {
        /* the certificates (-boundx, -boundy), with all the digits of their check */
        HPJob Jc = *J;
        Jc.X = certX; Jc.y = certy;
        if (par->bound_xfile && !certX) fprintf(stderr, "brisk: -boundx: no certificate X to write (it needs -bound p or pd and a certified candidate)\n");
        if (par->bound_yfile && !certy) fprintf(stderr, "brisk: -boundy: no certificate y to write (it needs -bound d or pd and a certified candidate)\n");
        hp_write(&Jc, certy ? par->bound_yfile : NULL, certX ? par->bound_xfile : NULL, NULL, ndw);
    }
    if (res) {
        /* the library interface: the same fields as a double solve; X, y and Z rounded to
         * doubles (the full digits are in the -x / -y / -z files) */
        static const int st[6] = { 0, 5, 6, 4, 1, 2 }, ec[6] = { 0, 10, 15, 14, 11, 12 };
        const int s = rc >= 0 && rc <= 5 ? rc : 3;
        res->status = rc == 2 && J->iters >= J->maxit ? 3 : st[s];
        res->exit_code = rc == 2 && J->iters >= J->maxit ? 13 : ec[s];
        snprintf(res->status_str, sizeof(res->status_str), "%s", status);
        res->iters = J->iters + it_warm; res->pobj = mpx_getd(J->pobj); res->dobj = mpx_getd(J->dobj);
        for (int e = 0; e < 7; e++) res->err[e] = 0;
        if (J->have_iter && rc != 4 && rc != 5) for (int e = 1; e <= 6; e++) res->err[e] = dimacs[e];
        else { res->err[1] = J->best_pinf; res->err[3] = J->best_dinf; res->err[5] = J->best_gap; res->err[6] = J->best_compl; }
        res->time = wtime() - J->tstart; res->m = m; res->nblk = nb;
        res->bs = (int *)malloc(sizeof(int) * (size_t)(nb + 1)); memcpy(res->bs, J->bs, sizeof(int) * (size_t)nb);
        res->y = (double *)malloc(sizeof(double) * (size_t)(m + 1));
        for (int p = 0; p < m; p++) res->y[p] = mpx_getd(J->y + (size_t)p * Wm);
        res->X = (double **)malloc(sizeof(double *) * (size_t)(nb + 1)); res->Z = (double **)malloc(sizeof(double *) * (size_t)(nb + 1));
        for (int k = 0; k < nb; k++) {
            const size_t len = J->bs[k] < 0 ? (size_t)(-J->bs[k]) : (size_t)J->bs[k] * J->bs[k];
            res->X[k] = (double *)malloc(sizeof(double) * (len + 1)); res->Z[k] = (double *)malloc(sizeof(double) * (len + 1));
            for (size_t i = 0; i < len; i++) { res->X[k][i] = mpx_getd(J->X[k] + i * Wm); res->Z[k][i] = mpx_getd(J->Z[k] + i * Wm); }
        }
        res->have_x = J->have_iter;
        res->bound_side = 0; res->bound_rigorous = NAN; res->n_resolves = 0; res->t_resolves = 0;
        for (int sd = 1; sd <= 2; sd++) {
            /* each side asked for; the certificate rounded to doubles (all its digits: -boundx /
             * -boundy, brisk_hp_text) */
            memset(&res->bnd[sd - 1], 0, sizeof(res->bnd[0]));
            res->bnd[sd - 1].value = NAN; res->bnd[sd - 1].rigorous = NAN; res->bnd[sd - 1].lammin = NAN;
            if (!(bside & sd)) continue;
            res->bnd[sd - 1].asked = 1;
            if (!tried_cert[sd - 1]) continue;
            const int hc = have_cert[sd - 1];
            res->bnd[sd - 1].have = hc; res->bnd[sd - 1].certified = hc; res->bnd[sd - 1].valid = hc ? 2 : 0;
            res->bnd[sd - 1].value = hc ? mpx_getd(cert[sd - 1].value) : NAN; res->bnd[sd - 1].resid = cert[sd - 1].resid;
            res->bnd[sd - 1].rigorous = hc ? hpc_bound_dbl(cert[sd - 1].bound, sd == 1 ? +1 : -1) : NAN;
            if (hc) { memcpy(res->bnd[sd - 1].src, cert_src[sd - 1], sizeof(res->bnd[0].src)); res->bnd[sd - 1].src[sizeof(res->bnd[0].src) - 1] = 0; }
            if (sd == 1 && certX) {
                res->bnd[0].X = (double **)calloc((size_t)nb + 1, sizeof(double *));
                for (int k = 0; k < nb; k++) {
                    const size_t len = J->bs[k] < 0 ? (size_t)(-J->bs[k]) : (size_t)J->bs[k] * J->bs[k];
                    res->bnd[0].X[k] = (double *)malloc(sizeof(double) * (len + 1));
                    for (size_t i = 0; i < len; i++) res->bnd[0].X[k][i] = mpx_getd(certX[k] + i * Wm);
                }
            }
            if (sd == 2 && certy) { res->bnd[1].y = (double *)malloc(sizeof(double) * (size_t)(m + 1)); for (int p = 0; p < m; p++) res->bnd[1].y[p] = mpx_getd(certy + (size_t)p * Wm); }
        }
        {
            const int s0 = (bside & 1) ? 0 : 1;
            if (bside && tried_cert[s0]) {
                res->bound_side = s0 + 1; res->bound_certified = res->bnd[s0].certified; res->bound_valid = res->bnd[s0].valid;
                res->bound_value = res->bnd[s0].value; res->bound_resid = res->bnd[s0].resid; res->bound_lammin = NAN; res->bound_rigorous = res->bnd[s0].rigorous;
            }
        }
        snprintf(res->cause, sizeof(res->cause), "%s", rc == 0 ? "" : rc == 1 ? "the accuracy this precision can reach on this problem; a higher -prec goes further" : "");
    }
    /* the solution stays available as text (brisk_hp_text) */
    hp_last_free();
    g_hp_last.digits = nd; g_hp_last.cdigits = ndw; g_hp_last.Lm = J->Lm; g_hp_last.m = m; g_hp_last.nb = nb;
    g_hp_last.bs = (int *)malloc(sizeof(int) * (size_t)(nb + 1)); memcpy(g_hp_last.bs, J->bs, sizeof(int) * (size_t)nb);
    g_hp_last.X = J->X; g_hp_last.Z = J->Z; g_hp_last.y = J->y; g_hp_last.pobj = J->pobj; g_hp_last.dobj = J->dobj;
    for (int sd = 1; sd <= 2; sd++) if (have_cert[sd - 1]) {
        g_hp_last.bnd[sd - 1] = (uint64_t *)malloc(Wm * sizeof(uint64_t)); memcpy(g_hp_last.bnd[sd - 1], cert[sd - 1].bound, Wm * sizeof(uint64_t));
        if (!g_hp_last.bound) { g_hp_last.bside = sd; g_hp_last.bound = (uint64_t *)malloc(Wm * sizeof(uint64_t)); memcpy(g_hp_last.bound, cert[sd - 1].bound, Wm * sizeof(uint64_t)); }
    }
    g_hp_last.certX = certX; g_hp_last.certy = certy; certX = NULL; certy = NULL;
    J->X = NULL; J->Z = NULL; J->y = NULL; J->pobj = NULL; J->dobj = NULL;
    free(J->dep); J->dep = NULL;
    lr_free(J); hp_free(J);
    return rc == 0 ? 0 : rc == 1 ? 10 : rc == 2 ? (J_.iters >= J_.maxit ? 13 : 15) : rc == 4 ? 11 : rc == 5 ? 12 : 14;
}

/* 5.2: -certify-x / -certify-y with -prec: the rigorous check of a given certificate (the -x /
 * -y formats, any number of digits) in the master precision of that -prec; no solve.
 * Exit code 0 when certified, 20 when not, 2 on an input error. */
int brisk_hp_certify_given(const char *fname, const BriskData *data, BriskResult *res, const char *cx, const char *cy, int kind, int digits, int verbose) {
    HPJob J_, *J = &J_;
    memset(J, 0, sizeof(*J));
    const double t0 = wtime();
    const int target_bits = kind == 1 ? 106 : kind == 2 ? 212 : (int)ceil(digits * 3.3219280949) + 8;
    mpx_set_prec(target_bits + (target_bits + 64 > 192 ? target_bits + 64 : 192));
    J->Lm = mpx_L;
    if (data ? hp_from_data(data, J) : hp_read(fname, J)) { hp_free(J); return 2; }
    const size_t Wm = (size_t)J->Lm + 2;
    const int nb = J->nblk, m = J->m, side = cx ? 1 : 2;
    FILE *f = fopen(cx ? cx : cy, "rb");
    if (!f) { fprintf(stderr, "brisk: cannot read %s\n", cx ? cx : cy); hp_free(J); return 2; }
    fseek(f, 0, SEEK_END); const long sz = ftell(f); fseek(f, 0, SEEK_SET);
    char *buf = (char *)malloc((size_t)sz + 2);
    if (!buf || fread(buf, 1, (size_t)sz, f) != (size_t)sz) { fclose(f); free(buf); hp_free(J); hp_fail("cannot read the certificate"); return 2; }
    fclose(f); buf[sz] = '\n'; buf[sz + 1] = 0;
    uint64_t **X = NULL, *y = NULL;
    const char *p = buf;
    if (side == 1) {
        X = hpc_alloc_blocks(J);
        for (;;) {
            char *end; long q[3]; int okk = 1;
            for (int a = 0; a < 3; a++) { q[a] = strtol(p, &end, 10); if (end == p) { okk = 0; break; } p = end; }
            if (!okk) break;
            uint64_t v[MPX_MAXL + 2]; const char *e2;
            if (mpx_from_str(v, p, &e2)) break;
            p = e2;
            if (q[0] < 1 || q[0] > nb) continue;
            const int k = (int)q[0] - 1, n = abs(J->bs[k]);
            long i = q[1], j = q[2];
            if (i < 1 || j < 1 || i > n || j > n) continue;
            if (J->bs[k] < 0) { mpx_copy(X[k] + (size_t)(i - 1) * Wm, v); continue; }
            mpx_copy(X[k] + ((size_t)(i - 1) * n + (j - 1)) * Wm, v); mpx_copy(X[k] + ((size_t)(j - 1) * n + (i - 1)) * Wm, v);
        }
    } else {
        y = (uint64_t *)calloc(((size_t)m + 1) * Wm, sizeof(uint64_t));
        for (int i = 0; i < m; i++) { const char *e2; if (mpx_from_str(y + (size_t)i * Wm, p, &e2)) break; p = e2; }
    }
    free(buf);
    HPPairs Q;
    hp_find_pairs(J, &Q);
    double lgG = NAN, lgP = NAN;
    {
        PSOrig *Oc = hp_psorig(J);
        if (side == 1) lgG = brisk_gram_lammin_lower(Oc, data != NULL);
        else if (Q.np > 0) lgP = brisk_pairs_lammin_lower(Oc, data != NULL, Q.np, Q.K1, Q.I1);
        ps_orig_free(Oc);
    }
    HPCertOut out;
    const int ok = hp_cert_check(J, &Q, lgG, lgP, side, X, y, &out);
    const int nd = (int)((kind == 1 ? 106 : kind == 2 ? 212 : target_bits) * 0.30103);
    if (verbose >= 0) {
        char *s1 = (char *)malloc((size_t)nd + 64), *s2 = (char *)malloc((size_t)nd + 64);
        uint64_t t[MPX_MAXL + 2];
        if (ok) {
            hpc_bound_str(s1, out.bound, nd, side == 1 ? +1 : -1);
            mpx_neg(t, out.bound); hpc_bound_str(s2, t, nd, side == 1 ? -1 : +1);
            printf("CERTIFIED (%s): rigorous %s bound %s on the optimal value of (P)\n           (SDPA convention: %s bound %s; %d-bit arithmetic, distance to the feasible point %.1e; %.2fs)\n",
                   side == 1 ? "P" : "D", side == 1 ? "upper" : "lower", s1, side == 1 ? "a lower" : "an upper", s2, 64 * (J->Lm - 1), out.delta, wtime() - t0);
        } else printf("NOT CERTIFIED (%s): %s\n", side == 1 ? "P" : "D", out.why);
        free(s1); free(s2);
    }
    if (res) {
        res->status = -1; res->exit_code = ok ? 0 : 20;
        snprintf(res->status_str, sizeof(res->status_str), "%s", ok ? "CERTIFIED" : "NOT CERTIFIED");
        res->m = m; res->nblk = nb;
        res->bs = (int *)malloc(sizeof(int) * (size_t)(nb + 1)); memcpy(res->bs, J->bs, sizeof(int) * (size_t)nb);
        res->bound_side = side; res->bound_certified = ok; res->bound_rigorous = ok ? hpc_bound_dbl(out.bound, side == 1 ? +1 : -1) : NAN;
        res->bound_value = mpx_getd(out.value); res->bound_valid = -1;
        snprintf(res->cause, sizeof(res->cause), "%s", ok ? "" : out.why);
    }
    hp_last_free();
    if (ok) {
        g_hp_last.digits = nd; g_hp_last.Lm = J->Lm; g_hp_last.bside = side;
        g_hp_last.bound = (uint64_t *)malloc(Wm * sizeof(uint64_t)); memcpy(g_hp_last.bound, out.bound, Wm * sizeof(uint64_t));
    }
    hp_pairs_free(&Q);
    hpc_free_blocks(X, nb); free(y);
    hp_free(J);
    return ok ? 0 : 20;
}
