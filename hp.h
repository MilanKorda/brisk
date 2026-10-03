/* SPDX-License-Identifier: Apache-2.0
 * High-precision arithmetic for BRISK (5.0): double-double (dd, ~32 digits), quad-double
 * (qd, ~64 digits) and a variable-precision binary float (mpx, any number of 64-bit limbs).
 * No external library. dd and qd follow Hida, Li and Bailey (the QD library); mpx is a
 * sign-magnitude float with a limb-granular exponent, like GMP's mpf.
 */
#ifndef BRISK_HP_H
#define BRISK_HP_H
#include <math.h>
#include <stdint.h>
#include <stddef.h>
#include <string.h>

/* ------------------------------------------------------------------ error-free transforms */
static inline double hp_qts(double a, double b, double *e) { double s = a + b; *e = b - (s - a); return s; }
static inline double hp_ts(double a, double b, double *e) { double s = a + b, bb = s - a; *e = (a - (s - bb)) + (b - bb); return s; }
static inline double hp_tp(double a, double b, double *e) { double p = a * b; *e = fma(a, b, -p); return p; }

/* ------------------------------------------------------------------ double-double */
typedef struct { double h, l; } dd_t;

static inline dd_t dd_set(double a) { dd_t r = { a, 0.0 }; return r; }
static inline double dd_get(dd_t a) { return a.h + a.l; }
static inline dd_t dd_neg(dd_t a) { dd_t r = { -a.h, -a.l }; return r; }
static inline dd_t dd_add(dd_t a, dd_t b) {
    double s, e, t, f;
    s = hp_ts(a.h, b.h, &e); t = hp_ts(a.l, b.l, &f);
    e += t; s = hp_qts(s, e, &e); e += f;
    dd_t r; r.h = hp_qts(s, e, &r.l); return r;
}
static inline dd_t dd_sub(dd_t a, dd_t b) { return dd_add(a, dd_neg(b)); }
static inline dd_t dd_mul(dd_t a, dd_t b) {
    double p, e; p = hp_tp(a.h, b.h, &e);
    e += a.h * b.l + a.l * b.h;
    dd_t r; r.h = hp_qts(p, e, &r.l); return r;
}
static inline dd_t dd_muld(dd_t a, double b) {
    double p, e; p = hp_tp(a.h, b, &e); e += a.l * b;
    dd_t r; r.h = hp_qts(p, e, &r.l); return r;
}
static inline dd_t dd_div(dd_t a, dd_t b) {
    double q1 = a.h / b.h;
    dd_t r = dd_sub(a, dd_muld(b, q1));
    double q2 = r.h / b.h;
    r = dd_sub(r, dd_muld(b, q2));
    double q3 = r.h / b.h;
    dd_t q; q.h = hp_qts(q1, q2, &q.l);
    return dd_add(q, dd_set(q3));
}
static inline dd_t dd_sqrt(dd_t a) {
    if (!(a.h > 0)) return dd_set(0.0);
    double x = 1.0 / sqrt(a.h), ax = a.h * x, e;
    double p = hp_tp(ax, ax, &e);
    dd_t t = dd_sub(a, (dd_t){ p, e });
    dd_t r; r.h = hp_qts(ax, t.h * x * 0.5, &r.l); return r;
}
static inline int dd_cmp(dd_t a, dd_t b) { return a.h < b.h ? -1 : a.h > b.h ? 1 : a.l < b.l ? -1 : a.l > b.l ? 1 : 0; }

/* ------------------------------------------------------------------ quad-double */
typedef struct { double x[4]; } qd_t;

static inline void hp_three_sum(double *a, double *b, double *c) {
    double t1, t2, t3; t1 = hp_ts(*a, *b, &t2); *a = hp_ts(*c, t1, &t3); *b = hp_ts(t2, t3, c);
}
static inline void hp_three_sum2(double *a, double *b, double *c) {
    double t1, t2, t3; t1 = hp_ts(*a, *b, &t2); *a = hp_ts(*c, t1, &t3); *b = t2 + t3;
}
static inline qd_t qd_renorm5(double c0, double c1, double c2, double c3, double c4) {
    double s0, s1, s2 = 0.0, s3 = 0.0;
    s0 = hp_qts(c3, c4, &c4); s0 = hp_qts(c2, s0, &c3); s0 = hp_qts(c1, s0, &c2); c0 = hp_qts(c0, s0, &c1);
    s0 = c0; s1 = c1;
    if (s1 != 0.0) {
        s1 = hp_qts(s1, c2, &s2);
        if (s2 != 0.0) { s2 = hp_qts(s2, c3, &s3); if (s3 != 0.0) s3 += c4; else s2 = hp_qts(s2, c4, &s3); }
        else { s1 = hp_qts(s1, c3, &s2); if (s2 != 0.0) s2 = hp_qts(s2, c4, &s3); else s1 = hp_qts(s1, c4, &s2); }
    } else {
        s0 = hp_qts(s0, c2, &s1);
        if (s1 != 0.0) { s1 = hp_qts(s1, c3, &s2); if (s2 != 0.0) s2 = hp_qts(s2, c4, &s3); else s1 = hp_qts(s1, c4, &s2); }
        else { s0 = hp_qts(s0, c3, &s1); if (s1 != 0.0) s1 = hp_qts(s1, c4, &s2); else s0 = hp_qts(s0, c4, &s1); }
    }
    qd_t r = { { s0, s1, s2, s3 } }; return r;
}
static inline qd_t qd_set(double a) { qd_t r = { { a, 0.0, 0.0, 0.0 } }; return r; }
static inline double qd_get(qd_t a) { return a.x[0] + a.x[1]; }
static inline qd_t qd_neg(qd_t a) { qd_t r = { { -a.x[0], -a.x[1], -a.x[2], -a.x[3] } }; return r; }
static inline qd_t qd_add(qd_t a, qd_t b) {
    double s0, s1, s2, s3, t0, t1, t2, t3;
    s0 = hp_ts(a.x[0], b.x[0], &t0); s1 = hp_ts(a.x[1], b.x[1], &t1);
    s2 = hp_ts(a.x[2], b.x[2], &t2); s3 = hp_ts(a.x[3], b.x[3], &t3);
    s1 = hp_ts(s1, t0, &t0);
    hp_three_sum(&s2, &t0, &t1);
    hp_three_sum2(&s3, &t0, &t2);
    t0 = t0 + t1 + t3;
    return qd_renorm5(s0, s1, s2, s3, t0);
}
static inline qd_t qd_sub(qd_t a, qd_t b) { return qd_add(a, qd_neg(b)); }
static inline qd_t qd_mul(qd_t a, qd_t b) {
    double p0, p1, p2, p3, p4, p5, q0, q1, q2, q3, q4, q5, t0, t1, s0, s1, s2;
    p0 = hp_tp(a.x[0], b.x[0], &q0);
    p1 = hp_tp(a.x[0], b.x[1], &q1); p2 = hp_tp(a.x[1], b.x[0], &q2);
    p3 = hp_tp(a.x[0], b.x[2], &q3); p4 = hp_tp(a.x[1], b.x[1], &q4); p5 = hp_tp(a.x[2], b.x[0], &q5);
    hp_three_sum(&p1, &p2, &q0);
    hp_three_sum(&p2, &q1, &q2);
    hp_three_sum(&p3, &p4, &p5);
    s0 = hp_ts(p2, p3, &t0); s1 = hp_ts(q1, p4, &t1); s2 = q2 + p5;
    s1 = hp_ts(s1, t0, &t0); s2 += (t0 + t1);
    s1 += a.x[0] * b.x[3] + a.x[1] * b.x[2] + a.x[2] * b.x[1] + a.x[3] * b.x[0] + q0 + q3 + q4 + q5;
    return qd_renorm5(p0, p1, s0, s1, s2);
}
static inline qd_t qd_muld(qd_t a, double b) {
    double p0, p1, p2, p3, q0, q1, q2, s0, s1, s2, s3, s4;
    p0 = hp_tp(a.x[0], b, &q0); p1 = hp_tp(a.x[1], b, &q1); p2 = hp_tp(a.x[2], b, &q2); p3 = a.x[3] * b;
    s0 = p0; s1 = hp_ts(q0, p1, &s2);
    hp_three_sum(&s2, &q1, &p2);
    hp_three_sum2(&q1, &q2, &p3);
    s3 = q1; s4 = q2 + p2;
    return qd_renorm5(s0, s1, s2, s3, s4);
}
static inline qd_t qd_div(qd_t a, qd_t b) {
    double q0, q1, q2, q3, q4; qd_t r;
    q0 = a.x[0] / b.x[0]; r = qd_sub(a, qd_muld(b, q0));
    q1 = r.x[0] / b.x[0]; r = qd_sub(r, qd_muld(b, q1));
    q2 = r.x[0] / b.x[0]; r = qd_sub(r, qd_muld(b, q2));
    q3 = r.x[0] / b.x[0]; r = qd_sub(r, qd_muld(b, q3));
    q4 = r.x[0] / b.x[0];
    return qd_renorm5(q0, q1, q2, q3, q4);
}
static inline qd_t qd_sqrt(qd_t a) {
    if (!(a.x[0] > 0)) return qd_set(0.0);
    qd_t x = qd_set(1.0 / sqrt(a.x[0])), h = qd_muld(a, 0.5);
    for (int k = 0; k < 3; k++)            /* x <- x + x (1/2 - h x^2) */
        x = qd_add(x, qd_mul(x, qd_sub(qd_set(0.5), qd_mul(h, qd_mul(x, x)))));
    qd_t s = qd_mul(a, x);
    return qd_add(s, qd_mul(qd_muld(x, 0.5), qd_sub(a, qd_mul(s, s))));    /* one correction on s */
}
static inline int qd_cmp(qd_t a, qd_t b) {
    for (int k = 0; k < 4; k++) { if (a.x[k] < b.x[k]) return -1; if (a.x[k] > b.x[k]) return 1; }
    return 0;
}

/* ------------------------------------------------------------------ mpx: variable precision
 * An element is mpx_W = mpx_L + 2 words: w[0] = sign (0, +1, -1 as int64), w[1] = exponent e
 * (int64, in limbs), w[2..] = mpx_L limbs, least significant first, top limb nonzero.
 * Value = sign * 0.[limbs] * 2^(64 e). The precision is set once per solve (mpx_set_prec). */
#define MPX_MAXL 130                      /* 8320 bits, about 2500 digits */
extern int mpx_L, mpx_W;
void   mpx_set_prec(int bits);            /* at least this many bits */
int    mpx_bits(void);
void   mpx_setd(uint64_t *r, double a);
double mpx_getd(const uint64_t *a);
void   mpx_add(uint64_t *r, const uint64_t *a, const uint64_t *b);
void   mpx_sub(uint64_t *r, const uint64_t *a, const uint64_t *b);
void   mpx_mul(uint64_t *r, const uint64_t *a, const uint64_t *b);
void   mpx_muld(uint64_t *r, const uint64_t *a, double b);
void   mpx_div(uint64_t *r, const uint64_t *a, const uint64_t *b);
void   mpx_sqrt(uint64_t *r, const uint64_t *a);
int    mpx_cmp(const uint64_t *a, const uint64_t *b);
static inline void mpx_copy(uint64_t *r, const uint64_t *a) { if (r != a) memcpy(r, a, sizeof(uint64_t) * (size_t)mpx_W); }
static inline void mpx_neg(uint64_t *r, const uint64_t *a) { mpx_copy(r, a); r[0] = (uint64_t)(-(int64_t)r[0]); }
static inline int  mpx_sgn(const uint64_t *a) { return (int)(int64_t)a[0]; }
/* r -= a*b and r += a*b */
void   mpx_submul(uint64_t *r, const uint64_t *a, const uint64_t *b);
void   mpx_addmul(uint64_t *r, const uint64_t *a, const uint64_t *b);
/* r = sum x_i y_i over n elements of mpx_W words (r may not overlap x, y) */
void   mpx_dot(size_t n, const uint64_t *x, const uint64_t *y, uint64_t *r);
/* conversions (dd/qd <-> mpx are exact up to the target's precision) */
void   mpx_set_dd(uint64_t *r, dd_t a);
void   mpx_set_qd(uint64_t *r, qd_t a);
dd_t   mpx_get_dd(const uint64_t *a);
qd_t   mpx_get_qd(const uint64_t *a);
/* decimal I/O: returns 0 on success; *end is set past the number */
int    mpx_from_str(uint64_t *r, const char *s, const char **end);
/* nd significant digits, scientific notation, into buf (at least nd + 32 bytes) */
void   mpx_to_str(char *buf, const uint64_t *a, int nd);
/* a re-rounded to another limb count: src has Ls limbs (Ws = Ls + 2 words), r has mpx_L */
void   mpx_from_other(uint64_t *r, const uint64_t *src, int Ls);
void   mpx_conv(uint64_t *r, int Lr, const uint64_t *s, int Ls);

#endif
