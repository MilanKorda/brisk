/* SPDX-License-Identifier: Apache-2.0
 * mpx: BRISK's variable-precision binary float (5.0). See hp.h for the layout.
 * Truncating arithmetic (no correct rounding): the error of an operation is below one unit of
 * the last limb, and one limb more than requested is carried as a guard.
 */
#include "hp.h"
#include <stdio.h>
#include <stdlib.h>

int mpx_L = 4, mpx_W = 6;
typedef unsigned __int128 u128;

void mpx_set_prec(int bits) {
    int L = (bits + 63) / 64 + 1;
    if (L < 2) L = 2;
    if (L > MPX_MAXL) L = MPX_MAXL;
    mpx_L = L; mpx_W = L + 2;
}
int mpx_bits(void) { return 64 * (mpx_L - 1); }

#define SGN(a) ((int64_t)(a)[0])
#define EXP(a) ((int64_t)(a)[1])
#define LIMB(a) ((a) + 2)
static inline void set_zero(uint64_t *r) { memset(r, 0, sizeof(uint64_t) * (size_t)mpx_W); }

static int cmp_mag(const uint64_t *a, const uint64_t *b) {      /* both nonzero */
    if (EXP(a) != EXP(b)) return EXP(a) < EXP(b) ? -1 : 1;
    const uint64_t *x = LIMB(a), *y = LIMB(b);
    for (int i = mpx_L - 1; i >= 0; i--) if (x[i] != y[i]) return x[i] < y[i] ? -1 : 1;
    return 0;
}
/* |a| + |b|, exp(a) >= exp(b), sign s */
static void add_mag(uint64_t *r, const uint64_t *a, const uint64_t *b, int64_t s) {
    const int L = mpx_L;
    const int64_t sh = EXP(a) - EXP(b);
    if (sh >= L) { mpx_copy(r, a); r[0] = (uint64_t)s; return; }
    const uint64_t *x = LIMB(a), *y = LIMB(b);
    uint64_t t[MPX_MAXL];
    unsigned char c = 0;
    for (int i = 0; i < L; i++) {
        const uint64_t yi = i + sh < L ? y[i + sh] : 0;
        u128 v = (u128)x[i] + yi + c;
        t[i] = (uint64_t)v; c = (unsigned char)(v >> 64);
    }
    int64_t e = EXP(a);
    uint64_t *z = LIMB(r);
    if (c) { for (int i = 0; i < L - 1; i++) z[i] = t[i + 1]; z[L - 1] = 1; e++; }
    else memcpy(z, t, sizeof(uint64_t) * (size_t)L);
    r[0] = (uint64_t)s; r[1] = (uint64_t)e;
}
/* |a| - |b|, |a| > |b|, sign s */
static void sub_mag(uint64_t *r, const uint64_t *a, const uint64_t *b, int64_t s) {
    const int L = mpx_L;
    const int64_t sh = EXP(a) - EXP(b);
    if (sh >= L) { mpx_copy(r, a); r[0] = (uint64_t)s; return; }
    const uint64_t *x = LIMB(a), *y = LIMB(b);
    uint64_t t[MPX_MAXL];
    unsigned char bw = 0;
    for (int i = 0; i < L; i++) {
        const uint64_t yi = i + sh < L ? y[i + sh] : 0;
        const uint64_t d = x[i] - yi - bw;
        bw = (x[i] < yi) || (x[i] == yi && bw);
        t[i] = d;
    }
    int k = L - 1;
    while (k >= 0 && t[k] == 0) k--;
    if (k < 0) { set_zero(r); return; }
    const int z0 = L - 1 - k;
    const int64_t e = EXP(a) - z0;
    uint64_t *z = LIMB(r);
    for (int i = L - 1; i >= 0; i--) z[i] = i - z0 >= 0 ? t[i - z0] : 0;
    r[0] = (uint64_t)s; r[1] = (uint64_t)e;
}
static void add_signed(uint64_t *r, const uint64_t *a, const uint64_t *b, int64_t sb) {
    const int64_t sa = SGN(a);
    if (sb == 0) { mpx_copy(r, a); return; }
    if (sa == 0) { mpx_copy(r, b); r[0] = (uint64_t)sb; return; }
    if (sa == sb) { if (EXP(a) >= EXP(b)) add_mag(r, a, b, sa); else add_mag(r, b, a, sa); return; }
    const int c = cmp_mag(a, b);
    if (c == 0) { set_zero(r); return; }
    if (c > 0) sub_mag(r, a, b, sa); else sub_mag(r, b, a, sb);
}
void mpx_add(uint64_t *r, const uint64_t *a, const uint64_t *b) { add_signed(r, a, b, SGN(b)); }
void mpx_sub(uint64_t *r, const uint64_t *a, const uint64_t *b) { add_signed(r, a, b, -SGN(b)); }

void mpx_mul(uint64_t *r, const uint64_t *a, const uint64_t *b) {
    const int L = mpx_L;
    if (SGN(a) == 0 || SGN(b) == 0) { set_zero(r); return; }
    const uint64_t *x = LIMB(a), *y = LIMB(b);
    uint64_t p[2 * MPX_MAXL + 1];
    memset(p, 0, sizeof(uint64_t) * (size_t)(2 * L));
    /* short product: the columns below L - 2 only reach the result through carries below its
     * last limb */
    for (int i = 0; i < L; i++) {
        const uint64_t xi = x[i];
        if (!xi) continue;
        int j = L - 2 - i; if (j < 0) j = 0;
        uint64_t c = 0;
        for (; j < L; j++) {
            u128 v = (u128)xi * y[j] + p[i + j] + c;
            p[i + j] = (uint64_t)v; c = (uint64_t)(v >> 64);
        }
        p[i + L] = c;
    }
    const int64_t s = SGN(a) * SGN(b);
    int64_t e = EXP(a) + EXP(b);
    uint64_t *z = LIMB(r);
    if (p[2 * L - 1]) memcpy(z, p + L, sizeof(uint64_t) * (size_t)L);
    else { memcpy(z, p + L - 1, sizeof(uint64_t) * (size_t)L); e--; }
    r[0] = (uint64_t)s; r[1] = (uint64_t)e;
}
void mpx_submul(uint64_t *r, const uint64_t *a, const uint64_t *b) { uint64_t t[MPX_MAXL + 2]; mpx_mul(t, a, b); mpx_sub(r, r, t); }
void mpx_addmul(uint64_t *r, const uint64_t *a, const uint64_t *b) { uint64_t t[MPX_MAXL + 2]; mpx_mul(t, a, b); mpx_add(r, r, t); }

/* r = sum_i x_i y_i over n elements (stride mpx_W). The products are accumulated unnormalised
 * in a fixed-point accumulator of L + 3 limbs whose unit is 2^(64 (Emax - L - 1)), Emax the
 * largest exponent of a product: the exponent counts whole limbs, so the alignment of a term
 * is a word offset. Each term is the short product down to one column below the unit, added
 * (or subtracted) row by row straight into the accumulator; one normalisation at the end.
 * The error is of the order of that of n multiply-adds of the plain operations
 * (tools/mpx_check). A column-wise product and an intrinsics version of the row loop were
 * slower. */
void mpx_dot(size_t n, const uint64_t *x, const uint64_t *y, uint64_t *r) {
    const int L = mpx_L, W = mpx_W, A = L + 3;
    int64_t emax = INT64_MIN;
    for (size_t i = 0; i < n; i++) {
        const uint64_t *a = x + i * (size_t)W, *b = y + i * (size_t)W;
        if (a[0] == 0 || b[0] == 0) continue;
        const int64_t e = (int64_t)a[1] + (int64_t)b[1];
        if (e > emax) emax = e;
    }
    if (emax == INT64_MIN) { set_zero(r); return; }
    uint64_t acc[MPX_MAXL + 4];
    memset(acc, 0, sizeof(uint64_t) * (size_t)A);
    for (size_t i = 0; i < n; i++) {
        const uint64_t *a = x + i * (size_t)W, *b = y + i * (size_t)W;
        if (a[0] == 0 || b[0] == 0) continue;
        const int64_t d = emax - ((int64_t)a[1] + (int64_t)b[1]);
        if (d > L) continue;                                   /* below the guard limb */
        const int t0 = L - 1 + (int)d;                         /* product limb t0 lands on acc[0] */
        const uint64_t *xa = LIMB(a), *yb = LIMB(b);
        if ((SGN(a) < 0) == (SGN(b) < 0)) {
            for (int p = 0; p < L; p++) {
                const uint64_t xp = xa[p];
                if (!xp) continue;
                int q = t0 - 1 - p; if (q < 0) q = 0;
                if (q >= L) continue;
                int k = p + q - t0;                            /* -1 (the column below the unit) or >= 0 */
                uint64_t c = 0;
                if (k < 0) { c = (uint64_t)(((u128)xp * yb[q]) >> 64); q++; k = 0; }
                for (; q < L; q++, k++) {
                    const u128 v = (u128)xp * yb[q] + acc[k] + c;
                    acc[k] = (uint64_t)v; c = (uint64_t)(v >> 64);
                }
                for (; c && k < A; k++) { const uint64_t s = acc[k] + c; c = s < c; acc[k] = s; }
            }
        } else {
            for (int p = 0; p < L; p++) {
                const uint64_t xp = xa[p];
                if (!xp) continue;
                int q = t0 - 1 - p; if (q < 0) q = 0;
                if (q >= L) continue;
                int k = p + q - t0;
                uint64_t c = 0;
                if (k < 0) { c = (uint64_t)(((u128)xp * yb[q]) >> 64); q++; k = 0; }
                for (; q < L; q++, k++) {
                    const u128 v = (u128)xp * yb[q] + c;       /* (a high half of 2^64 - 1 comes with a low half of 0: c stays in range) */
                    const uint64_t lo = (uint64_t)v, old = acc[k];
                    acc[k] = old - lo; c = (uint64_t)(v >> 64) + (old < lo);
                }
                for (; c && k < A; k++) { const uint64_t old = acc[k]; acc[k] = old - c; c = old < c; }
            }
        }
    }
    /* sign and normalisation */
    int64_t s = 1;
    if (acc[A - 1] >> 63) {
        s = -1;
        uint64_t c = 1;
        for (int k = 0; k < A; k++) { const uint64_t v = ~acc[k] + c; c = c && v == 0; acc[k] = v; }
    }
    int t = A - 1;
    while (t >= 0 && acc[t] == 0) t--;
    if (t < 0) { set_zero(r); return; }
    uint64_t *z = LIMB(r);
    for (int k = 0; k < L; k++) { const int src = t - L + 1 + k; z[k] = src >= 0 ? acc[src] : 0; }
    r[0] = (uint64_t)s; r[1] = (uint64_t)(emax + t - L);
}

void mpx_setd(uint64_t *r, double a) {
    set_zero(r);
    if (a == 0.0 || !isfinite(a)) return;
    const int L = mpx_L;
    int e2;
    const double f = frexp(fabs(a), &e2);                     /* |a| = f 2^e2, f in [1/2, 1) */
    const uint64_t mant = (uint64_t)ldexp(f, 53);             /* integer, 53 bits */
    const int64_t k = (int64_t)e2 - 53;                       /* |a| = mant 2^k */
    int64_t q = k >= 0 ? k / 64 : -((-k + 63) / 64);
    const int sh = (int)(k - 64 * q);                         /* 0..63 */
    const u128 v = (u128)mant << sh;
    const uint64_t lo = (uint64_t)v, hi = (uint64_t)(v >> 64);
    uint64_t *z = LIMB(r);
    if (hi) { z[L - 1] = hi; z[L - 2] = lo; r[1] = (uint64_t)(q + 2); }
    else { z[L - 1] = lo; r[1] = (uint64_t)(q + 1); }
    r[0] = (uint64_t)(int64_t)(a < 0 ? -1 : 1);
}
static double mant_d(const uint64_t *a) {                     /* 0.[limbs] as a double */
    const uint64_t *x = LIMB(a);
    const int L = mpx_L;
    double d = ldexp((double)x[L - 1], -64) + ldexp((double)x[L - 2], -128);
    if (L > 2) d += ldexp((double)x[L - 3], -192);
    return d;
}
double mpx_getd(const uint64_t *a) {
    if (SGN(a) == 0) return 0.0;
    const int64_t e = EXP(a);
    if (e > 20) return SGN(a) > 0 ? HUGE_VAL : -HUGE_VAL;
    if (e < -20) return 0.0;
    const double d = ldexp(mant_d(a), (int)(64 * e));
    return SGN(a) > 0 ? d : -d;
}
void mpx_muld(uint64_t *r, const uint64_t *a, double b) { uint64_t t[MPX_MAXL + 2]; mpx_setd(t, b); mpx_mul(r, a, t); }

int mpx_cmp(const uint64_t *a, const uint64_t *b) {
    const int64_t sa = SGN(a), sb = SGN(b);
    if (sa != sb) return sa < sb ? -1 : 1;
    if (sa == 0) return 0;
    const int c = cmp_mag(a, b);
    return sa > 0 ? c : -c;
}
static int newton_iters(void) { int it = 1, bits = 48; while (bits < 64 * mpx_L) { bits *= 2; it++; } return it; }

void mpx_div(uint64_t *r, const uint64_t *a, const uint64_t *b) {
    if (SGN(a) == 0 || SGN(b) == 0) { set_zero(r); return; }
    uint64_t am[MPX_MAXL + 2], bm[MPX_MAXL + 2], x[MPX_MAXL + 2], t[MPX_MAXL + 2], one[MPX_MAXL + 2];
    mpx_copy(am, a); am[0] = 1; am[1] = 0;
    mpx_copy(bm, b); bm[0] = 1; bm[1] = 0;
    mpx_setd(one, 1.0);
    mpx_setd(x, 1.0 / mant_d(bm));
    const int it = newton_iters();
    for (int k = 0; k < it; k++) {                 /* x <- x + x (1 - b x) */
        mpx_mul(t, bm, x); mpx_sub(t, one, t); mpx_mul(t, t, x); mpx_add(x, x, t);
    }
    uint64_t q[MPX_MAXL + 2];
    mpx_mul(q, am, x);
    mpx_mul(t, bm, q); mpx_sub(t, am, t); mpx_mul(t, t, x); mpx_add(q, q, t);   /* q <- q + x (a - b q) */
    const int64_t s = SGN(a) * SGN(b);
    q[1] = (uint64_t)(EXP(q) + EXP(a) - EXP(b));
    q[0] = (uint64_t)s;
    mpx_copy(r, q);
}
void mpx_sqrt(uint64_t *r, const uint64_t *a) {
    if (SGN(a) <= 0) { set_zero(r); return; }
    const int64_t e = EXP(a), s = e & 1, k = (e - s) / 2;
    uint64_t am[MPX_MAXL + 2], y[MPX_MAXL + 2], t[MPX_MAXL + 2], u[MPX_MAXL + 2], one[MPX_MAXL + 2], half[MPX_MAXL + 2];
    mpx_copy(am, a); am[1] = (uint64_t)s;
    mpx_setd(one, 1.0); mpx_setd(half, 0.5);
    mpx_setd(y, 1.0 / sqrt(mpx_getd(am)));
    const int it = newton_iters();
    for (int i = 0; i < it; i++) {                 /* y <- y + y (1 - a y^2) / 2 */
        mpx_mul(t, y, y); mpx_mul(t, t, am); mpx_sub(t, one, t); mpx_mul(t, t, y); mpx_mul(t, t, half); mpx_add(y, y, t);
    }
    mpx_mul(u, am, y);
    mpx_mul(t, u, u); mpx_sub(t, am, t); mpx_mul(t, t, y); mpx_mul(t, t, half); mpx_add(u, u, t);   /* u <- u + y (a - u^2) / 2 */
    u[1] = (uint64_t)(EXP(u) + k);
    mpx_copy(r, u);
}

void mpx_set_dd(uint64_t *r, dd_t a) { uint64_t t[MPX_MAXL + 2]; mpx_setd(r, a.h); mpx_setd(t, a.l); mpx_add(r, r, t); }
void mpx_set_qd(uint64_t *r, qd_t a) {
    uint64_t t[MPX_MAXL + 2];
    mpx_setd(r, a.x[3]);
    for (int k = 2; k >= 0; k--) { mpx_setd(t, a.x[k]); mpx_add(r, r, t); }
}
dd_t mpx_get_dd(const uint64_t *a) {
    uint64_t t[MPX_MAXL + 2], u[MPX_MAXL + 2];
    const double d0 = mpx_getd(a);
    mpx_setd(u, d0); mpx_sub(t, a, u);
    const double d1 = mpx_getd(t);
    mpx_setd(u, d1); mpx_sub(t, t, u);
    const double d2 = mpx_getd(t);
    double e, s = hp_ts(d0, d1, &e);
    e += d2;
    dd_t r; r.h = hp_qts(s, e, &r.l); return r;
}
qd_t mpx_get_qd(const uint64_t *a) {
    uint64_t t[MPX_MAXL + 2], u[MPX_MAXL + 2];
    double d[5];
    mpx_copy(t, a);
    for (int k = 0; k < 5; k++) { d[k] = mpx_getd(t); mpx_setd(u, d[k]); mpx_sub(t, t, u); }
    /* getd truncates, so neighbouring components can overlap: two passes of two-sums first */
    for (int pass = 0; pass < 2; pass++)
        for (int k = 3; k >= 0; k--) { double e; d[k] = hp_ts(d[k], d[k + 1], &e); d[k + 1] = e; }
    return qd_renorm5(d[0], d[1], d[2], d[3], d[4]);
}
void mpx_from_other(uint64_t *r, const uint64_t *src, int Ls) {
    const int L = mpx_L;
    r[0] = src[0]; r[1] = src[1];
    for (int i = 0; i < L; i++) r[2 + L - 1 - i] = i < Ls ? src[2 + Ls - 1 - i] : 0;
}

static void pow10_mpx(uint64_t *r, long k) {                 /* 10^k, k >= 0 */
    uint64_t base[MPX_MAXL + 2];
    mpx_setd(r, 1.0); mpx_setd(base, 10.0);
    while (k > 0) { if (k & 1) mpx_mul(r, r, base); k >>= 1; if (k) mpx_mul(base, base, base); }
}
int mpx_from_str(uint64_t *r, const char *s, const char **end) {
    const char *p = s;
    while (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r') p++;
    int neg = 0;
    if (*p == '+' || *p == '-') { neg = *p == '-'; p++; }
    uint64_t m[MPX_MAXL + 2], t[MPX_MAXL + 2];
    set_zero(m);
    long dexp = 0; int nd = 0, seen_pt = 0;
    double chunk = 0; int nc = 0;
    static const double p10[16] = { 1, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9, 1e10, 1e11, 1e12, 1e13, 1e14, 1e15 };
    for (;; p++) {
        if (*p >= '0' && *p <= '9') {
            chunk = chunk * 10 + (*p - '0'); nc++; nd++;
            if (seen_pt) dexp--;
            if (nc == 15) { mpx_muld(m, m, p10[nc]); mpx_setd(t, chunk); mpx_add(m, m, t); chunk = 0; nc = 0; }
        } else if (*p == '.' && !seen_pt) seen_pt = 1;
        else break;
    }
    if (nd == 0) { if (end) *end = s; set_zero(r); return 1; }
    if (nc) { mpx_muld(m, m, p10[nc]); mpx_setd(t, chunk); mpx_add(m, m, t); }
    if (*p == 'e' || *p == 'E' || *p == 'd' || *p == 'D') {
        const char *q = p + 1; int eneg = 0; long ev = 0; int ok = 0;
        if (*q == '+' || *q == '-') { eneg = *q == '-'; q++; }
        while (*q >= '0' && *q <= '9') { ev = ev * 10 + (*q - '0'); q++; ok = 1; }
        if (ok) { dexp += eneg ? -ev : ev; p = q; }
    }
    if (SGN(m) != 0 && dexp != 0) {
        pow10_mpx(t, dexp < 0 ? -dexp : dexp);
        if (dexp > 0) mpx_mul(m, m, t); else mpx_div(m, m, t);
    }
    if (neg) m[0] = (uint64_t)(-(int64_t)m[0]);
    mpx_copy(r, m);
    if (end) *end = p;
    return 0;
}
void mpx_to_str(char *buf, const uint64_t *a, int nd) {
    const int L = mpx_L;
    if (SGN(a) == 0) { strcpy(buf, "0"); return; }
    const int ndmax = (int)(64.0 * (L - 1) * 0.30102999566) - 1;
    if (nd > ndmax) nd = ndmax;
    if (nd < 1) nd = 1;
    uint64_t t[MPX_MAXL + 2], pw[MPX_MAXL + 2];
    long dexp = (long)floor(log10(mant_d(a)) + 64.0 * (double)EXP(a) * 0.30102999566398120);
    char digs[64 * MPX_MAXL / 3 + 64];
    int ndig = 0;
    for (int attempt = 0; attempt < 4; attempt++) {
        const long sc = nd - 1 - dexp;                        /* |a| 10^sc in [10^(nd-1), 10^nd) */
        mpx_copy(t, a); t[0] = 1;
        pow10_mpx(pw, sc < 0 ? -sc : sc);
        if (sc > 0) mpx_mul(t, t, pw); else if (sc < 0) mpx_div(t, t, pw);
        const int64_t et = EXP(t);
        if (et < 1) { dexp--; continue; }
        if (et > L) { dexp++; continue; }
        uint64_t n[MPX_MAXL + 1];
        const uint64_t *x = LIMB(t);
        int K = (int)et;
        for (int i = 0; i < K; i++) n[i] = x[L - K + i];
        if (K < L && (x[L - K - 1] >> 63)) {                  /* round to nearest */
            int i = 0;
            while (i < K && ++n[i] == 0) i++;
            if (i == K) n[K++] = 1;
        }
        ndig = 0;
        char rev[64 * MPX_MAXL / 3 + 64];
        while (K > 0) {                                       /* divide by 10^19 repeatedly */
            const uint64_t D = 10000000000000000000ULL;
            u128 rem = 0;
            for (int i = K - 1; i >= 0; i--) { u128 cur = (rem << 64) | n[i]; n[i] = (uint64_t)(cur / D); rem = cur % D; }
            while (K > 0 && n[K - 1] == 0) K--;
            uint64_t rr = (uint64_t)rem;
            for (int d = 0; d < 19; d++) { rev[ndig++] = (char)('0' + rr % 10); rr /= 10; if (K == 0 && rr == 0) break; }
        }
        for (int i = 0; i < ndig; i++) digs[i] = rev[ndig - 1 - i];
        if (ndig == nd) break;
        if (ndig > nd) { dexp += ndig - nd; if (attempt == 3) break; }
        else { dexp -= nd - ndig; if (attempt == 3) break; }
    }
    char *o = buf;
    if (SGN(a) < 0) *o++ = '-';
    *o++ = digs[0];
    if (nd > 1) { *o++ = '.'; const int k = ndig < nd ? ndig : nd; memcpy(o, digs + 1, (size_t)(k - 1)); o += k - 1; }
    sprintf(o, "e%+03ld", dexp + (ndig > nd ? ndig - nd : 0));
}
/* s (Ls limbs) re-rounded into r (Lr limbs); the two may have different limb counts */
void mpx_conv(uint64_t *r, int Lr, const uint64_t *s, int Ls) {
    r[0] = s[0]; r[1] = s[1];
    for (int i = 0; i < Lr; i++) r[2 + Lr - 1 - i] = i < Ls ? s[2 + Ls - 1 - i] : 0;
}
