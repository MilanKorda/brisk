/* fastnum.h - decimal numbers without strtod where one rounding makes them exact (the SOCP developer; header-only, so that
 * every reader includes it without a new object in the build lists). An integer mantissa M of at most 19 significant digits,
 * M <= 2^53, with a decimal exponent e, |e| <= 22, is M * 10^e or M / 10^-e: one correctly rounded operation on two exact
 * operands (Clinger's fast path), so the same double as strtod's; any other form, and a number not followed by a blank or
 * the end, goes to strtod. (23 October 2026 in the CBF reader; 24 October 2026 moved here for the QPS reader too) */
#ifndef BRISK_FASTNUM_H
#define BRISK_FASTNUM_H
#include <stdlib.h>
static inline double fast_strtod(const char *p0, char **end) {
    static const double p10[23] = { 1e0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9, 1e10, 1e11, 1e12, 1e13, 1e14, 1e15, 1e16, 1e17, 1e18, 1e19, 1e20, 1e21, 1e22 };
    const char *p = p0;
    while (*p == ' ' || *p == '\t') p++;
    const char *st = p;
    int neg = 0; if (*p == '-' || *p == '+') { neg = *p == '-'; p++; }
    unsigned long long M = 0; int nd = 0, e = 0, any = 0;
    while (*p == '0') { p++; any = 1; }                         /* (leading zeros: not significant) */
    while (*p >= '0' && *p <= '9') { if (nd < 19) M = 10 * M + (unsigned)(*p - '0'); else e++; nd++; p++; any = 1; }
    if (*p == '.') {
        p++;
        if (nd == 0) while (*p == '0') { e--; p++; any = 1; }
        while (*p >= '0' && *p <= '9') { if (nd < 19) { M = 10 * M + (unsigned)(*p - '0'); e--; } nd++; p++; any = 1; }
    }
    if (!any || nd > 19) return strtod(st, end);
    if (*p == 'e' || *p == 'E') {
        const char *q = p + 1; int en = 0, es = 0, ed = 0;
        if (*q == '-' || *q == '+') { es = *q == '-'; q++; }
        while (*q >= '0' && *q <= '9') { if (en < 10000) en = 10 * en + (*q - '0'); q++; ed++; }
        if (!ed) return strtod(st, end);
        e += es ? -en : en; p = q;
    }
    if (*p != 0 && *p != ' ' && *p != '\t' && *p != '\n' && *p != '\r') return strtod(st, end);
    if (M > (1ULL << 53) || e < -22 || e > 22) return strtod(st, end);
    double v = (double)M;
    v = e >= 0 ? v * p10[e] : v / p10[-e];
    if (end) *end = (char *)p;
    return neg ? -v : v;
}
#endif
