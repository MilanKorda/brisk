/* qp_fminmax.h (1 November 2026, the SOCP developer): fmax and fmin as inline functions of exactly the C library's results
 * (glibc's x86-64 fmax: maxsd, a > b ? a : b, and the other argument when one is a NaN; fmin likewise with minsd), so that
 * the compiler inlines them: without -ffinite-math-only it calls the library for each, 4.5 million calls in a solve of a
 * 2,572-variable MPC QP (the equilibration's and the error measures' loops; 3.6 % of the call's instructions). Included
 * after <math.h> by the QP solver's files. */
#ifndef QP_FMINMAX_H
#define QP_FMINMAX_H
static inline double qp_fmax_(double a, double b) { return (a > b || b != b) ? a : b; }
static inline double qp_fmin_(double a, double b) { return (a < b || b != b) ? a : b; }
#define fmax(a, b) qp_fmax_((a), (b))
#define fmin(a, b) qp_fmin_((a), (b))
#endif
