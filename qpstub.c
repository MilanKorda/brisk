/* qpstub.c - the printing functions of main.c for the standalone QP programs (tests/qp_test, brisk_qp), which link the
 * QP objects without main.c (the SOCP developer, 1 November 2026: sparsechol.c, qp.c and qpipm.c print through
 * brisk_printf and the default log's row, and the standalone link had failed since). Plain printing: no default-log
 * filter (g_log_brief stays 0) */
#include <stdio.h>
#include <stdarg.h>
int g_log_brief = 0, g_log_user = 0;
int brisk_printf(const char *fmt, ...) { va_list ap; va_start(ap, fmt); int n = vprintf(fmt, ap); va_end(ap); return n; }
int brisk_fprintf(FILE *f, const char *fmt, ...) { va_list ap; va_start(ap, fmt); int n = vfprintf(f, fmt, ap); va_end(ap); return n; }
void brisk_log_head(int ipm) { (void)ipm; }
void brisk_log_row(int it, double po, double dob, double pinf, double dinf, double gap, double mu, double step, double t) {
    (void)it; (void)po; (void)dob; (void)pinf; (void)dinf; (void)gap; (void)mu; (void)step; (void)t;
}
/* (5 November 2026, the merge for 1.6.5) the default log's iteration tables of main.c (brisk_itab_*, brisk_stab_*): nothing
 * printed by the standalone programs */
int g_itab = 0, g_stab = 0;
void brisk_itab_row(int it, double pobj, double dobj, double pinf, double dinf, double gap, double mu, double step) {
    (void)it; (void)pobj; (void)dobj; (void)pinf; (void)dinf; (void)gap; (void)mu; (void)step;
}
void brisk_itab_end(void) {}
int brisk_stab_due(long it) { (void)it; return 0; }
void brisk_stab_row(long it, const char *what, double obj, double pinf, int ninf) { (void)it; (void)what; (void)obj; (void)pinf; (void)ninf; }
void brisk_stab_end(long it, const char *what, double obj, double pinf, int ninf) { (void)it; (void)what; (void)obj; (void)pinf; (void)ninf; }
