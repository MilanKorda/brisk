/* The thread count of a library call is the call's own (31 October 2026, ISSUES 727; RELEASE_CHECKS R17): a positive
 * -threads k forces k for that solve ("(forced by user)"), 0 or absent is the automatic count, and no call's choice
 * carries over to the next one, an error included. The log's "Number of threads: k" at the start and in the Summary
 * are the same line. usage: tools/thread_test [examples-dir]   (make threadtest) */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
typedef struct BriskResult BriskResult;
BriskResult *brisk_result_new(void);
void brisk_result_delete(BriskResult *r);
int brisk_result_status(const BriskResult *r);
int brisk_solve_file(const char *fname, int nopt, const char *const *opts, BriskResult *res);
void brisk_set_output(int mode, int (*cb)(const char *s, int is_err));
static char g_log[1 << 20]; static size_t g_len;
static int cb(const char *s, int is_err) { (void)is_err; const size_t l = strlen(s); if (g_len + l < sizeof g_log - 1) { memcpy(g_log + g_len, s, l); g_len += l; g_log[g_len] = 0; } return 0; }
/* the solve's two "Number of threads:" lines: the count, and the text after it; -1 when they are missing or differ */
static int solve(const char *f, int nopt, const char *const *o, char *label) {
    g_len = 0; g_log[0] = 0;
    BriskResult *r = brisk_result_new();
    brisk_solve_file(f, nopt, o, r);
    brisk_result_delete(r);
    char l[2][128] = {"", ""}; int n = 0;
    for (const char *p = g_log; (p = strstr(p, "Number of threads: ")) && n < 2; p++) {
        const char *e = strchr(p, '\n'); size_t len = e ? (size_t)(e - p) : strlen(p); if (len > 127) len = 127;
        memcpy(l[n], p, len); l[n][len] = 0; n++;
    }
    if (n < 2 || strcmp(l[0], l[1])) return -1;
    int k = -1; char rest[128] = "";
    if (sscanf(l[0], "Number of threads: %d%127[^\n]", &k, rest) < 1) return -1;
    strcpy(label, rest);
    return k;
}
static int nfail = 0;
static void check(const char *what, int ok) { printf("  %s  %s\n", ok ? "PASS" : "FAIL", what); if (!ok) nfail++; }
int main(int argc, char **argv) {
    const char *E = argc > 1 ? argv[1] : "examples";
    char t3[512], t1[512], lab[128], buf[16];
    snprintf(t3, sizeof t3, "%s/theta3.dat-s", E); snprintf(t1, sizeof t1, "%s/theta1.dat-s", E);
    brisk_set_output(2, cb);
    const char *v[] = {"-v"}, *one[] = {"-v", "-threads", "1"}, *zero[] = {"-v", "-threads", "0"}, *bad[] = {"-v", "-threads", "1", "-nosuchoption"};
    const int nt = solve(t3, 1, v, lab);
    check("automatic: the start and the Summary state the same count, without a reason", nt >= 1 && lab[0] == 0);
    int k = solve(t1, 1, v, lab);
    check("automatic on a small problem: 1 (problem too small for multithreading)", nt == 1 ? k == 1 : (k == 1 && !strcmp(lab, " (problem too small for multithreading)")));
    k = solve(t3, 3, one, lab);
    check("threads 1: 1 (forced by user)", k == 1 && !strcmp(lab, " (forced by user)"));
    k = solve(t3, 1, v, lab);
    check("automatic after threads 1: the automatic count again", k == nt && lab[0] == 0);
    snprintf(buf, sizeof buf, "%d", nt);
    const char *same[] = {"-v", "-threads", buf};
    k = solve(t3, 3, same, lab);
    check("threads k equal to the automatic count: still (forced by user)", k == nt && !strcmp(lab, " (forced by user)"));
    k = solve(t1, 3, same, lab);
    check("threads k on a small problem: k (forced by user), no reduction", k == nt && !strcmp(lab, " (forced by user)"));
    k = solve(t3, 3, zero, lab);
    check("threads 0: the automatic count", k == nt && lab[0] == 0);
    solve(t3, 4, bad, lab);
    k = solve(t3, 1, v, lab);
    check("automatic after a call with threads 1 that failed: the automatic count", k == nt && lab[0] == 0);
    printf(nfail ? "thread_test: %d check(s) failed\n" : "thread_test: all checks passed\n", nfail);
    return nfail != 0;
}
