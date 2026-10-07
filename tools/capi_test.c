/* capi_test.c - the C interface of libbrisk (capi.c) against the file path.
 *   make capitest
 * A small SDP (an SDP block, an LP block) is solved from memory and from the SDPA file with
 * the same numbers: status, objectives, y, X, Z must be identical (same pipeline). Also:
 * output modes, error returns (bad data, bad option), repeated solves. Exit 0 when all pass. */
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

typedef struct BriskResult BriskResult;
BriskResult *brisk_result_new(void);
void brisk_result_delete(BriskResult *r);
int brisk_result_status(const BriskResult *r);
const char *brisk_result_status_str(const BriskResult *r);
int brisk_result_iterations(const BriskResult *r);
int brisk_result_m(const BriskResult *r);
int brisk_result_nblk(const BriskResult *r);
int brisk_result_blocksize(const BriskResult *r, int k);
int brisk_result_have_x(const BriskResult *r);
double brisk_result_pobj(const BriskResult *r);
double brisk_result_dobj(const BriskResult *r);
void brisk_result_dimacs(const BriskResult *r, double *err);
const double *brisk_result_y(const BriskResult *r);
const double *brisk_result_X(const BriskResult *r, int k);
const double *brisk_result_Z(const BriskResult *r, int k);
int brisk_solve_data(int m, int nblk, const int *bs, const double *c, int64_t nnz, const int *mat, const int *blk,
                     const int *i, const int *j, const double *v, int nopt, const char *const *opts, BriskResult *res);
int brisk_solve_file(const char *fname, int nopt, const char *const *opts, BriskResult *res);
void brisk_set_output(int mode, int (*cb)(const char *s, int is_err));
const char *brisk_version(void);
int brisk_hp_digits(void);
int brisk_hp_nblk(void);
int brisk_hp_blocksize(int block);
long long brisk_hp_count(int what, int block);
long long brisk_hp_text(int what, int block, char *buf, long long len);

static int fails = 0;
#define CHECK(c, ...) do { if (!(c)) { fails++; printf("FAIL line %d: ", __LINE__); printf(__VA_ARGS__); printf("\n"); } } while (0)

static size_t g_chars = 0;
static int count_cb(const char *s, int is_err) { g_chars += strlen(s); return 0; }
static char g_log[1 << 16]; static size_t g_nlog = 0;
static int log_cb(const char *s, int is_err) {
    const size_t l = strlen(s);
    if (g_nlog + l < sizeof g_log) { memcpy(g_log + g_nlog, s, l); g_nlog += l; g_log[g_nlog] = 0; }
    return 0;
}

enum { M = 6, NB = 2, NMAX = 400 };
static const int bs[NB] = { 4, -3 };
static int mat[NMAX], blk[NMAX], ii[NMAX], jj[NMAX];
static double vv[NMAX], cc[M];
static int nnz = 0;
static void add(int a, int b, int i, int j, double v) { mat[nnz] = a; blk[nnz] = b; ii[nnz] = i; jj[nnz] = j; vv[nnz] = v; nnz++; }

static double maxdiff(const double *a, const double *b, int n) {
    double d = 0;
    for (int k = 0; k < n; k++) d = fmax(d, fabs(a[k] - b[k]));
    return d;
}

int main(void) {
    printf("libbrisk %s\n", brisk_version());
    /* a strictly feasible pair: F0 = -I (x = 0 is interior), random symmetric F_i, c = <F_i, Y0>
     * for a positive definite Y0 (the dual is feasible): optimal and attained */
    srand(7);
    for (int i = 1; i <= 4; i++) add(0, 1, i, i, -1.0);
    for (int i = 1; i <= 3; i++) add(0, 2, i, i, -1.0);
    for (int t = 1; t <= M; t++) {
        for (int i = 1; i <= 4; i++)
            for (int j = i; j <= 4; j++)
                if (rand() % 3 == 0) add(t, 1, j, i, (rand() % 2001 - 1000) / 500.0);   /* lower triangle given: swapped */
        add(t, 2, 1 + t % 3, 1 + t % 3, (t % 2 ? 1.0 : -1.0));
    }
    const double Y0[4][4] = { { 2, .3, 0, .1 }, { .3, 1.5, .2, 0 }, { 0, .2, 1, .1 }, { .1, 0, .1, 1.2 } };
    for (int t = 1; t <= M; t++) {
        cc[t - 1] = 0;
        for (int q = 0; q < nnz; q++) if (mat[q] == t) {
            if (blk[q] == 1) { const int i = ii[q] - 1, j = jj[q] - 1; cc[t - 1] += (i == j ? 1.0 : 2.0) * vv[q] * Y0[i][j]; }
            else cc[t - 1] += vv[q] * 0.7;
        }
    }
    add(3, 1, 2, 2, 0.0);   /* a zero entry: skipped */
    /* the same numbers as an SDPA file */
    char fn[] = "/tmp/brisk_capi_XXXXXX";
    int fd = mkstemp(fn);
    FILE *f = fdopen(fd, "w");
    fprintf(f, "\"capi_test\n%d\n%d\n%d %d\n", M, NB, bs[0], bs[1]);
    for (int t = 0; t < M; t++) fprintf(f, "%.17g ", cc[t]);
    fprintf(f, "\n");
    for (int q = 0; q < nnz; q++) fprintf(f, "%d %d %d %d %.17g\n", mat[q], blk[q], ii[q], jj[q], vv[q]);
    fclose(f);

    const char *opts[] = { "-q" };
    BriskResult *rf = brisk_result_new(), *rm = brisk_result_new();
    int rc1 = brisk_solve_file(fn, 1, opts, rf);
    int rc2 = brisk_solve_data(M, NB, bs, cc, nnz, mat, blk, ii, jj, vv, 1, opts, rm);
    unlink(fn);
    CHECK(rc1 == 0 && rc2 == 0, "exit codes %d %d", rc1, rc2);
    CHECK(brisk_result_status(rm) == 0, "status %s", brisk_result_status_str(rm));
    CHECK(brisk_result_m(rm) == M && brisk_result_nblk(rm) == NB && brisk_result_blocksize(rm, 1) == -3, "sizes");
    CHECK(brisk_result_pobj(rf) == brisk_result_pobj(rm) && brisk_result_dobj(rf) == brisk_result_dobj(rm), "objectives %.17g %.17g", brisk_result_pobj(rf), brisk_result_pobj(rm));
    CHECK(maxdiff(brisk_result_y(rf), brisk_result_y(rm), M) == 0, "y differs");
    CHECK(brisk_result_have_x(rm), "no X");
    CHECK(maxdiff(brisk_result_X(rf, 0), brisk_result_X(rm, 0), 16) == 0 && maxdiff(brisk_result_X(rf, 1), brisk_result_X(rm, 1), 3) == 0, "X differs");
    CHECK(maxdiff(brisk_result_Z(rf, 0), brisk_result_Z(rm, 0), 16) == 0 && maxdiff(brisk_result_Z(rf, 1), brisk_result_Z(rm, 1), 3) == 0, "Z differs");
    double err[6];
    brisk_result_dimacs(rm, err);
    double emax = 0; for (int e = 0; e < 6; e++) emax = fmax(emax, fabs(err[e]));
    CHECK(emax < 1e-7, "DIMACS %g", emax);
    /* Z = C - sum y_i A_i on the data given (C = -F0) */
    const double *y = brisk_result_y(rm), *Z0 = brisk_result_Z(rm, 0);
    double Zc[16] = { 0 };
    for (int q = 0; q < nnz; q++) if (blk[q] == 1) {
        const int i = ii[q] - 1, j = jj[q] - 1;
        const double w = mat[q] == 0 ? -vv[q] : -y[mat[q] - 1] * vv[q];
        Zc[i + 4 * j] += w; if (i != j) Zc[j + 4 * i] += w;
    }
    CHECK(maxdiff(Zc, Z0, 16) < 1e-12, "Z != C - A'y (%g)", maxdiff(Zc, Z0, 16));
    printf("in memory: %s, %d iterations, pobj %.10f, DIMACS max %.1e\n", brisk_result_status_str(rm), brisk_result_iterations(rm), brisk_result_pobj(rm), emax);

    /* output modes: silent prints nothing through the callback path; the callback sees all */
    g_chars = 0;
    brisk_set_output(2, count_cb);
    brisk_solve_data(M, NB, bs, cc, nnz, mat, blk, ii, jj, vv, 0, NULL, rm);
    CHECK(g_chars > 200, "callback got %zu chars", g_chars);
    brisk_set_output(1, NULL);
    brisk_solve_data(M, NB, bs, cc, nnz, mat, blk, ii, jj, vv, 0, NULL, rm);
    CHECK(brisk_result_status(rm) == 0, "silent run status");
    {   /* 1.3.2: the advice at the end of the log is in the caller's syntax: a C program's option
         * strings by default, an interface's command when the options name it (-caller), and the
         * tag does not stay for the next call; a bound that was asked for is not advised again */
        const char *tag[] = { "-caller", "python:solve_sdpa" }, *bnd[] = { "-bound", "d" };
        g_nlog = 0; g_log[0] = 0; brisk_set_output(2, log_cb);
        brisk_solve_data(M, NB, bs, cc, nnz, mat, blk, ii, jj, vv, 0, NULL, rm);
        CHECK(strstr(g_log, "For higher accuracy, in the option strings of the call:") && strstr(g_log, "\"-acc\", \"high\"")
              && strstr(g_log, "\"-prec\", \"dd\"") && strstr(g_log, "For a guaranteed bound") && strstr(g_log, "\"-bound\", \"d\"")
              && strstr(g_log, "\"-certify\""), "advice in C syntax");
        g_nlog = 0; g_log[0] = 0;
        int rct = brisk_solve_data(M, NB, bs, cc, nnz, mat, blk, ii, jj, vv, 2, tag, rm);
        CHECK(rct == 0 && strstr(g_log, "in brisk.solve_sdpa(..., options={...})") && strstr(g_log, "\"acc\": \"high\"")
              && strstr(g_log, "\"bound\": \"d\"") && strstr(g_log, "\"certify\": True"), "advice in the syntax of the named caller (rc %d)", rct);
        g_nlog = 0; g_log[0] = 0;
        brisk_solve_data(M, NB, bs, cc, nnz, mat, blk, ii, jj, vv, 2, bnd, rm);
        CHECK(strstr(g_log, "in the option strings of the call") && !strstr(g_log, "For a guaranteed bound")
              && strstr(g_log, "For a rigorous check of the bound"), "advice after a bound, and the caller's tag gone");
        brisk_set_output(1, NULL);
    }
    {   /* high precision: quad-double, and all the digits as text */
        const char *hopts[] = { "-prec", "qd", "-q" };
        const int rch = brisk_solve_data(M, NB, bs, cc, nnz, mat, blk, ii, jj, vv, 3, hopts, rm);
        double eh[6]; brisk_result_dimacs(rm, eh);
        double emax = 0; for (int e = 0; e < 6; e++) if (fabs(eh[e]) > emax) emax = fabs(eh[e]);
        CHECK(rch == 0 && brisk_result_status(rm) == 0, "high precision: exit %d status %s", rch, brisk_result_status_str(rm));
        CHECK(emax < 1e-38, "high precision: error %.1e", emax);
        CHECK(brisk_hp_digits() >= 63 && brisk_hp_nblk() == NB && brisk_hp_blocksize(1) == -3, "high precision: digits %d", brisk_hp_digits());
        CHECK(brisk_hp_count(2, 0) == M && brisk_hp_count(3, 0) == 16 && brisk_hp_count(3, 1) == 3, "high precision: counts");
        const long long need = -brisk_hp_text(1, 0, NULL, 0);
        char *buf = (char *)malloc((size_t)(need > 0 ? need : 1));
        const long long len = brisk_hp_text(1, 0, buf, need);
        CHECK(need > 60 && len > 60 && fabs(atof(buf) - brisk_result_dobj(rm)) <= 1e-12 * (1 + fabs(brisk_result_dobj(rm))), "high precision: text of the dual objective");
        free(buf);
    }
    /* errors: an entry outside its block, an unknown option */
    int badi = ii[5]; ii[5] = 9;
    int rc = brisk_solve_data(M, NB, bs, cc, nnz, mat, blk, ii, jj, vv, 0, NULL, rm);
    ii[5] = badi;
    CHECK(rc == 2 && brisk_result_status(rm) == -1, "bad entry: rc %d status %d", rc, brisk_result_status(rm));
    const char *bad[] = { "-nosuchoption" };
    rc = brisk_solve_data(M, NB, bs, cc, nnz, mat, blk, ii, jj, vv, 1, bad, rm);
    CHECK(rc == 1, "bad option: rc %d", rc);
    const char *badv[] = { "-maxit", "abc" };
    rc = brisk_solve_data(M, NB, bs, cc, nnz, mat, blk, ii, jj, vv, 2, badv, rm);
    CHECK(rc != 0 && brisk_result_status(rm) == -1, "bad option value: rc %d", rc);
    brisk_set_output(0, NULL);
    /* repeated solves give the same answer */
    brisk_set_output(1, NULL);
    for (int r = 0; r < 20; r++) {
        brisk_solve_data(M, NB, bs, cc, nnz, mat, blk, ii, jj, vv, 1, opts, rm);
        if (maxdiff(brisk_result_y(rf), brisk_result_y(rm), M) != 0) { CHECK(0, "repeat %d differs", r); break; }
    }
    brisk_set_output(0, NULL);
    brisk_result_delete(rf);
    brisk_result_delete(rm);
    printf("%s (%d failures)\n", fails ? "FAILED" : "capi: all checks passed", fails);
    return fails != 0;
}
