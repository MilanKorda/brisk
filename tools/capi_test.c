/* capi_test.c - the C interface of libbrisk (capi.c) against the file path.
 *   make capitest
 * A small SDP (an SDP block, an LP block) is solved from memory and from the SDPA file with
 * the same numbers: status, objectives, y, X, Z must be identical (same pipeline). Also:
 * output modes, error returns (bad data, bad option), repeated solves. Exit 0 when all pass. */
#include <math.h>
#if defined(__x86_64__) || defined(__i386__)
#include <xmmintrin.h>
#endif
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
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
double brisk_result_time(const BriskResult *r);
void brisk_result_dimacs(const BriskResult *r, double *err);
const double *brisk_result_y(const BriskResult *r);
const double *brisk_result_X(const BriskResult *r, int k);
const double *brisk_result_Z(const BriskResult *r, int k);
int brisk_solve_data(int m, int nblk, const int *bs, const double *c, int64_t nnz, const int *mat, const int *blk,
                     const int *i, const int *j, const double *v, int nopt, const char *const *opts, BriskResult *res);
int brisk_solve_file(const char *fname, int nopt, const char *const *opts, BriskResult *res);
int brisk_solve_qp(int n, int m, const int *Hp, const int *Hi, const double *Hx, const double *c, double c0,
                   const int *Ap, const int *Ai, const double *Ax, const double *rl, const double *ru,
                   const double *xl, const double *xu, int nopt, const char *const *opts, BriskResult *res);
int brisk_result_sn(const BriskResult *r);
long brisk_qp_lower(int n, const int *Hp, const int *Hi, const double *Hx, int *Lp, int *Li, double *Lx);
const double *brisk_result_sx(const BriskResult *r);
int brisk_solve_lp(int n, int m, const int *Ap, const int *Ai, const double *Ax, const double *c, double c0,
                   const double *rl, const double *ru, const double *xl, const double *xu,
                   int nopt, const char *const *opts, BriskResult *res);
int brisk_result_have_x(const BriskResult *r);
int brisk_solve_socp(int n, int m, const int *Ap, const int *Ai, const double *Ax, const double *c, double c0,
                     const double *rl, const double *ru, const double *xl, const double *xu,
                     int nq, const int *qp, const int *qi, int nr, const int *rp, const int *ri,
                     int nopt, const char *const *opts, BriskResult *res);
const double *brisk_result_sz(const BriskResult *r);
void *brisk_qp_create(int n, int m, const int *Hp, const int *Hi, const double *Hx, const double *c, double c0,
                      const int *Ap, const int *Ai, const double *Ax, const double *rl, const double *ru,
                      const double *xl, const double *xu, int nopt, const char *const *opts);
int brisk_qp_set_var_bounds(void *h, int j, double lo, double hi);
int brisk_qp_resolve(void *h, BriskResult *res);
void brisk_qp_destroy(void *h);
void *brisk_socp_create(int m, int n, const int *Ap, const int *Ai, const double *Ax, const double *b, const double *c,
                        int nf, int nl, int nq, const int *q, int nr, const int *r, int nopt, const char *const *opts);
int brisk_socp_set_var_bounds(void *h, int j, double lo, double hi);
void brisk_socp_set_cutoff(void *h, double v);
int brisk_socp_resolve(void *h, BriskResult *res);
void brisk_socp_destroy(void *h);
void *brisk_socp_save_point(void *h);
void brisk_socp_restore_point(void *h, void *p);
void brisk_socp_point_free(void *p);
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
         * tag does not stay for the next call; a bound that was asked for is not advised again (and it
         * is checked rigorously anyway: -certify is not advised either) */
        /* (31 October 2026, ISSUES 725: the default log gives no advice after an OPTIMAL result; -v, the diagnosis, does) */
        const char *tag[] = { "-caller", "python:solve_sdpa", "-v" }, *bnd[] = { "-bound", "d", "-v" }, *diag[] = { "-v" };
        g_nlog = 0; g_log[0] = 0; brisk_set_output(2, log_cb);
        brisk_solve_data(M, NB, bs, cc, nnz, mat, blk, ii, jj, vv, 1, diag, rm);
        CHECK(strstr(g_log, "For higher accuracy, in the option strings of the call:") && strstr(g_log, "\"-acc\", \"high\"")
              && strstr(g_log, "\"-prec\", \"dd\"") && strstr(g_log, "For a guaranteed bound") && strstr(g_log, "\"-bound\", \"d\""),
              "advice in C syntax");
        g_nlog = 0; g_log[0] = 0;
        int rct = brisk_solve_data(M, NB, bs, cc, nnz, mat, blk, ii, jj, vv, 3, tag, rm);
        CHECK(rct == 0 && strstr(g_log, "in brisk.solve_sdpa(..., options={...})") && strstr(g_log, "\"acc\": \"high\"")
              && strstr(g_log, "\"bound\": \"d\""), "advice in the syntax of the named caller (rc %d)", rct);
        g_nlog = 0; g_log[0] = 0;
        brisk_solve_data(M, NB, bs, cc, nnz, mat, blk, ii, jj, vv, 3, bnd, rm);
        CHECK(strstr(g_log, "in the option strings of the call") && !strstr(g_log, "For a guaranteed bound")
              && !strstr(g_log, "For a rigorous check of the bound"), "advice after a bound, and the caller's tag gone");
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
    {   /* 24 October 2026: every method through the library, on a problem whose block of order 4 holds two independent
         * 2x2 parts (the high-precision solver splits it; max -tr Y with Y12 = Y34 = 1: <C,X> = 4 at the optimum). The
         * result's time lies within the call's wall clock (for a split problem the high-precision solver returned the
         * clock's raw reading, ISSUES 278), the status agrees with the exit code, a solved result carries X and the value */
        char fs[] = "/tmp/brisk_capi_split_XXXXXX";
        FILE *g = fdopen(mkstemp(fs), "w");
        fprintf(g, "\"two 2x2 parts in one block\n2\n1\n4\n1 1\n0 1 1 1 -1\n0 1 2 2 -1\n0 1 3 3 -1\n0 1 4 4 -1\n1 1 1 2 0.5\n2 1 3 4 0.5\n");
        fclose(g);
        static const char *const meth[][3] = { { "-q" }, { "-q", "-hsd" }, { "-q", "-nohsd" }, { "-q", "-mfipm", "1" }, { "-q", "-fom", "1" },
                                               { "-q", "-lralm", "1" }, { "-q", "-prec", "dd" }, { "-q", "-prec", "qd" } };
        static const int nopt[] = { 1, 2, 2, 3, 3, 3, 3, 3 };
        static const int exit_of[7] = { 0, 11, 12, 13, 14, 10, 15 };   /* the exit code of each status (capi.c) */
        BriskResult *rs = brisk_result_new();
        brisk_set_output(1, NULL);
        for (int q = 0; q < 8; q++) {
            struct timespec a, b; clock_gettime(CLOCK_MONOTONIC, &a);
            const int rcq = brisk_solve_file(fs, nopt[q], meth[q], rs);
            clock_gettime(CLOCK_MONOTONIC, &b);
            const double wall = (b.tv_sec - a.tv_sec) + 1e-9 * (b.tv_nsec - a.tv_nsec), t = brisk_result_time(rs);
            const int st = brisk_result_status(rs);
            const char *nm = nopt[q] > 1 ? meth[q][1] : "default";
            CHECK(t >= 0 && t <= wall + 0.05, "%s %s: time %.3f s for a call of %.3f s", nm, nopt[q] > 2 ? meth[q][2] : "", t, wall);
            CHECK(st >= 0 && st <= 6 && exit_of[st] == rcq, "%s %s: status %d (%s) with exit code %d", nm, nopt[q] > 2 ? meth[q][2] : "", st, brisk_result_status_str(rs), rcq);
            /* (the low-rank augmented Lagrangian never forms X = R R': no X by design, python/README.md) */
            const int want_x = strcmp(nm, "-lralm") != 0;
            if (st == 0 || st == 5)
                CHECK(brisk_result_have_x(rs) == want_x && fabs(brisk_result_pobj(rs) - 4.0) < 1e-3, "%s %s: value %.6g (4), X %s", nm, nopt[q] > 2 ? meth[q][2] : "", brisk_result_pobj(rs), brisk_result_have_x(rs) ? "returned" : "missing");
        }
        brisk_set_output(0, NULL);
        unlink(fs);
        brisk_result_delete(rs);
    }
    /* a convex QP (HS21): min 0.01 x1^2 + x2^2 - 100 s.t. 10 x1 - x2 >= 10, 2 <= x1 <= 50, -50 <= x2 <= 50; -99.96 at (2, 0) */
    {
        const int Hp[3] = { 0, 1, 2 }, Hi[2] = { 0, 1 }, Ap[3] = { 0, 1, 2 }, Ai[2] = { 0, 0 };
        const double Hx[2] = { 0.02, 2.0 }, qc[2] = { 0, 0 }, Ax[2] = { 10.0, -1.0 }, rl[1] = { 10.0 }, ru[1] = { 1e30 };
        const double xl[2] = { 2.0, -50.0 }, xu[2] = { 50.0, 50.0 };
        const char *qo[] = { "-q" };
        for (int as = 0; as < 2; as++) {
            const char *qa[] = { "-q", "-qpas" };
            rc = brisk_solve_qp(2, 1, Hp, Hi, Hx, qc, -100.0, Ap, Ai, Ax, rl, ru, xl, xu, as ? 2 : 1, as ? qa : qo, rm);
            const double *x = brisk_result_sx(rm), *y = brisk_result_y(rm), *z = brisk_result_sz(rm);
            CHECK(rc == 0 && brisk_result_status(rm) == 0 && fabs(brisk_result_pobj(rm) + 99.96) <= 1e-7 && brisk_result_sn(rm) == 2 && x && y && z,
                  "QP (%s): rc %d status %d obj %.10g", as ? "active set" : "interior point", rc, brisk_result_status(rm), brisk_result_pobj(rm));
            if (x && y && z) {   /* H x + c - A'y - z = 0 */
                const double r0 = 0.02 * x[0] - 10.0 * y[0] - z[0], r1 = 2.0 * x[1] + y[0] - z[1];
                CHECK(fabs(x[0] - 2) <= 1e-5 && fabs(x[1]) <= 1e-5 && fabs(r0) <= 1e-7 && fabs(r1) <= 1e-7, "QP multipliers: x %.3g %.3g, residual %g %g", x[0] - 2, x[1], r0, r1);
            }
        }
    }
    /* the QP solver object: HS21 warm, then x1 >= 3 (the optimum (3, 0), -99.91), then back */
    {
        const int Hp[3] = { 0, 1, 2 }, Hi[2] = { 0, 1 }, Ap[3] = { 0, 1, 2 }, Ai[2] = { 0, 0 };
        const double Hx[2] = { 0.02, 2.0 }, qc[2] = { 0, 0 }, Ax[2] = { 10.0, -1.0 }, rl[1] = { 10.0 }, ru[1] = { 1e30 };
        const double xl[2] = { 2.0, -50.0 }, xu[2] = { 50.0, 50.0 };
        const char *qo[] = { "-q" };
        void *h = brisk_qp_create(2, 1, Hp, Hi, Hx, qc, -100.0, Ap, Ai, Ax, rl, ru, xl, xu, 1, qo);
        CHECK(h != NULL, "brisk_qp_create");
        if (h) {
            rc = brisk_qp_resolve(h, rm);
            CHECK(rc == 0 && fabs(brisk_result_pobj(rm) + 99.96) <= 1e-7, "QP object: rc %d obj %.10g", rc, brisk_result_pobj(rm));
            brisk_qp_set_var_bounds(h, 0, 3.0, 50.0); rc = brisk_qp_resolve(h, rm);
            CHECK(rc == 0 && fabs(brisk_result_pobj(rm) + 99.91) <= 1e-7 && fabs(brisk_result_sx(rm)[0] - 3) <= 1e-9, "QP object, x1 >= 3: rc %d obj %.10g", rc, brisk_result_pobj(rm));
            brisk_qp_set_var_bounds(h, 0, 2.0, 50.0); rc = brisk_qp_resolve(h, rm);
            CHECK(rc == 0 && fabs(brisk_result_pobj(rm) + 99.96) <= 1e-7, "QP object, back: rc %d obj %.10g", rc, brisk_result_pobj(rm));
            brisk_qp_destroy(h);
        }
        const char *bad[] = { "-nosuch" };
        CHECK(brisk_qp_create(2, 1, Hp, Hi, Hx, qc, -100.0, Ap, Ai, Ax, rl, ru, xl, xu, 1, bad) == NULL, "QP object: an unknown option refused");
    }
    /* the SOCP model (warm starts for branch and bound): the distance from (x1, x2) to (0.3, 0.6), min t, (t, x1 - 0.3,
     * x2 - 0.6) in a second-order cone; columns x1, x2 free, then the cone (t, u1, u2), rows u1 - x1 = -0.3, u2 - x2 = -0.6.
     * No bound: 0; x1 >= 1: 0.7 (warm); x1 <= 0: 0.3; no bound again: 0; x1 >= 1 with the cutoff 0.5: stopped by it */
    {
        const int Ap[6] = { 0, 1, 2, 2, 3, 4 }, Ai[4] = { 0, 1, 0, 1 }, q[1] = { 3 };
        const double Ax[4] = { -1.0, -1.0, 1.0, 1.0 }, b[2] = { -0.3, -0.6 }, c[5] = { 0, 0, 1, 0, 0 };
        const char *so[] = { "-q" };
        void *h = brisk_socp_create(2, 5, Ap, Ai, Ax, b, c, 2, 0, 1, q, 0, NULL, 1, so);
        CHECK(h != NULL, "brisk_socp_create");
        if (h) {
            rc = brisk_socp_resolve(h, rm);
            CHECK(rc == 0 && fabs(brisk_result_pobj(rm)) <= 1e-6, "SOCP model: rc %d obj %.10g", rc, brisk_result_pobj(rm));
            brisk_socp_set_var_bounds(h, 0, 1.0, 1e30); rc = brisk_socp_resolve(h, rm);
            CHECK(rc == 0 && fabs(brisk_result_pobj(rm) - 0.7) <= 1e-6 && brisk_result_sx(rm) && fabs(brisk_result_sx(rm)[0] - 1) <= 1e-6,
                  "SOCP model, x1 >= 1: rc %d obj %.10g", rc, brisk_result_pobj(rm));
            void *pt = brisk_socp_save_point(h);
            CHECK(pt != NULL, "SOCP model: the point saved");
            brisk_socp_restore_point(h, pt); brisk_socp_point_free(pt);
            brisk_socp_set_var_bounds(h, 0, -1e30, 0.0); rc = brisk_socp_resolve(h, rm);
            CHECK(rc == 0 && fabs(brisk_result_pobj(rm) - 0.3) <= 1e-6, "SOCP model, x1 <= 0: rc %d obj %.10g", rc, brisk_result_pobj(rm));
            brisk_socp_set_var_bounds(h, 0, -1e30, 1e30); rc = brisk_socp_resolve(h, rm);
            CHECK(rc == 0 && fabs(brisk_result_pobj(rm)) <= 1e-6, "SOCP model, no bound: rc %d obj %.10g", rc, brisk_result_pobj(rm));
            brisk_socp_set_var_bounds(h, 0, 1.0, 1.0); brisk_socp_set_var_bounds(h, 1, 0.6, 2.0); rc = brisk_socp_resolve(h, rm);
            CHECK(rc == 0 && fabs(brisk_result_pobj(rm) - 0.7) <= 1e-6 && fabs(brisk_result_sx(rm)[0] - 1) <= 1e-7,
                  "SOCP model, x1 = 1 and 0.6 <= x2 <= 2: rc %d obj %.10g", rc, brisk_result_pobj(rm));
            brisk_socp_set_var_bounds(h, 1, -1e30, 1e30);
            brisk_socp_set_var_bounds(h, 0, 1.0, 1e30); brisk_socp_set_cutoff(h, 0.5); rc = brisk_socp_resolve(h, rm);
            CHECK(rc == 9, "SOCP model, x1 >= 1 and the cutoff 0.5: rc %d (%s) obj %.10g", rc, brisk_result_status_str(rm), brisk_result_pobj(rm));
#if defined(__x86_64__) || defined(__i386__)
            /* (24 October 2026) the SOCP solver sets flush-to-zero and denormals-are-zero for its solve and gives the
             * caller's thread its mode back */
            CHECK((_mm_getcsr() & 0x8040u) == 0, "SOCP model: the caller's floating-point mode restored (MXCSR %#x)", _mm_getcsr());
#endif
            brisk_socp_destroy(h);
        }
        const char *bad[] = { "-nosuch" };
        CHECK(brisk_socp_create(2, 5, Ap, Ai, Ax, b, c, 2, 0, 1, q, 0, NULL, 1, bad) == NULL, "SOCP model: an unknown option refused");
        CHECK(brisk_socp_create(2, 5, Ap, Ai, Ax, b, c, 2, 1, 1, q, 0, NULL, 0, NULL) == NULL, "SOCP model: cone sizes that do not add up refused");
    }
    {   /* (23 October 2026) brisk_qp_lower: H = [2 1; 1 3] given whole, as its upper triangle and as its lower, and
         * [2 1; 0.5 3], which is neither symmetric nor triangular */
        const int Fp[] = { 0, 2, 4 }, Fi[] = { 0, 1, 0, 1 }; const double Fx[] = { 2, 1, 1, 3 };
        const int Up[] = { 0, 1, 3 }, Ui[] = { 0, 0, 1 }; const double Ux[] = { 2, 1, 3 };
        const int Lp0[] = { 0, 2, 3 }, Li0[] = { 0, 1, 1 }; const double Lx0[] = { 2, 1, 3 };
        const double Nx[] = { 2, 0.5, 1, 3 };
        int Lp[3], Li[4]; double Lx[4];
        long k = brisk_qp_lower(2, Fp, Fi, Fx, Lp, Li, Lx);
        CHECK(k == 3 && Lp[1] == 2 && Li[0] == 0 && Li[1] == 1 && Li[2] == 1 && Lx[0] == 2 && Lx[1] == 1 && Lx[2] == 3, "QP lower triangle: the whole symmetric matrix (%ld)", k);
        k = brisk_qp_lower(2, Up, Ui, Ux, Lp, Li, Lx);
        CHECK(k == 3 && Lp[1] == 2 && Li[1] == 1 && Lx[1] == 1 && Lx[2] == 3, "QP lower triangle: the upper triangle transposed (%ld)", k);
        k = brisk_qp_lower(2, Lp0, Li0, Lx0, Lp, Li, Lx);
        CHECK(k == 3 && Lp[1] == 2 && Lx[1] == 1, "QP lower triangle: the lower triangle as it is (%ld)", k);
        CHECK(brisk_qp_lower(2, Fp, Fi, Nx, Lp, Li, Lx) == -1, "QP lower triangle: an unsymmetric matrix refused");
    }
    {   /* (10 November 2026) brisk_solve_socp: an SOCP in general form (rows rl <= A x <= ru, bounds xl <= x <= xu, cones as lists of
         * variables), all answers known exactly */
        const char *qo[] = { "-q" }; const double I_ = INFINITY;
        /* 1. min t  s.t.  |(x1, x2)| <= t,  x1 + x2 >= 1: x1 = x2 = 1/2, t = 1/sqrt(2) */
        { const int Ap1[] = { 0, 0, 1, 2 }, Ai1[] = { 0, 0 }; const double Ax1[] = { 1, 1 }, c1[] = { 1, 0, 0 }, rl1[] = { 1 }, ru1[] = { I_ };
          const int qp1[] = { 0, 3 }, qi1[] = { 0, 1, 2 };
          int rc = brisk_solve_socp(3, 1, Ap1, Ai1, Ax1, c1, 0.0, rl1, ru1, NULL, NULL, 1, qp1, qi1, 0, NULL, NULL, 1, qo, rm);
          const double *x = brisk_result_sx(rm), *y = brisk_result_y(rm);
          CHECK(rc == 0 && brisk_result_status(rm) == 0 && fabs(brisk_result_pobj(rm) - sqrt(0.5)) < 1e-7 && x && fabs(x[1] - 0.5) < 1e-6 && fabs(x[2] - 0.5) < 1e-6
                && y && y[0] > 0 && fabs(brisk_result_dobj(rm) - sqrt(0.5)) < 1e-7, "general SOCP 1: rc %d status %d obj %.10g", rc, brisk_result_status(rm), brisk_result_pobj(rm)); }
        /* 2. min 10 - x1 - x2  s.t.  |(x1, x2)| <= t,  t <= 1 (a cone head's upper bound),  0 <= x1 <= 1/2 (a box),  -2 <= x1 - x2 <= 2 (a ranged
         *    row, inactive),  x1 + 7 x2 free (a row with no bound: dropped): x1 = 1/2, x2 = sqrt(3)/2, the value 10 - (1 + sqrt(3)) / 2 */
        { const int Ap2[] = { 0, 0, 2, 4 }, Ai2[] = { 0, 1, 0, 1 }; const double Ax2[] = { 1, 1, -1, 7 }, c2[] = { 0, -1, -1 };
          const double rl2[] = { -2, -I_ }, ru2[] = { 2, I_ }, xl2[] = { -I_, 0, -I_ }, xu2[] = { 1, 0.5, I_ };
          const int qp2[] = { 0, 3 }, qi2[] = { 0, 1, 2 };
          int rc = brisk_solve_socp(3, 2, Ap2, Ai2, Ax2, c2, 10.0, rl2, ru2, xl2, xu2, 1, qp2, qi2, 0, NULL, NULL, 1, qo, rm);
          const double *x = brisk_result_sx(rm), *y = brisk_result_y(rm), *z = brisk_result_sz(rm); const double v = 10 - (1 + sqrt(3.0)) / 2;
          CHECK(rc == 0 && fabs(brisk_result_pobj(rm) - v) < 1e-6 && x && fabs(x[1] - 0.5) < 1e-6 && fabs(x[2] - sqrt(0.75)) < 1e-5 && fabs(x[0] - 1) < 1e-6
                && y && fabs(y[0]) < 1e-6 && y[1] == 0.0 && z && z[1] < -1e-3, "general SOCP 2: rc %d obj %.10g (%.10g)", rc, brisk_result_pobj(rm), v); }
        /* 3. min x2  s.t.  2 x0 x1 >= x2^2 (rotated),  x0 = 1 (fixed),  x1 <= 2: x2 = -2 */
        { const int Ap3[] = { 0, 0, 0, 0 }; const double c3[] = { 0, 0, 1 }, xl3[] = { 1, -I_, -I_ }, xu3[] = { 1, 2, I_ };
          const int rp3[] = { 0, 3 }, ri3[] = { 0, 1, 2 };
          int rc = brisk_solve_socp(3, 0, Ap3, NULL, NULL, c3, 0.0, NULL, NULL, xl3, xu3, 0, NULL, NULL, 1, rp3, ri3, 1, qo, rm);
          const double *x = brisk_result_sx(rm);
          CHECK(rc == 0 && fabs(brisk_result_pobj(rm) + 2) < 1e-6 && x && fabs(x[0] - 1) < 1e-7 && fabs(x[1] - 2) < 1e-6, "general SOCP 3 (rotated): rc %d obj %.10g", rc, brisk_result_pobj(rm)); }
        /* 4. primal infeasible: 0 <= x1 <= 1, x1 >= 2, |x2| <= t */
        { const int Ap4[] = { 0, 1, 1, 1 }, Ai4[] = { 0 }; const double Ax4[] = { 1 }, c4[] = { 0, 1, 0 }, rl4[] = { 2 }, xl4[] = { 0, -I_, -I_ }, xu4[] = { 1, I_, I_ };
          const int qp4[] = { 0, 2 }, qi4[] = { 1, 2 };
          int rc = brisk_solve_socp(3, 1, Ap4, Ai4, Ax4, c4, 0.0, rl4, NULL, xl4, xu4, 1, qp4, qi4, 0, NULL, NULL, 1, qo, rm);
          CHECK(brisk_result_status(rm) == 1, "general SOCP 4 (infeasible): rc %d status %d", rc, brisk_result_status(rm)); }
        /* 5. dual infeasible (unbounded): min -t  s.t.  |x| <= t,  x <= 1 */
        { const int Ap5[] = { 0, 0, 0 }; const double c5[] = { -1, 0 }, xu5[] = { I_, 1 };
          const int qp5[] = { 0, 2 }, qi5[] = { 0, 1 };
          int rc = brisk_solve_socp(2, 0, Ap5, NULL, NULL, c5, 0.0, NULL, NULL, NULL, xu5, 1, qp5, qi5, 0, NULL, NULL, 1, qo, rm);
          const double *x = brisk_result_sx(rm);
          CHECK(brisk_result_status(rm) == 2 && x && x[0] > 0, "general SOCP 5 (unbounded): rc %d status %d", rc, brisk_result_status(rm)); }
        /* 6. bad data: a variable in two cones, an index out of range, rl > ru */
        { const int Ap6[] = { 0, 0, 0, 0 }; const double c6[] = { 1, 0, 0 };
          const int qp6[] = { 0, 2, 4 }, qi6[] = { 0, 1, 0, 2 }, qi6b[] = { 0, 3 }; const double rl6[] = { 1 }, ru6[] = { 0 };
          const int Ap6r[] = { 0, 1, 1, 1 }, Ai6r[] = { 0 }; const double Ax6r[] = { 1 };
          CHECK(brisk_solve_socp(3, 0, Ap6, NULL, NULL, c6, 0.0, NULL, NULL, NULL, NULL, 2, qp6, qi6, 0, NULL, NULL, 1, qo, rm) == 2, "general SOCP: a variable in two cones refused");
          CHECK(brisk_solve_socp(3, 0, Ap6, NULL, NULL, c6, 0.0, NULL, NULL, NULL, NULL, 1, qp6, qi6b, 0, NULL, NULL, 1, qo, rm) == 2, "general SOCP: a cone index out of range refused");
          CHECK(brisk_solve_socp(3, 1, Ap6r, Ai6r, Ax6r, c6, 0.0, rl6, ru6, NULL, NULL, 0, NULL, NULL, 0, NULL, NULL, 1, qo, rm) == 2, "general SOCP: rl > ru refused"); }
    }
    {   /* (10 November 2026) brisk_solve_lp: an LP in general form (rows rl <= A x <= ru, bounds xl <= x <= xu), the path of an MPS file */
        const double I_ = INFINITY;
        /* min -x1 - 2 x2 + c0  s.t.  x1 + x2 <= 4,  x1 + 3 x2 <= 6,  -1 <= x1 - x2 <= 3 (ranged),  0 <= x1 <= 3,  x2 >= 0:  x = (3, 1), -5 + c0 */
        const int Ap[] = { 0, 3, 6 }, Ai[] = { 0, 1, 2, 0, 1, 2 }; const double Ax[] = { 1, 1, 1, 1, 3, -1 }, c[] = { -1, -2 };
        const double rl[] = { -I_, -I_, -1 }, ru[] = { 4, 6, 3 }, xl[] = { 0, 0 }, xu[] = { 3, I_ };
        const char *lq[] = { "-q", "-threads", "1" };
        int rc = brisk_solve_lp(2, 3, Ap, Ai, Ax, c, 10.0, rl, ru, xl, xu, 3, lq, rm);
        const double *x = brisk_result_sx(rm);
        CHECK(rc == 0 && brisk_result_status(rm) == 0 && fabs(brisk_result_pobj(rm) - 5.0) < 1e-9 && x && fabs(x[0] - 3) < 1e-9 && fabs(x[1] - 1) < 1e-9,
              "general LP: rc %d status %d obj %.12g", rc, brisk_result_status(rm), brisk_result_pobj(rm));
        /* the duals from the dual simplex method: z = c - A'y, y <= 0 at the rows' upper sides, 0 on the inactive row, z_2 = 0, z_1 <= 0
         * at x1's upper bound, and the dual value 4 y0 + 6 y1 + 3 z1 equal to the optimum -5 */
        {   const char *l2[] = { "-q", "-threads", "1", "-lpmethod", "2" };
            rc = brisk_solve_lp(2, 3, Ap, Ai, Ax, c, 0.0, rl, ru, xl, xu, 5, l2, rm);
            const double *y = brisk_result_y(rm), *z = brisk_result_sz(rm);
            CHECK(rc == 0 && y && z && fabs(z[0] - (-1 - y[0] - y[1] - y[2])) < 1e-12 && fabs(z[1] - (-2 - y[0] - 3 * y[1] + y[2])) < 1e-12
                  && y[0] <= 1e-9 && y[1] <= 1e-9 && fabs(y[2]) <= 1e-9 && fabs(z[1]) <= 1e-9 && z[0] <= 1e-9 && fabs(4 * y[0] + 6 * y[1] + 3 * z[0] + 5) < 1e-9,
                  "general LP: the duals (rc %d, y %p, z %p)", rc, (const void *)y, (const void *)z); }
        /* (11 November 2026) the interior-point method alone (-lpmethod 1, one thread): its duals through the presolves' dual postsolve,
         * z = c - A'y and the dual value equal to the optimum within the method's accuracy */
        {   const char *l1[] = { "-q", "-threads", "1", "-lpmethod", "1" };
            rc = brisk_solve_lp(2, 3, Ap, Ai, Ax, c, 0.0, rl, ru, xl, xu, 5, l1, rm);
            const double *y = brisk_result_y(rm), *z = brisk_result_sz(rm);
            CHECK(rc == 0 && y && z && fabs(z[0] - (-1 - y[0] - y[1] - y[2])) < 1e-12 && fabs(z[1] - (-2 - y[0] - 3 * y[1] + y[2])) < 1e-12
                  && fabs(4 * y[0] + 6 * y[1] + 3 * z[0] + 5) < 1e-6,
                  "general LP: the interior-point method's duals (rc %d, y %p, z %p)", rc, (const void *)y, (const void *)z); }
        /* every method gives the same optimum */
        const char *meth[] = { "1", "2", "3", "4", "5" };
        for (int k = 0; k < 5; k++) {
            const char *lo[] = { "-q", "-threads", "1", "-lpmethod", meth[k] };
            rc = brisk_solve_lp(2, 3, Ap, Ai, Ax, c, 0.0, rl, ru, xl, xu, 5, lo, rm); x = brisk_result_sx(rm);
            CHECK(rc == 0 && fabs(brisk_result_pobj(rm) + 5.0) < 1e-7 && x && fabs(x[0] - 3) < 1e-6 && fabs(x[1] - 1) < 1e-6, "general LP, -lpmethod %s: rc %d obj %.12g", meth[k], rc, brisk_result_pobj(rm));
        }
        /* the same LP as an MPS file through brisk_solve_file: the same status, objective and x, bit for bit (one thread, and the default) */
        {   char fn[256]; const char *td = getenv("TMPDIR"); snprintf(fn, sizeof fn, "%s/brisk_capi_lp_%d.mps", td && *td ? td : "/tmp", (int)getpid());
            FILE *f = fopen(fn, "w");
            if (f) {
                fputs("NAME T\nROWS\n N obj\n L r0\n L r1\n G r2\nCOLUMNS\n x1 obj -1 r0 1\n x1 r1 1 r2 1\n x2 obj -2 r0 1\n x2 r1 3 r2 -1\n"
                      "RHS\n rhs r0 4 r1 6\n rhs r2 -1\nRANGES\n rng r2 4\nBOUNDS\n UP bnd x1 3\nENDATA\n", f);
                fclose(f);
                for (int t = 0; t < 2; t++) {
                    const char *o1[] = { "-q", "-threads", "1" }; const int no = t ? 1 : 3;
                    int r1 = brisk_solve_file(fn, no, o1, rf);
                    int r2 = brisk_solve_lp(2, 3, Ap, Ai, Ax, c, 0.0, rl, ru, xl, xu, no, o1, rm);
                    const double *xa = brisk_result_sx(rf), *xb = brisk_result_sx(rm);
                    CHECK(r1 == r2 && brisk_result_status(rf) == brisk_result_status(rm) && brisk_result_pobj(rf) == brisk_result_pobj(rm) && xa && xb
                          && !memcmp(xa, xb, 2 * sizeof(double)), "general LP: the MPS file's result, bit for bit (%s)", t ? "the default threads" : "one thread");
                }
                remove(fn);
            }
        }
        /* primal infeasible: x1 >= 2 with x1 <= 1 */
        {   const int Ap2[] = { 0, 1 }, Ai2[] = { 0 }; const double Ax2[] = { 1 }, c2[] = { 1 }, rl2[] = { 2 }, xl2[] = { 0 }, xu2[] = { 1 };
            rc = brisk_solve_lp(1, 1, Ap2, Ai2, Ax2, c2, 0.0, rl2, NULL, xl2, xu2, 3, lq, rm);
            CHECK(brisk_result_status(rm) == 1, "general LP (infeasible): rc %d status %d", rc, brisk_result_status(rm)); }
        /* dual infeasible: min -x1  s.t.  x1 - x2 <= 1,  x >= 0 */
        {   const int Ap3[] = { 0, 1, 2 }, Ai3[] = { 0, 0 }; const double Ax3[] = { 1, -1 }, c3[] = { -1, 0 }, ru3[] = { 1 }, xl3[] = { 0, 0 };
            rc = brisk_solve_lp(2, 1, Ap3, Ai3, Ax3, c3, 0.0, NULL, ru3, xl3, NULL, 3, lq, rm);
            CHECK(brisk_result_status(rm) == 2, "general LP (unbounded): rc %d status %d", rc, brisk_result_status(rm)); }
        /* bad data: a value not finite, an index out of range, columns out of order, rl > ru */
        {   const double cn[] = { NAN, -2 }; const int Aib[] = { 0, 1, 3, 0, 1, 2 }, Apb[] = { 0, 4, 3 }; const double rlb[] = { 5, -I_, -1 };
            CHECK(brisk_solve_lp(2, 3, Ap, Ai, Ax, cn, 0.0, rl, ru, xl, xu, 3, lq, rm) == 2, "general LP: a NaN cost refused");
            CHECK(brisk_solve_lp(2, 3, Ap, Aib, Ax, c, 0.0, rl, ru, xl, xu, 3, lq, rm) == 2, "general LP: a row index out of range refused");
            CHECK(brisk_solve_lp(2, 3, Apb, Ai, Ax, c, 0.0, rl, ru, xl, xu, 3, lq, rm) == 2, "general LP: columns out of order refused");
            CHECK(brisk_solve_lp(2, 3, Ap, Ai, Ax, c, 0.0, rlb, ru, xl, xu, 3, lq, rm) == 2, "general LP: rl > ru refused"); }
    }
    brisk_result_delete(rf);
    brisk_result_delete(rm);
    printf("%s (%d failures)\n", fails ? "FAILED" : "capi: all checks passed", fails);
    return fails != 0;
}
