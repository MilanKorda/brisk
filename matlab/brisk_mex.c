/* BRISK MEX gateway (MATLAB and GNU Octave).
 *
 *   [y, X, Z, info] = brisk_mex(filename, args)
 *   [y, X, Z, info] = brisk_mex(m, blocksizes, b, T, args)
 *   [x, y, z, info] = brisk_mex('sedumi', A, b, c, Kf, Kl, Kq, Kr, Ks, args)
 *   [x, y, z, info] = brisk_mex('qp', H, f, A, l, u, lb, ub, args)
 *   h = brisk_mex('qp_create', H, f, A, l, u, lb, ub, args); brisk_mex('qp_set_var_bounds', h, j, lo, hi);
 *   brisk_mex('qp_set_row_bounds', h, i, lo, hi); brisk_mex('qp_set_cost', h, f);
 *   [x, y, z, info] = brisk_mex('qp_solve', h); brisk_mex('qp_free', h)
 *   [x, y, z, info] = brisk_mex('qp_file', 'problem.qps', args)
 *
 * filename   an SDPA sparse file (.dat-s)
 * m          number of constraints
 * blocksizes block sizes, SDPA convention (LP blocks negative)
 * b          right-hand side (the SDPA "c" vector), length m
 * T          nnz x 5 array [mat blk i j value], the body of an SDPA sparse file: mat = 0 for F0
 *            (= -C), 1..m for F_i (= A_i); blk 1-based; i, j 1-based (i <= j; swapped if not)
 * args       cell array of BRISK command-line options, e.g. {'-q', '-acc', 'high'}; the MEX
 *            option '-silent' suppresses all printed output except error messages
 *
 * Problem solved (BRISK convention):
 *   (P) min <C,X> s.t. <A_i,X> = b_i, X in K     (D) max b'y s.t. C - sum_i y_i A_i = Z in K
 * Outputs: y (m x 1); X, Z cell arrays (SDP block: n x n; LP block: n x 1); info struct.
 * 'sedumi': the problem in SeDuMi format, min c'x s.t. A x = b, x in K: A sparse m x n, the
 * cone as K.f, K.l (scalars) and K.q, K.r, K.s (vectors of dimensions); x, y and z = c - A'y
 * are returned. Problems without semidefinite blocks go to the second-order cone solver.
 * The MATLAB wrappers brisk_sdpa.m and brisk_sedumi.m build these arguments.
 * 'qp': the convex QP min 1/2 x'Hx + f'x s.t. l <= A x <= u, lb <= x <= ub (H and A sparse, H symmetric or one
 * triangle of it; the vectors full, +-Inf for an absent bound), solved by the C library's brisk_solve_qp; 'qp_create'
 * keeps it as a QP model (brisk_qp_create) whose bounds, row bounds and cost can be changed and which 'qp_solve'
 * re-solves from the previous active set (j and i 1-based; h the model's number); 'qp_file' solves a QPS file as the command
 * line does (brisk_solve_file). brisk_qp.m and BriskQpModel.m
 * build these arguments.
 *
 * The data goes to brisk_run_data in memory (4.39; until 4.38 a temporary SDPA file) and is
 * solved by the same code as the command-line solver (brisk_run in main.c, built with
 * -DBRISK_LIBRARY): the same numbers as the file path, without the disk.
 */
#include "mex.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#define BRISK_NO_IO_MACROS
#include "brisk.h"
#include "sedumi.h"

static int g_silent = 0;
#include <dlfcn.h>
/* MATLAB holds mexPrintf output until a long MEX call returns; its (undocumented) ioFlush()
 * in libmwservices pushes it out at once, so the log of a long solve appears as it is printed.
 * Looked up at run time: where it does not exist (Octave, other MATLAB releases) nothing is done. */
static int mex_print(const char *s, int is_err) {
    static void (*io_flush)(void); static int looked;
    if (!looked) { io_flush = (void (*)(void))dlsym(RTLD_DEFAULT, "_Z7ioFlushv"); looked = 1; }
    if (!g_silent || is_err) { mexPrintf("%s", s); if (io_flush) io_flush(); }
    return 0;
}

static BriskData g_d;
static int *g_bs, *g_mat, *g_blk, *g_i, *g_j; static double *g_v, *g_c;
static void cleanup_tmp(void) { }   /* 4.39: no temporary file (the arrays are mxCalloc'd: freed with the call) */

static void fail(const char *id, const char *msg) {
    cleanup_tmp();
    mexErrMsgIdAndTxt(id, "%s", msg);
}

static double scalar_arg(const mxArray *a, const char *name) {
    if (!mxIsDouble(a) || mxIsComplex(a) || mxGetNumberOfElements(a) != 1) {
        char msg[256]; snprintf(msg, sizeof(msg), "brisk_mex: %s must be a real scalar", name);
        fail("brisk:input", msg);
    }
    return mxGetScalar(a);
}

/* the problem as an SDPA sparse file in the temporary directory */
static void set_data(int m, const mxArray *abs_, const mxArray *ab, const mxArray *aT) {
    if (!mxIsDouble(abs_) || mxIsComplex(abs_) || mxIsSparse(abs_)) fail("brisk:input", "brisk_mex: blocksizes must be a real full vector");
    if (!mxIsDouble(ab) || mxIsComplex(ab) || mxIsSparse(ab)) fail("brisk:input", "brisk_mex: b must be a real full vector");
    if (!mxIsDouble(aT) || mxIsComplex(aT) || mxIsSparse(aT)) fail("brisk:input", "brisk_mex: T must be a real full nnz x 5 array");
    const size_t nb = mxGetNumberOfElements(abs_);
    const double *bs = mxGetPr(abs_);
    if ((size_t)mxGetNumberOfElements(ab) != (size_t)m) fail("brisk:input", "brisk_mex: length(b) must equal m");
    const double *b = mxGetPr(ab);
    const size_t nt = mxGetM(aT);
    if (nt > 0 && mxGetN(aT) != 5) fail("brisk:input", "brisk_mex: T must have 5 columns [mat blk i j value]");
    const double *T = mxGetPr(aT);
    for (size_t k = 0; k < nb; k++)
        if (bs[k] == 0 || bs[k] != (double)(long)bs[k]) fail("brisk:input", "brisk_mex: block sizes must be nonzero integers");
    for (size_t q = 0; q < nt; q++) {
        const double mat = T[q], blk = T[q + nt], i = T[q + 2 * nt], j = T[q + 3 * nt];
        if (mat < 0 || mat > m || mat != (double)(long)mat) fail("brisk:input", "brisk_mex: T(:,1) must be integers in 0..m");
        if (blk < 1 || blk > (double)nb || blk != (double)(long)blk) fail("brisk:input", "brisk_mex: T(:,2) must be block indices 1..numel(blocksizes)");
        const double n = bs[(size_t)blk - 1] < 0 ? -bs[(size_t)blk - 1] : bs[(size_t)blk - 1];
        if (i < 1 || j < 1 || i > n || j > n || i != (double)(long)i || j != (double)(long)j)
            fail("brisk:input", "brisk_mex: T(:,3:4) must be indices within the block");
        if (bs[(size_t)blk - 1] < 0 && i != j) fail("brisk:input", "brisk_mex: LP blocks take diagonal entries only (i == j)");
    }
    /* 4.39 (ISSUES 23): the numbers go to brisk_run_data in memory (was: a temporary SDPA file) */
    g_d.m = m; g_d.nblk = (int)nb;
    g_bs = mxCalloc(nb + 1, sizeof(int)); g_mat = mxCalloc(nt + 1, sizeof(int)); g_blk = mxCalloc(nt + 1, sizeof(int));
    g_i = mxCalloc(nt + 1, sizeof(int)); g_j = mxCalloc(nt + 1, sizeof(int)); g_v = mxCalloc(nt + 1, sizeof(double)); g_c = mxCalloc((size_t)m + 1, sizeof(double));
    for (size_t k = 0; k < nb; k++) g_bs[k] = (int)bs[k];
    for (int i = 0; i < m; i++) g_c[i] = b[i];
    size_t q2 = 0;
    for (size_t q = 0; q < nt; q++) {
        const double v = T[q + 4 * nt];
        if (v == 0) continue;
        int i = (int)T[q + 2 * nt], j = (int)T[q + 3 * nt];
        if (i > j) { const int t = i; i = j; j = t; }
        g_mat[q2] = (int)T[q]; g_blk[q2] = (int)T[q + nt]; g_i[q2] = i; g_j[q2] = j; g_v[q2] = v; q2++;
    }
    g_d.bs = g_bs; g_d.c = g_c; g_d.nnz = q2; g_d.mat = g_mat; g_d.blk = g_blk; g_d.i = g_i; g_d.j = g_j; g_d.v = g_v;
}

static mxArray *block_cell(const BriskResult *r, double **B) {
    mxArray *c = mxCreateCellMatrix((mwSize)r->nblk, 1);
    for (int k = 0; k < r->nblk; k++) {
        const size_t n = (size_t)(r->bs[k] < 0 ? -r->bs[k] : r->bs[k]);
        /* 4.42: no matrix when the block is not returned (X of a large chordal block: a zero
         * n x n matrix of a 35 000 block would be 10 GB): an empty array instead */
        mxArray *a = !(B && B[k]) ? mxCreateDoubleMatrix(0, 0, mxREAL)
                   : r->bs[k] < 0 ? mxCreateDoubleMatrix((mwSize)n, 1, mxREAL) : mxCreateDoubleMatrix((mwSize)n, (mwSize)n, mxREAL);
        if (B && B[k]) memcpy(mxGetPr(a), B[k], sizeof(double) * (r->bs[k] < 0 ? n : n * n));
        mxSetCell(c, (mwIndex)k, a);
    }
    return c;
}

/* bound mode: the sides asked for, info.bounds.p / .d, each with its certificate (X: a cell of
 * blocks, or x for a problem in SeDuMi format; y); an empty struct when no side was asked for */
static mxArray *bound_side_struct(int side, int have, double value, double resid, double lammin, int valid, int certified,
                                  double rigorous, const char *src, mxArray *cert) {
    const char *f[] = { "kind", "have", "value", "resid", "lammin", "valid", "certified", "rigorous", "src", side == 1 ? "X" : "y" };
    mxArray *b = mxCreateStructMatrix(1, 1, 10, f);
    mxSetField(b, 0, "kind", mxCreateString(side == 1 ? "upper" : "lower"));
    mxSetField(b, 0, "have", mxCreateDoubleScalar(have));
    mxSetField(b, 0, "value", mxCreateDoubleScalar(have ? value : mxGetNaN()));
    mxSetField(b, 0, "resid", mxCreateDoubleScalar(resid));
    mxSetField(b, 0, "lammin", mxCreateDoubleScalar(lammin));
    mxSetField(b, 0, "valid", mxCreateDoubleScalar(valid));
    mxSetField(b, 0, "certified", mxCreateDoubleScalar(certified));
    mxSetField(b, 0, "rigorous", mxCreateDoubleScalar(certified ? rigorous : mxGetNaN()));
    mxSetField(b, 0, "src", mxCreateString(src));
    mxSetField(b, 0, side == 1 ? "X" : "y", cert ? cert : mxCreateDoubleMatrix(0, 0, mxREAL));
    return b;
}
static mxArray *vec_or_empty(const double *v, int n) {
    if (!v || n <= 0) return mxCreateDoubleMatrix(0, 0, mxREAL);
    mxArray *a = mxCreateDoubleMatrix((mwSize)n, 1, mxREAL);
    memcpy(mxGetPr(a), v, sizeof(double) * (size_t)n);
    return a;
}

static int *int_list(const mxArray *a, int *n, const char *name) {
    if (!mxIsDouble(a) || mxIsComplex(a) || mxIsSparse(a)) { char msg[128]; snprintf(msg, sizeof msg, "brisk_mex: %s must be a real full vector", name); fail("brisk:input", msg); }
    const size_t k = mxGetNumberOfElements(a);
    int *v = mxCalloc(k + 1, sizeof(int)); int c = 0;
    for (size_t i = 0; i < k; i++) if (mxGetPr(a)[i] > 0) v[c++] = (int)mxGetPr(a)[i];
    *n = c;
    return v;
}
/* [x, y, z, info] = brisk_mex('sedumi', A, b, c, Kf, Kl, Kq, Kr, Ks, args) */
static void mex_sedumi(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
    if (nrhs < 9 || nrhs > 10) fail("brisk:input", "brisk_mex: usage brisk_mex('sedumi', A, b, c, Kf, Kl, Kq, Kr, Ks, args)");
    const mxArray *A = prhs[1];
    if (!mxIsSparse(A) || !mxIsDouble(A) || mxIsComplex(A)) fail("brisk:input", "brisk_mex: A must be a real sparse matrix");
    SedumiProb P; memset(&P, 0, sizeof P);
    P.m = (int)mxGetM(A); P.n = (int)mxGetN(A);
    if ((int)mxGetNumberOfElements(prhs[2]) != P.m || (int)mxGetNumberOfElements(prhs[3]) != P.n || mxIsSparse(prhs[2]) || mxIsSparse(prhs[3]) || !mxIsDouble(prhs[2]) || !mxIsDouble(prhs[3]))
        fail("brisk:input", "brisk_mex: b and c must be full real vectors matching A");
    const mwIndex *jc = mxGetJc(A), *ir = mxGetIr(A);
    const size_t nnz = (size_t)jc[P.n];
    P.Ap = mxCalloc((size_t)P.n + 2, sizeof(int)); P.Ai = mxCalloc(nnz + 1, sizeof(int));
    for (int j = 0; j <= P.n; j++) P.Ap[j] = (int)jc[j];
    for (size_t p = 0; p < nnz; p++) P.Ai[p] = (int)ir[p];
    P.Ax = mxGetPr(A); P.b = mxGetPr(prhs[2]); P.c = mxGetPr(prhs[3]);
    P.nf = (int)scalar_arg(prhs[4], "K.f"); P.nl = (int)scalar_arg(prhs[5], "K.l");
    P.q = int_list(prhs[6], &P.nq, "K.q"); P.r = int_list(prhs[7], &P.nr, "K.r"); P.s = int_list(prhs[8], &P.ns, "K.s");
    long tot = (long)P.nf + P.nl;
    for (int k = 0; k < P.nq; k++) tot += P.q[k];
    for (int k = 0; k < P.nr; k++) { if (P.r[k] < 2) fail("brisk:input", "brisk_mex: a rotated cone (K.r) has at least 2 entries"); tot += P.r[k]; }
    for (int k = 0; k < P.ns; k++) tot += (long)P.s[k] * P.s[k];
    if (P.nf < 0 || P.nl < 0 || tot != P.n) fail("brisk:input", "brisk_mex: K does not match the number of columns of A");
    int nargs = 0;
    const mxArray *acell = nrhs == 10 ? prhs[9] : NULL;
    if (acell) { if (!mxIsCell(acell)) fail("brisk:input", "brisk_mex: args must be a cell array of strings"); nargs = (int)mxGetNumberOfElements(acell); }
    char **argv = mxCalloc((size_t)nargs + 2, sizeof(char *));
    int na = 0;
    g_silent = 0;
    for (int i = 0; i < nargs; i++) {
        const mxArray *e = mxGetCell(acell, (mwIndex)i);
        if (!e || !mxIsChar(e)) fail("brisk:input", "brisk_mex: every element of args must be a string");
        char *s = mxArrayToString(e);
        if (!strcmp(s, "-silent")) { g_silent = 1; mxFree(s); continue; }
        argv[na++] = s;
    }
    if (g_silent) argv[na++] = "-q";
    SedumiRes R;
    brisk_print_hook = mex_print;
    const int rc = brisk_run_sedumi(&P, na, argv, &R);
    brisk_print_hook = NULL;
    if (rc < 0 || R.status < 0) {
        sedumi_result_free(&R);
        if (rc < 0) mexErrMsgIdAndTxt("brisk:aborted", "BRISK stopped (code %d): an invalid option or out of memory; see the messages above", -rc == 1000 ? 0 : -rc);
        mexErrMsgIdAndTxt("brisk:input", "BRISK could not solve the problem (code %d): see the messages above", rc);
    }
    const int have_x = P.ns == 0 || R.have_x;
    plhs[0] = mxCreateDoubleMatrix((mwSize)P.n, 1, mxREAL);
    if (R.x && have_x) memcpy(mxGetPr(plhs[0]), R.x, sizeof(double) * (size_t)P.n);
    if (nlhs > 1) { plhs[1] = mxCreateDoubleMatrix((mwSize)P.m, 1, mxREAL); if (R.y) memcpy(mxGetPr(plhs[1]), R.y, sizeof(double) * (size_t)P.m); }
    if (nlhs > 2) { plhs[2] = mxCreateDoubleMatrix((mwSize)P.n, 1, mxREAL); if (R.z) memcpy(mxGetPr(plhs[2]), R.z, sizeof(double) * (size_t)P.n); }
    if (nlhs > 3) {
        const char *fields[] = { "status", "statuscode", "exitcode", "iter", "pobj", "dobj", "dimacs", "time", "have_x", "version", "bounds" };
        mxArray *info = mxCreateStructMatrix(1, 1, 11, fields);
        {   /* bound mode: each side asked for, with its certificate (x for p, y for d) */
            const char *sf[] = { "p", "d" };
            int ns_ = 0; const char *names[2];
            for (int s = 0; s < 2; s++) if (R.bnd[s].asked) names[ns_++] = sf[s];
            mxArray *bs_ = mxCreateStructMatrix(1, 1, ns_, names);
            for (int s = 0; s < 2; s++) if (R.bnd[s].asked)
                mxSetField(bs_, 0, sf[s], bound_side_struct(s + 1, R.bnd[s].have, R.bnd[s].value, R.bnd[s].resid, R.bnd[s].lammin, R.bnd[s].valid,
                                                            R.bnd[s].certified, R.bnd[s].rigorous, R.bnd[s].src,
                                                            s == 0 ? vec_or_empty(R.bnd[0].x, P.n) : vec_or_empty(R.bnd[1].y, P.m)));
            mxSetField(info, 0, "bounds", bs_);
        }
        mxSetField(info, 0, "status", mxCreateString(R.status_str));
        mxSetField(info, 0, "statuscode", mxCreateDoubleScalar(R.status));
        mxSetField(info, 0, "exitcode", mxCreateDoubleScalar(rc));
        mxSetField(info, 0, "iter", mxCreateDoubleScalar(R.iters));
        mxSetField(info, 0, "pobj", mxCreateDoubleScalar(R.pobj));
        mxSetField(info, 0, "dobj", mxCreateDoubleScalar(R.dobj));
        mxArray *d = mxCreateDoubleMatrix(1, 6, mxREAL);
        for (int e = 0; e < 6; e++) mxGetPr(d)[e] = R.err[e];
        mxSetField(info, 0, "dimacs", d);
        mxSetField(info, 0, "time", mxCreateDoubleScalar(R.time));
        mxSetField(info, 0, "have_x", mxCreateDoubleScalar(have_x));
        mxSetField(info, 0, "version", mxCreateString(BRISK_VERSION));
        plhs[3] = info;
    }
    sedumi_result_free(&R);
}

/* ---- QPs: the C library's functions (capi.c) ------------------------------------------------------------ */
BriskResult *brisk_result_new(void);
void brisk_result_delete(BriskResult *r);
int brisk_result_status(const BriskResult *r);
const char *brisk_result_status_str(const BriskResult *r);
int brisk_result_iterations(const BriskResult *r);
int brisk_result_m(const BriskResult *r);
int brisk_result_sn(const BriskResult *r);
double brisk_result_pobj(const BriskResult *r);
double brisk_result_time(const BriskResult *r);
void brisk_result_dimacs(const BriskResult *r, double *err);
const double *brisk_result_sx(const BriskResult *r);
const double *brisk_result_sz(const BriskResult *r);
const double *brisk_result_y(const BriskResult *r);
void brisk_set_output(int mode, int (*cb)(const char *s, int is_err));
int brisk_solve_file(const char *fname, int nopt, const char *const *opts, BriskResult *res);
int brisk_solve_qp(int n, int m, const int *Hp, const int *Hi, const double *Hx, const double *c, double c0,
                   const int *Ap, const int *Ai, const double *Ax, const double *rl, const double *ru,
                   const double *xl, const double *xu, int nopt, const char *const *opts, BriskResult *res);
long brisk_qp_lower(int n, const int *Hp, const int *Hi, const double *Hx, int *Lp, int *Li, double *Lx);
void *brisk_qp_create(int n, int m, const int *Hp, const int *Hi, const double *Hx, const double *c, double c0,
                      const int *Ap, const int *Ai, const double *Ax, const double *rl, const double *ru,
                      const double *xl, const double *xu, int nopt, const char *const *opts);
int brisk_qp_set_var_bounds(void *h, int j, double lo, double hi);
int brisk_qp_set_row_bounds(void *h, int i, double lo, double hi);
int brisk_qp_set_cost(void *h, const double *c);
int brisk_qp_resolve(void *h, BriskResult *res);
void brisk_qp_destroy(void *h);

/* the QP models of this MEX file, numbered from 1; freed by 'qp_free' and when the MEX file is cleared */
#define QP_MODELS 4096
static void *g_qp[QP_MODELS];
static int g_qp_silent[QP_MODELS], g_qp_n[QP_MODELS];
static const char *const qp_modes[] = { "qp", "qp_create", "qp_set_var_bounds", "qp_set_row_bounds", "qp_set_cost", "qp_solve", "qp_free", "qp_file" };
static void qp_free_all(void) { for (int k = 0; k < QP_MODELS; k++) if (g_qp[k]) { brisk_qp_destroy(g_qp[k]); g_qp[k] = NULL; } }

/* a QP's data from H, f, A, l, u, lb, ub (prhs[0..6]): H's lower triangle and A by columns as int arrays */
typedef struct { int n, m, *Hp, *Hi, *Ap, *Ai; double *Hx, *f, *Ax, *l, *u, *lb, *ub; } MexQp;
static void qp_vec(const mxArray *a, int k, const char *name) {
    if (!mxIsDouble(a) || mxIsComplex(a) || mxIsSparse(a) || (int)mxGetNumberOfElements(a) != k) {
        char msg[160]; snprintf(msg, sizeof msg, "brisk_mex: %s must be a full real vector of length %d", name, k); fail("brisk:input", msg);
    }
}
static void qp_data(const mxArray *const *a, MexQp *Q) {
    const mxArray *H = a[0], *A = a[2];
    if (!mxIsSparse(H) || !mxIsDouble(H) || mxIsComplex(H) || mxGetM(H) != mxGetN(H)) fail("brisk:input", "brisk_mex: H must be a real sparse square matrix");
    if (!mxIsSparse(A) || !mxIsDouble(A) || mxIsComplex(A)) fail("brisk:input", "brisk_mex: A must be a real sparse matrix");
    Q->n = (int)mxGetN(H); Q->m = (int)mxGetM(A);
    if ((int)mxGetN(A) != Q->n) fail("brisk:input", "brisk_mex: A must have as many columns as H");
    qp_vec(a[1], Q->n, "f"); qp_vec(a[3], Q->m, "l"); qp_vec(a[4], Q->m, "u"); qp_vec(a[5], Q->n, "lb"); qp_vec(a[6], Q->n, "ub");
    Q->f = mxGetPr(a[1]); Q->l = mxGetPr(a[3]); Q->u = mxGetPr(a[4]); Q->lb = mxGetPr(a[5]); Q->ub = mxGetPr(a[6]);
    const mwIndex *jc = mxGetJc(H), *ir = mxGetIr(H);
    const size_t hz = (size_t)jc[Q->n];
    int *Hp = mxCalloc((size_t)Q->n + 1, sizeof(int)), *Hi = mxCalloc(hz + 1, sizeof(int));
    for (int j = 0; j <= Q->n; j++) Hp[j] = (int)jc[j];
    for (size_t p = 0; p < hz; p++) Hi[p] = (int)ir[p];
    Q->Hp = mxCalloc((size_t)Q->n + 1, sizeof(int)); Q->Hi = mxCalloc(hz + 1, sizeof(int)); Q->Hx = mxCalloc(hz + 1, sizeof(double));
    if (brisk_qp_lower(Q->n, Hp, Hi, mxGetPr(H), Q->Hp, Q->Hi, Q->Hx) < 0) fail("brisk:input", "brisk_mex: H is neither symmetric nor triangular");
    jc = mxGetJc(A); ir = mxGetIr(A);
    const size_t az = (size_t)jc[Q->n];
    Q->Ap = mxCalloc((size_t)Q->n + 1, sizeof(int)); Q->Ai = mxCalloc(az + 1, sizeof(int));
    for (int j = 0; j <= Q->n; j++) Q->Ap[j] = (int)jc[j];
    for (size_t p = 0; p < az; p++) Q->Ai[p] = (int)ir[p];
    Q->Ax = mxGetPr(A);
}
/* the options from a cell array of strings; '-silent' is the MEX's own (no output but errors) */
static char **qp_args(const mxArray *acell, int *na, int *silent) {
    int nargs = 0;
    if (acell) { if (!mxIsCell(acell)) fail("brisk:input", "brisk_mex: args must be a cell array of strings"); nargs = (int)mxGetNumberOfElements(acell); }
    char **av = mxCalloc((size_t)nargs + 2, sizeof(char *));
    *na = 0; *silent = 0;
    for (int i = 0; i < nargs; i++) {
        const mxArray *e = mxGetCell(acell, (mwIndex)i);
        if (!e || !mxIsChar(e)) fail("brisk:input", "brisk_mex: every element of args must be a string");
        char *t = mxArrayToString(e);
        if (!strcmp(t, "-silent")) { *silent = 1; mxFree(t); continue; }
        av[(*na)++] = t;
    }
    return av;
}
/* x, y, z and info from a QP result */
static void qp_outputs(int nlhs, mxArray *plhs[], BriskResult *r) {
    if (brisk_result_status(r) < 0) {
        brisk_result_delete(r);
        mexErrMsgIdAndTxt("brisk:input", "BRISK did not solve the QP (not convex, or an unknown option): see the messages above");
    }
    const int n = brisk_result_sn(r), m = brisk_result_m(r);
    plhs[0] = vec_or_empty(brisk_result_sx(r), n);
    if (nlhs > 1) plhs[1] = m > 0 ? vec_or_empty(brisk_result_y(r), m) : mxCreateDoubleMatrix(0, 1, mxREAL);
    if (nlhs > 2) plhs[2] = vec_or_empty(brisk_result_sz(r), n);
    if (nlhs > 3) {
        const char *fields[] = { "status", "statuscode", "obj", "iter", "time", "errors", "version" };
        mxArray *info = mxCreateStructMatrix(1, 1, 7, fields);
        double err[6]; brisk_result_dimacs(r, err);
        mxArray *e = mxCreateDoubleMatrix(1, 3, mxREAL);
        mxGetPr(e)[0] = err[0]; mxGetPr(e)[1] = err[2]; mxGetPr(e)[2] = err[4];
        mxSetField(info, 0, "status", mxCreateString(brisk_result_status_str(r)));
        mxSetField(info, 0, "statuscode", mxCreateDoubleScalar(brisk_result_status(r)));
        mxSetField(info, 0, "obj", mxCreateDoubleScalar(brisk_result_pobj(r)));
        mxSetField(info, 0, "iter", mxCreateDoubleScalar(brisk_result_iterations(r)));
        mxSetField(info, 0, "time", mxCreateDoubleScalar(brisk_result_time(r)));
        mxSetField(info, 0, "errors", e);
        mxSetField(info, 0, "version", mxCreateString(BRISK_VERSION));
        plhs[3] = info;
    }
    brisk_result_delete(r);
}
static int qp_model(const mxArray *a) {
    const int k = (int)scalar_arg(a, "the QP model") - 1;
    if (k < 0 || k >= QP_MODELS || !g_qp[k]) fail("brisk:input", "brisk_mex: not a QP model (freed, or never created)");
    return k;
}
static void mex_qp(const char *mode, int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
    mexAtExit(qp_free_all);
    brisk_set_output(2, mex_print);
    if (!strcmp(mode, "qp_file")) {
        if (nrhs < 2 || nrhs > 3 || !mxIsChar(prhs[1])) fail("brisk:input", "brisk_mex: usage brisk_mex('qp_file', 'problem.qps', args)");
        char *fname = mxArrayToString(prhs[1]);
        int na, silent; char **av = qp_args(nrhs == 3 ? prhs[2] : NULL, &na, &silent);
        g_silent = silent;
        if (silent) av[na++] = "-q";
        BriskResult *r = brisk_result_new();
        brisk_solve_file(fname, na, (const char *const *)av, r);
        mxFree(fname);
        qp_outputs(nlhs, plhs, r);
        return;
    }
    if (!strcmp(mode, "qp") || !strcmp(mode, "qp_create")) {
        if (nrhs < 8 || nrhs > 9) fail("brisk:input", "brisk_mex: usage brisk_mex('qp', H, f, A, l, u, lb, ub, args)");
        MexQp Q; qp_data(prhs + 1, &Q);
        int na, silent; char **av = qp_args(nrhs == 9 ? prhs[8] : NULL, &na, &silent);
        g_silent = silent;
        if (silent) av[na++] = "-q";
        if (!strcmp(mode, "qp")) {
            BriskResult *r = brisk_result_new();
            brisk_solve_qp(Q.n, Q.m, Q.Hp, Q.Hi, Q.Hx, Q.f, 0.0, Q.Ap, Q.Ai, Q.Ax, Q.l, Q.u, Q.lb, Q.ub, na, (const char *const *)av, r);
            qp_outputs(nlhs, plhs, r);
            return;
        }
        int k = 0; while (k < QP_MODELS && g_qp[k]) k++;
        if (k == QP_MODELS) fail("brisk:input", "brisk_mex: too many QP models (free some with 'qp_free')");
        g_qp[k] = brisk_qp_create(Q.n, Q.m, Q.Hp, Q.Hi, Q.Hx, Q.f, 0.0, Q.Ap, Q.Ai, Q.Ax, Q.l, Q.u, Q.lb, Q.ub, na, (const char *const *)av);
        if (!g_qp[k]) fail("brisk:input", "brisk_mex: the QP model was not created (an unknown option, or bad data)");
        g_qp_silent[k] = silent; g_qp_n[k] = Q.n;
        plhs[0] = mxCreateDoubleScalar(k + 1);
        return;
    }
    if (nrhs < 2) fail("brisk:input", "brisk_mex: the QP model is missing");
    const int k = qp_model(prhs[1]);
    if (!strcmp(mode, "qp_set_var_bounds") || !strcmp(mode, "qp_set_row_bounds")) {
        if (nrhs != 5) fail("brisk:input", "brisk_mex: usage brisk_mex('qp_set_var_bounds' or 'qp_set_row_bounds', h, index, lo, hi)");
        const int i = (int)scalar_arg(prhs[2], "the index") - 1;
        const double lo = scalar_arg(prhs[3], "lo"), hi = scalar_arg(prhs[4], "hi");
        const int rc = mode[7] == 'v' ? brisk_qp_set_var_bounds(g_qp[k], i, lo, hi) : brisk_qp_set_row_bounds(g_qp[k], i, lo, hi);
        if (rc) fail("brisk:input", "brisk_mex: the index is out of range");
    } else if (!strcmp(mode, "qp_set_cost")) {
        if (nrhs != 3) fail("brisk:input", "brisk_mex: usage brisk_mex('qp_set_cost', h, f)");
        qp_vec(prhs[2], g_qp_n[k], "f");
        brisk_qp_set_cost(g_qp[k], mxGetPr(prhs[2]));
    } else if (!strcmp(mode, "qp_solve")) {
        g_silent = g_qp_silent[k];
        BriskResult *r = brisk_result_new();
        brisk_qp_resolve(g_qp[k], r);
        qp_outputs(nlhs, plhs, r);
    } else if (!strcmp(mode, "qp_free")) {
        brisk_qp_destroy(g_qp[k]); g_qp[k] = NULL;
    } else fail("brisk:input", "brisk_mex: unknown QP mode");
}

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
    const mxArray *acell = NULL;
    if (nrhs >= 1 && mxIsChar(prhs[0])) {
        char *mode = mxArrayToString(prhs[0]);
        for (size_t q = 0; mode && q < sizeof qp_modes / sizeof *qp_modes; q++)
            if (!strcmp(mode, qp_modes[q])) { mxFree(mode); mex_qp(qp_modes[q], nlhs, plhs, nrhs, prhs); return; }
        mxFree(mode);
    }
    if (nrhs > 2 && mxIsChar(prhs[0])) {
        char *mode = mxArrayToString(prhs[0]);
        const int sd = mode && !strcmp(mode, "sedumi");
        mxFree(mode);
        if (sd) { mex_sedumi(nlhs, plhs, nrhs, prhs); return; }
    }
    char *fname = NULL;
    int own_file = 0;
    if (nrhs >= 1 && mxIsChar(prhs[0])) {
        if (nrhs > 2) fail("brisk:input", "brisk_mex: usage brisk_mex(filename, args)");
        fname = mxArrayToString(prhs[0]);
        if (nrhs == 2) acell = prhs[1];
    } else {
        if (nrhs < 4 || nrhs > 5) fail("brisk:input", "brisk_mex: usage brisk_mex(m, blocksizes, b, T, args) or brisk_mex(filename, args)");
        const double md = scalar_arg(prhs[0], "m");
        if (md < 1 || md != (double)(long)md || md > 2e9) fail("brisk:input", "brisk_mex: m must be a positive integer");
        set_data((int)md, prhs[1], prhs[2], prhs[3]);
        if (nrhs == 5) acell = prhs[4];
        fname = "(problem in memory)";
        own_file = 1;
    }
    /* argv: "brisk" file options... */
    int nargs = 0;
    if (acell) {
        if (!mxIsCell(acell)) { if (!own_file) mxFree(fname); fail("brisk:input", "brisk_mex: args must be a cell array of strings"); }
        nargs = (int)mxGetNumberOfElements(acell);
    }
    char **argv = mxCalloc((size_t)nargs + 3, sizeof(char *));
    argv[0] = "brisk";
    argv[1] = fname;
    int na = 2;
    g_silent = 0;
    for (int i = 0; i < nargs; i++) {
        const mxArray *e = mxGetCell(acell, (mwIndex)i);
        if (!e || !mxIsChar(e)) { if (!own_file) mxFree(fname); fail("brisk:input", "brisk_mex: every element of args must be a string"); }
        char *s = mxArrayToString(e);
        if (!strcmp(s, "-silent")) { g_silent = 1; mxFree(s); continue; }   /* MEX only: no output except errors */
        argv[na++] = s;
    }
    BriskResult res;
    brisk_print_hook = mex_print;
    int rc;
    if (own_file) { argv[1] = "brisk"; rc = brisk_run_data(&g_d, na - 1, argv + 1, &res); }   /* argv without the file name */
    else rc = brisk_run(na, argv, &res);
    brisk_print_hook = NULL;
    for (int i = 2; i < na; i++) mxFree(argv[i]);
    mxFree(argv);
    if (!own_file) mxFree(fname);
    cleanup_tmp();
    if (rc < 0 || res.status < 0) {
        brisk_result_free(&res);
        if (rc < 0) mexErrMsgIdAndTxt("brisk:aborted", "BRISK stopped (code %d): an invalid option or out of memory; see the messages above", -rc == 1000 ? 0 : -rc);
        mexErrMsgIdAndTxt("brisk:input", "BRISK could not solve the problem (code %d): see the messages above", rc);
    }
    /* outputs */
    plhs[0] = mxCreateDoubleMatrix((mwSize)res.m, 1, mxREAL);
    if (res.y) memcpy(mxGetPr(plhs[0]), res.y, sizeof(double) * (size_t)res.m);
    if (nlhs > 1) plhs[1] = block_cell(&res, res.have_x ? res.X : NULL);
    if (nlhs > 2) plhs[2] = block_cell(&res, res.Z);
    if (nlhs > 3) {
        const char *fields[] = { "status", "statuscode", "exitcode", "iter", "pobj", "dobj", "dimacs", "time", "have_x", "version",
                                 "bound", "resolves", "cause", "bounds" };
        mxArray *info = mxCreateStructMatrix(1, 1, 14, fields);
        {   /* bound mode: each side asked for (info.bounds.p / .d), with its certificate (a cell of X blocks for p, y for d) */
            const char *sf[] = { "p", "d" };
            int ns_ = 0; const char *names[2];
            for (int s = 0; s < 2; s++) if (res.bnd[s].asked) names[ns_++] = sf[s];
            mxArray *bs_ = mxCreateStructMatrix(1, 1, ns_, names);
            for (int s = 0; s < 2; s++) if (res.bnd[s].asked)
                mxSetField(bs_, 0, sf[s], bound_side_struct(s + 1, res.bnd[s].have, res.bnd[s].value, res.bnd[s].resid, res.bnd[s].lammin, res.bnd[s].valid,
                                                            res.bnd[s].certified, res.bnd[s].rigorous, res.bnd[s].src,
                                                            s == 0 ? (res.bnd[0].X ? block_cell(&res, res.bnd[0].X) : NULL) : vec_or_empty(res.bnd[1].y, res.m)));
            mxSetField(info, 0, "bounds", bs_);
        }
        {   /* 4.38: the bound mode (opts.bound), the re-solves and the likely cause */
            const char *bf[] = { "side", "value", "resid", "lammin", "valid", "certified", "rigorous" };
            mxArray *b = mxCreateStructMatrix(1, 1, 7, bf);
            mxSetField(b, 0, "side", mxCreateString(res.bound_side == 1 ? "p" : res.bound_side == 2 ? "d" : ""));
            mxSetField(b, 0, "value", mxCreateDoubleScalar(res.bound_side ? res.bound_value : mxGetNaN()));
            mxSetField(b, 0, "resid", mxCreateDoubleScalar(res.bound_resid));
            mxSetField(b, 0, "lammin", mxCreateDoubleScalar(res.bound_lammin));
            mxSetField(b, 0, "valid", mxCreateDoubleScalar(res.bound_valid));
            mxSetField(b, 0, "certified", mxCreateDoubleScalar(res.bound_certified));
            mxSetField(b, 0, "rigorous", mxCreateDoubleScalar(res.bound_side ? res.bound_rigorous : mxGetNaN()));
            mxSetField(info, 0, "bound", b);
            const char *rf[] = { "n", "time" };
            mxArray *rs = mxCreateStructMatrix(1, 1, 2, rf);
            mxSetField(rs, 0, "n", mxCreateDoubleScalar(res.n_resolves));
            mxSetField(rs, 0, "time", mxCreateDoubleScalar(res.t_resolves));
            mxSetField(info, 0, "resolves", rs);
            mxSetField(info, 0, "cause", mxCreateString(res.cause));
        }
        mxSetField(info, 0, "status", mxCreateString(res.status_str));
        mxSetField(info, 0, "statuscode", mxCreateDoubleScalar(res.status));
        mxSetField(info, 0, "exitcode", mxCreateDoubleScalar(res.exit_code));
        mxSetField(info, 0, "iter", mxCreateDoubleScalar(res.iters));
        mxSetField(info, 0, "pobj", mxCreateDoubleScalar(res.pobj));
        mxSetField(info, 0, "dobj", mxCreateDoubleScalar(res.dobj));
        mxArray *d = mxCreateDoubleMatrix(1, 6, mxREAL);
        for (int e = 0; e < 6; e++) mxGetPr(d)[e] = res.err[e + 1];
        mxSetField(info, 0, "dimacs", d);
        mxSetField(info, 0, "time", mxCreateDoubleScalar(res.time));
        mxSetField(info, 0, "have_x", mxCreateDoubleScalar(res.have_x));
        mxSetField(info, 0, "version", mxCreateString(BRISK_VERSION));
        plhs[3] = info;
    }
    brisk_result_free(&res);
}
