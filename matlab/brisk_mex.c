/* BRISK MEX gateway (MATLAB and GNU Octave).
 *
 *   [y, X, Z, info] = brisk_mex(filename, args)
 *   [y, X, Z, info] = brisk_mex(m, blocksizes, b, T, args)
 *   [x, y, z, info] = brisk_mex('sedumi', A, b, c, Kf, Kl, Kq, Kr, Ks, args)
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
static int mex_print(const char *s, int is_err) { if (!g_silent || is_err) mexPrintf("%s", s); return 0; }

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
        const char *fields[] = { "status", "statuscode", "exitcode", "iter", "pobj", "dobj", "dimacs", "time", "have_x", "version" };
        mxArray *info = mxCreateStructMatrix(1, 1, 10, fields);
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

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
    const mxArray *acell = NULL;
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
                                 "bound", "resolves", "cause" };
        mxArray *info = mxCreateStructMatrix(1, 1, 13, fields);
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
