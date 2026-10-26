/* qcqp.c - convex quadratically constrained QPs by the SOCP solver (the SOCP developer, 22 October 2026).
 *
 *     min 1/2 x'Hx + c'x + c0   s.t.  rl <= a_i'x (+ 1/2 x'Q_i x) <= ru,  xl <= x <= xu
 *
 * Each quadratic row with one finite side, Q_i positive semidefinite on that side (Q_i for an upper side, -Q_i for a
 * lower one), becomes a rotated cone: with Q_i = F_i'F_i, u - a'x >= 1/2 |F_i x|^2 is (u - a'x, 1, F_i x) in
 * {2 t1 t2 >= |w|^2}; a quadratic objective becomes t >= 1/2 |F_0 x|^2, the cone (t, 1, F_0 x). This is SeDuMi's dual
 * form, max b'y s.t. c - A'y in K, with y = (x, t): equality rows and fixed variables are free entries (zero), the
 * other rows and the bounds nonnegative entries. The factors come from a Cholesky factorization with diagonal
 * pivoting of each connected block of Q_i on its support, which also decides convexity (a remaining diagonal below
 * -1e-9 times the largest: not convex; the rank where the remaining ones fall below 1e-12 times it). MOSEK and
 * Gurobi solve convex QCQPs through the same reformulation. The SOCP solver is the host's (qcqp_socp, set by the
 * command line); without it a QCQP is refused. */
#include "qp.h"
#include "sedumi.h"
#ifndef BL         /* (BL and dpotrf_ come with brisk.h through sedumi.h) */
#define BL(name) name
void dpotrf_(const char *, const int *, double *, const int *, int *);
#endif
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

int (*qcqp_socp)(const SedumiProb *P, int argc, char **argv, SedumiRes *R) = NULL;

static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + 1e-9 * t.tv_nsec; }
static void *xm(size_t s) { void *p = malloc(s ? s : 1); if (!p) { fprintf(stderr, "brisk: out of memory (QCQP)\n"); exit(2); } return p; }
static void *xz(size_t s) { void *p = calloc(1, s ? s : 1); if (!p) { fprintf(stderr, "brisk: out of memory (QCQP)\n"); exit(2); } return p; }
static int fin(double v) { return fabs(v) < QP_INF; }

/* the columns of SeDuMi's A, built one at a time: column e has c[e] and the entries (row r of y, value) */
typedef struct { int n, cap; double *c; int *cp; int nnz, zcap; int *ri; double *rv; } Cols;
static void col_begin(Cols *C, double c) {
    if (C->n == C->cap) { C->cap = C->cap ? 2 * C->cap : 1024; C->c = realloc(C->c, sizeof(double) * C->cap); C->cp = realloc(C->cp, sizeof(int) * (C->cap + 1)); }
    C->c[C->n] = c; C->cp[C->n] = C->nnz; C->n++; C->cp[C->n] = C->nnz;
}
static void col_add(Cols *C, int r, double v) {
    if (v == 0) return;
    if (C->nnz == C->zcap) { C->zcap = C->zcap ? 2 * C->zcap : 4096; C->ri = realloc(C->ri, sizeof(int) * C->zcap); C->rv = realloc(C->rv, sizeof(double) * C->zcap); }
    C->ri[C->nnz] = r; C->rv[C->nnz] = v; C->nnz++; C->cp[C->n] = C->nnz;
}

/* a quadratic form's factor: rows of F (each a list over the variables), Q = F'F */
typedef struct { int nr; int *rp, *ri; double *rv; int nz, cap, rcap; } Fac;
static void fac_row(Fac *F) {
    if (F->nr + 1 >= F->rcap) { F->rcap = F->rcap ? 2 * F->rcap : 64; F->rp = realloc(F->rp, sizeof(int) * (F->rcap + 1)); }
    if (F->nr == 0) F->rp[0] = 0;
    F->nr++; F->rp[F->nr] = F->nz;
}
static void fac_add(Fac *F, int j, double v) {
    if (F->nz == F->cap) { F->cap = F->cap ? 2 * F->cap : 1024; F->ri = realloc(F->ri, sizeof(int) * F->cap); F->rv = realloc(F->rv, sizeof(double) * F->cap); }
    F->ri[F->nz] = j; F->rv[F->nz] = v; F->nz++; F->rp[F->nr] = F->nz;
}
static void fac_free(Fac *F) { free(F->rp); free(F->ri); free(F->rv); memset(F, 0, sizeof(*F)); }

static int uf_find(int *p, int a) { while (p[a] != a) { p[a] = p[p[a]]; a = p[a]; } return a; }

#define QC_DENSE_MAX 4000
#ifdef HAVE_AMD
#include "amd/amd.h"
#endif
/* (22 October 2026) a sparse block's factor: an up-looking Cholesky (row by row, each row's pattern from the elimination
 * tree) in AMD's order, so that F = L' P' stays as sparse as the block allows; a pivot below -1e-9 of the largest
 * diagonal: not convex (1), below 1e-12 of it: a zero column (the block's rank). The block's entries: local (a, c, v),
 * each pair once. F's rows: the columns of L, over the variables var[]. */
/* (25 October 2026, the SOCP developer) a block scaled to a unit diagonal before its factorization, sc_k = M_kk^-1/2 (1 where
 * M_kk <= 0), the values in place, the factor's rows divided by sc afterwards: the rank cut (a pivot below 1e-12 of the
 * largest diagonal) and the convexity test (below -1e-9 of it) were relative to the block's largest diagonal, while under a
 * scaling of the variables each pivot scales with its own diagonal. QCSCALE_2000 (QCQP: the variables scaled by 10^U(-3, 3),
 * the diagonals of one 5 x 5 block over 1e12): directions dropped as rank-deficient, the cones looser than the quadratic
 * rows, the SOCP's optimum 728.2 against the QCQP's 718.9, NUMERICAL DIFFICULTIES. The rank of a positive semidefinite
 * matrix does not change under a diagonal scaling, so its numerical test should not either */
static void qc_unit_diag(int s, int ne, const int *ea, const int *ec, double *ev, double *sc) {
    for (int k = 0; k < s; k++) sc[k] = 0.0;
    for (int e = 0; e < ne; e++) if (ea[e] == ec[e]) sc[ea[e]] += ev[e];
    for (int k = 0; k < s; k++) sc[k] = sc[k] > 0 ? 1.0 / sqrt(sc[k]) : 1.0;
    for (int e = 0; e < ne; e++) ev[e] *= sc[ea[e]] * sc[ec[e]];
}
static int sparse_factor(int s, int ne, const int *ea, const int *ec, const double *ev, const int *var, const double *sc, Fac *F, double *negpiv) {
    int *perm = xm(sizeof(int) * (s + 1)), *pinv = xm(sizeof(int) * (s + 1));
    {   /* AMD on the full pattern */
        int *cnt = xz(sizeof(int) * (s + 1)), *Ap = xz(sizeof(int) * (s + 1)), *Ai;
        for (int e = 0; e < ne; e++) if (ea[e] != ec[e]) { cnt[ea[e]]++; cnt[ec[e]]++; }
        for (int k = 0; k < s; k++) Ap[k + 1] = Ap[k] + cnt[k];
        Ai = xm(sizeof(int) * ((size_t)Ap[s] + 1)); memcpy(cnt, Ap, sizeof(int) * s);
        for (int e = 0; e < ne; e++) if (ea[e] != ec[e]) { Ai[cnt[ea[e]]++] = ec[e]; Ai[cnt[ec[e]]++] = ea[e]; }
        int ok = 0;
#ifdef HAVE_AMD
        double Control[AMD_CONTROL], Info[AMD_INFO]; amd_defaults(Control);
        const int st = amd_order(s, Ap, Ai, perm, Control, Info); ok = st == AMD_OK || st == AMD_OK_BUT_JUMBLED;
#endif
        if (!ok) for (int k = 0; k < s; k++) perm[k] = k;
        free(cnt); free(Ap); free(Ai);
    }
    for (int k = 0; k < s; k++) pinv[perm[k]] = k;
    /* C = the upper triangle of P M P' by columns */
    int *Cp = xz(sizeof(int) * (s + 2)), *Ci; double *Cx;
    for (int e = 0; e < ne; e++) { const int i = pinv[ea[e]], j = pinv[ec[e]]; Cp[(i > j ? i : j) + 1]++; }
    for (int k = 0; k < s; k++) Cp[k + 1] += Cp[k];
    Ci = xm(sizeof(int) * ((size_t)Cp[s] + 1)); Cx = xm(sizeof(double) * ((size_t)Cp[s] + 1));
    {   int *f = xm(sizeof(int) * (s + 1)); memcpy(f, Cp, sizeof(int) * s);
        for (int e = 0; e < ne; e++) { const int i = pinv[ea[e]], j = pinv[ec[e]], c = i > j ? i : j, r = i > j ? j : i; Ci[f[c]] = r; Cx[f[c]] = ev[e]; f[c]++; }
        free(f); }
    double d0 = 0; for (int k = 0; k < s; k++) for (int q = Cp[k]; q < Cp[k + 1]; q++) if (Ci[q] == k) d0 = fmax(d0, fabs(Cx[q]));
    /* the elimination tree of C */
    int *parent = xm(sizeof(int) * (s + 1)), *anc = xm(sizeof(int) * (s + 1));
    for (int k = 0; k < s; k++) {
        parent[k] = -1; anc[k] = -1;
        for (int q = Cp[k]; q < Cp[k + 1]; q++) {
            int i = Ci[q];
            while (i != -1 && i < k) { const int nx = anc[i]; anc[i] = k; if (nx == -1) parent[i] = k; i = nx; }
        }
    }
    /* L by columns, grown as the rows are computed; the diagonal first */
    int *Lcap = xm(sizeof(int) * (s + 1)), *Llen = xz(sizeof(int) * (s + 1)), **Lr = xm(sizeof(int *) * (s + 1)); double **Lv = xm(sizeof(double *) * (s + 1));
    for (int k = 0; k < s; k++) { Lcap[k] = 4; Lr[k] = xm(sizeof(int) * 4); Lv[k] = xm(sizeof(double) * 4); }
    double *x = xz(sizeof(double) * (s + 1)); int *mark = xm(sizeof(int) * (s + 1)), *stk = xm(sizeof(int) * (s + 1)), *pat = xm(sizeof(int) * (s + 1));
    for (int k = 0; k < s; k++) mark[k] = -1;
    int rc = 0; size_t nlr = 0; const double nlcap = s <= QC_DENSE_MAX ? (getenv("BRISK_QCSPFRAC") ? atof(getenv("BRISK_QCSPFRAC")) : 0.1) * 0.5 * (double)s * (s + 1) : HUGE_VAL;
    for (int k = 0; k < s && !rc; k++) {
        if ((double)nlr > nlcap) { rc = 3; break; }          /* (denser than the bound below: stopped at once) */
        int top = s; mark[k] = k;
        for (int q = Cp[k]; q < Cp[k + 1]; q++) {
            int i = Ci[q]; x[i] += Cx[q];
            int len = 0; for (; mark[i] != k; i = parent[i]) { stk[len++] = i; mark[i] = k; }
            while (len > 0) pat[--top] = stk[--len];
        }
        double d = x[k]; x[k] = 0;
        for (int t = top; t < s; t++) {
            const int j = pat[t]; const double ljj = Lv[j][0];
            const double lkj = ljj > 0 ? x[j] / ljj : 0.0; x[j] = 0;
            for (int q = 1; q < Llen[j]; q++) x[Lr[j][q]] -= Lv[j][q] * lkj;
            d -= lkj * lkj;
            if (lkj != 0) {
                if (Llen[j] == Lcap[j]) { Lcap[j] *= 2; Lr[j] = realloc(Lr[j], sizeof(int) * Lcap[j]); Lv[j] = realloc(Lv[j], sizeof(double) * Lcap[j]); }
                Lr[j][Llen[j]] = k; Lv[j][Llen[j]] = lkj; Llen[j]++; nlr++;
            }
        }
        if (d < -1e-9 * fmax(d0, 1e-300)) { rc = 1; *negpiv = fmin(*negpiv, d); break; }
        Lr[k][0] = k; Lv[k][0] = d > 1e-12 * d0 ? sqrt(d) : 0.0; Llen[k] = 1;
    }
    /* kept when it has at most a tenth of the dense factor's entries, or the block is beyond the dense limit (3: not
     * kept; RQCQP_3000's objective: 1.76 against 4.0 million entries, and with the dense factor the SOCP solver takes its
     * normal equations, 4.9 s of factorizations against 19.5 s of the full system's) */
    if (!rc && s <= QC_DENSE_MAX) { size_t nl = 0; for (int j = 0; j < s; j++) nl += Llen[j]; if ((double)nl > (getenv("BRISK_QCSPFRAC") ? atof(getenv("BRISK_QCSPFRAC")) : 0.1) * 0.5 * (double)s * (s + 1)) rc = 3; }
    /* F's rows: L's columns (the variables in the original order) */
    for (int j = 0; j < s && !rc; j++) {
        int any = 0; for (int q = 0; q < Llen[j]; q++) if (Lv[j][q] != 0) { any = 1; break; }
        if (!any) continue;
        fac_row(F);
        for (int q = 0; q < Llen[j]; q++) if (Lv[j][q] != 0) fac_add(F, var[perm[Lr[j][q]]], Lv[j][q] / sc[perm[Lr[j][q]]]);
    }
    for (int k = 0; k < s; k++) { free(Lr[k]); free(Lv[k]); }
    free(Lr); free(Lv); free(Lcap); free(Llen); free(x); free(mark); free(stk); free(pat);
    free(parent); free(anc); free(Cp); free(Ci); free(Cx); free(perm); free(pinv);
    return rc;
}

/* the factor of sg * Q (entries qi >= qj, values qv, ne of them; the full symmetric Q over n variables): 0, or 1 when
 * sg Q is not positive semidefinite, 2 when a connected block is too large for the dense factorization */
/* (24 October 2026, the SOCP developer) loc: a work array of n entries, all -1 on entry and on return (only the support's
 * entries are touched): set to -1 here for every constraint it was 4e9 writes on a QCQP with 44,700 quadratic rows over
 * 91,650 variables, 5.5 s of a 17 s run (LOSSFLOW_150) */
static int factor_psd(int n, int ne, const int *qi, const int *qj, const double *qv, double sg, Fac *F, double *negpiv, int *loc) {
    *negpiv = 0;
    if (ne == 0) return 0;
    /* the support and its connected blocks */
    (void)n;
    int ns = 0; int *sup = xm(sizeof(int) * (2 * (size_t)ne + 1));
    for (int e = 0; e < ne; e++) { if (loc[qi[e]] < 0) { loc[qi[e]] = ns; sup[ns++] = qi[e]; } if (loc[qj[e]] < 0) { loc[qj[e]] = ns; sup[ns++] = qj[e]; } }
    int *par = xm(sizeof(int) * (ns + 1)); for (int k = 0; k < ns; k++) par[k] = k;
    for (int e = 0; e < ne; e++) { const int a = uf_find(par, loc[qi[e]]), b = uf_find(par, loc[qj[e]]); if (a != b) par[a] = b; }
    int *bid = xm(sizeof(int) * (ns + 1)), nb = 0, *bsz;
    for (int k = 0; k < ns; k++) bid[k] = -1;
    int *root_b = xm(sizeof(int) * (ns + 1)); for (int k = 0; k < ns; k++) root_b[k] = -1;
    for (int k = 0; k < ns; k++) { const int r = uf_find(par, k); if (root_b[r] < 0) root_b[r] = nb++; bid[k] = root_b[r]; }
    bsz = xz(sizeof(int) * (nb + 1));
    int *bpos = xm(sizeof(int) * (ns + 1));          /* the position of a support variable within its block */
    for (int k = 0; k < ns; k++) bpos[k] = bsz[bid[k]]++;
    int rc = 0;
    /* the blocks' members, then each block dense */
    int *bst = xz(sizeof(int) * (nb + 2)); for (int b = 0; b < nb; b++) bst[b + 1] = bst[b] + bsz[b];
    int *mem = xm(sizeof(int) * (ns + 1)); for (int k = 0; k < ns; k++) mem[bst[bid[k]] + bpos[k]] = k;
    int *ebs = xz(sizeof(int) * (nb + 2));            /* the entries of each block */
    for (int e = 0; e < ne; e++) ebs[bid[loc[qi[e]]] + 1]++;
    for (int b = 0; b < nb; b++) ebs[b + 1] += ebs[b];
    int *eord = xm(sizeof(int) * (ne + 1)), *fill = xm(sizeof(int) * (nb + 1)); memcpy(fill, ebs, sizeof(int) * nb);
    for (int e = 0; e < ne; e++) eord[fill[bid[loc[qi[e]]]]++] = e;
    for (int b = 0; b < nb && !rc; b++) {
        const int s = bsz[b];
        /* a sparse block (at least 64 variables, fewer than a tenth of its pairs coupled) or one beyond the dense limit:
         * the sparse factor (BRISK_QCSPARSE=0: the dense path alone, blocks beyond 4,000 variables refused) */
        {   static int qsp = -1; if (qsp < 0) { const char *e = getenv("BRISK_QCSPARSE"); qsp = e ? atoi(e) : 1; }
            const int nb_e = ebs[b + 1] - ebs[b];
            if (qsp && s >= 64 && (s > QC_DENSE_MAX || (double)nb_e < 0.05 * (double)s * s)) {
                int *ea = xm(sizeof(int) * (nb_e + 1)), *ec = xm(sizeof(int) * (nb_e + 1)), *var = xm(sizeof(int) * (s + 1)); double *ev = xm(sizeof(double) * (nb_e + 1));
                for (int t = 0; t < nb_e; t++) { const int e = eord[ebs[b] + t]; ea[t] = bpos[loc[qi[e]]]; ec[t] = bpos[loc[qj[e]]]; ev[t] = sg * qv[e]; }
                for (int k = 0; k < s; k++) var[k] = sup[mem[bst[b] + k]];
                double *sc = xm(sizeof(double) * (s + 1)); qc_unit_diag(s, nb_e, ea, ec, ev, sc);
                rc = sparse_factor(s, nb_e, ea, ec, ev, var, sc, F, negpiv);
                free(ea); free(ec); free(ev); free(var); free(sc);
                if (rc != 3) continue;
                rc = 0;            /* (the dense path below) */
            }
        }
        if (s > QC_DENSE_MAX) {
            /* a diagonal block is a single variable; a large connected block needs a sparse factorization */
            rc = 2; break;
        }
        double *M = xz(sizeof(double) * (size_t)s * s);
        for (int t = ebs[b]; t < ebs[b + 1]; t++) {
            const int e = eord[t], a = bpos[loc[qi[e]]], c = bpos[loc[qj[e]]];
            M[a + (size_t)c * s] += sg * qv[e]; if (a != c) M[c + (size_t)a * s] += sg * qv[e];
        }
        /* (25 October 2026) the block scaled to a unit diagonal first (qc_unit_diag) */
        double *sc = xm(sizeof(double) * (s + 1));
        for (int k = 0; k < s; k++) { const double d = M[k + (size_t)k * s]; sc[k] = d > 0 ? 1.0 / sqrt(d) : 1.0; }
        for (int c = 0; c < s; c++) for (int k = 0; k < s; k++) M[k + (size_t)c * s] *= sc[k] * sc[c];
        /* Cholesky with diagonal pivoting: M = P L L' P' */
        int *perm = xm(sizeof(int) * (s + 1)); for (int k = 0; k < s; k++) perm[k] = k;
        double d0 = 0; for (int k = 0; k < s; k++) d0 = fmax(d0, fabs(M[k + (size_t)k * s]));
        int r = 0;
        /* (22 October 2026) a large block first by LAPACK's Cholesky (BLAS-3; the loops below took 4 s on a strictly convex
         * objective of 3,000 coupled variables): accepted when it succeeds with every pivot above 1e-12 of the largest
         * diagonal; otherwise the pivoted loops on the copy */
        if (s >= 64) {
            double *Mc = xm(sizeof(double) * (size_t)s * s); memcpy(Mc, M, sizeof(double) * (size_t)s * s);
            int info = 0; BL(dpotrf_)("L", &s, Mc, &s, &info);
            double pmin = HUGE_VAL; if (info == 0) for (int k = 0; k < s; k++) pmin = fmin(pmin, Mc[k + (size_t)k * s] * Mc[k + (size_t)k * s]);
            if (info == 0 && pmin > 1e-12 * d0) { memcpy(M, Mc, sizeof(double) * (size_t)s * s); r = s; }
            free(Mc);
        }
        for (; r < s; r++) {
            int p = r; for (int k = r + 1; k < s; k++) if (M[k + (size_t)k * s] > M[p + (size_t)p * s]) p = k;
            const double d = M[p + (size_t)p * s];
            if (!(d > 1e-12 * d0)) break;
            if (p != r) {           /* symmetric swap of r and p */
                for (int k = 0; k < s; k++) { double t = M[r + (size_t)k * s]; M[r + (size_t)k * s] = M[p + (size_t)k * s]; M[p + (size_t)k * s] = t; }
                for (int k = 0; k < s; k++) { double t = M[k + (size_t)r * s]; M[k + (size_t)r * s] = M[k + (size_t)p * s]; M[k + (size_t)p * s] = t; }
                const int t = perm[r]; perm[r] = perm[p]; perm[p] = t;
            }
            const double l = sqrt(d);
            M[r + (size_t)r * s] = l;
            for (int k = r + 1; k < s; k++) M[k + (size_t)r * s] /= l;
            /* the whole trailing block updated (both triangles): a later symmetric swap reads both */
            for (int c = r + 1; c < s; c++) { const double lc = M[c + (size_t)r * s]; if (lc == 0) continue; for (int k = r + 1; k < s; k++) M[k + (size_t)c * s] -= M[k + (size_t)r * s] * lc; }
        }
        /* the remaining Schur complement must be (numerically) zero and not negative */
        for (int k = r; k < s; k++) {
            const double d = M[k + (size_t)k * s];
            if (d < -1e-9 * fmax(d0, 1e-300)) { rc = 1; *negpiv = fmin(*negpiv, d); }
        }
        /* rows of F: column k of L (k < r) over the block's variables, perm mapped */
        for (int k = 0; k < r && !rc; k++) {
            fac_row(F);
            fac_add(F, sup[mem[bst[b] + perm[k]]], M[k + (size_t)k * s] / sc[perm[k]]);
            for (int i = k + 1; i < s; i++) if (M[i + (size_t)k * s] != 0) fac_add(F, sup[mem[bst[b] + perm[i]]], M[i + (size_t)k * s] / sc[perm[i]]);
        }
        free(perm); free(M); free(sc);
    }
    for (int k = 0; k < ns; k++) loc[sup[k]] = -1;
    free(sup); free(par); free(bid); free(root_b); free(bsz); free(bpos); free(bst); free(mem); free(ebs); free(eord); free(fill);
    return rc;
}

typedef struct SChol SChol;
SChol *schol_analyze_adj(int m, int *deg, int **nbr, size_t fillcap);
void   schol_free(SChol *S);
double *schol_values(SChol *S);
size_t schol_offset(const SChol *S, size_t r, size_t c);
void   schol_set_signs(SChol *S, const signed char *sgn_orig);
void   schol_set_dynpiv(SChol *S, double p);
int    schol_factor(SChol *S, double shift);
void   schol_solve(const SChol *S, double *b, double *work);
size_t schol_pansz(const SChol *S);
int    schol_set_amd(int mode);
void   qp_hmul(const QpProb *P, const double *x, double *y);

typedef struct { int i, j; double v; } PTrip;
static int ptrip_cmp(const void *a, const void *b) {
    const PTrip *x = a, *y = b;
    if (x->j != y->j) return x->j < y->j ? -1 : 1;
    return (x->i > y->i) - (x->i < y->i);
}
static void ptrip_add(PTrip **T, long *nt, long *cap, int i, int j, double v) {
    if (*nt == *cap) { *cap = *cap ? 2 * *cap : 4096; *T = realloc(*T, sizeof(PTrip) * *cap); }
    (*T)[*nt].i = i; (*T)[*nt].j = j; (*T)[*nt].v = v; (*nt)++;
}
static double maxerr(const QpProb *P, const double *x, const double *y, const double *z) {
    double a, b, c; qp_errors(P, x, y, z, &a, &b, &c); return fmax(a, fmax(b, c));
}

/* (22 October 2026) polishing: Newton steps on the QCQP's own optimality conditions, on the active set read off the SOCP
 * solver's point. Through the cones, a quadratic row's multiplier y_k and the cone's dual part v meet the QCQP's
 * condition F'v = y_k Q_k x only up to the square root of the cone's complementarity gap (two vectors in a cone with
 * inner product eps lie at an angle of order sqrt(eps)), so the errors on the problem as read stall near 1e-5 where the
 * SOCP is solved to 1e-9 (ELLIPS_50: the SOCP 2e-9, the QCQP's dual error 1e-5; Gurobi and MOSEK have the same
 * objective to 1e-10). With the active rows as equalities and the active bounds fixed, the conditions are a square
 * system of equations, (H - sum_k y_k Q_k) dx - J'dy = -r, J dx = t - g(x), whose Newton method converges
 * quadratically from that point (strict complementarity). The system is solved as a quasi-definite one, regularized
 * by delta and refined against the unregularized matrix; a step is kept only when the largest error on the problem as
 * read falls. A row or a bound is active when its multiplier, relative to the largest, exceeds its slack, relative to
 * its side. */
/* (24 October 2026, the SOCP developer) a sequential quadratic programming step when the Newton steps above do not reach
 * the tolerance: the QP in dx of the QCQP's model at x, min 1/2 dx'W dx + (Hx + c)'dx over rl <= g(x) + J dx <= ru and the
 * bounds, W = H - sum_k y_k Q_k the Lagrangian's Hessian (positive semidefinite: y_k <= 0 on a row at its upper side),
 * solved by the QP solver; x + dx and the QP's multipliers are the new point. Unlike the Newton step on the active set it
 * needs no independent active rows and keeps the multipliers' signs: degenerate QCQPs (sensor localization, lossy flows:
 * dependent active rows, ISSUES 277), where the errors on the problem as read are the square root of the cones' gap.
 * Kept while the largest error on the problem as read falls (at most three steps). */
static void qcqp_sqp_polish(const QpProb *P, double *x, double *y, double *z, double tol, double *ebest, int verbose) {
    const int n = P->n, m = P->m; const double t0 = now(), e_in = *ebest;
    double *xc = xm(sizeof(double) * (n + 1)), *yc = xm(sizeof(double) * (m + 1)), *zc = xm(sizeof(double) * (n + 1));
    memcpy(xc, x, sizeof(double) * n); memcpy(yc, y, sizeof(double) * m); memcpy(zc, z, sizeof(double) * n);
    int steps = 0, tried = 0;
    for (int it = 0; it < 3; it++) {
        tried++;
        /* W (lower triangle) */
        PTrip *T = NULL; long nt = 0, cap = 0;
        if (P->Hp) for (int j = 0; j < n; j++) for (int p = P->Hp[j]; p < P->Hp[j + 1]; p++) ptrip_add(&T, &nt, &cap, P->Hi[p], j, P->Hx[p]);
        for (int k = 0; k < P->nqc; k++) { const double yk = yc[P->qrow[k]]; if (yk == 0.0) continue;
            for (int e2 = P->qs[k]; e2 < P->qs[k + 1]; e2++) ptrip_add(&T, &nt, &cap, P->qi[e2], P->qj[e2], -yk * P->qv[e2]); }
        /* a proximal term rho I, rho = 1e-6 (1 + max_j W_jj): the QP strictly convex, so the step is unique where W is
         * singular (a linear objective whose quadratic rows do not reach every direction, QPLIB_2482: without it the
         * step's error 7.1e-8 -> 2.9e-7, with it 2.6e-9 in three steps; the degenerate localization problems keep their
         * quadratic convergence, 1.3e-5 -> 1.8e-13). BRISK_QCSQPRHO=r: r instead of 1e-6 */
        { static double rho = -1; if (rho < 0) { const char *e_ = getenv("BRISK_QCSQPRHO"); rho = e_ ? atof(e_) : 1e-6; }
          double wmax = 0; for (long q = 0; q < nt; q++) if (T[q].i == T[q].j) wmax = fmax(wmax, fabs(T[q].v));
          /* (25 October 2026, the SOCP developer) the term from each column's own diagonal, rho (1 + |W_jj|), not from the
           * largest one: with the quadratic rows' multipliers over eight orders of magnitude (QCFIXQ_2000, QCQP: 1 / (2 d_i),
           * d_i = 10^U(-4, 4)) rho (1 + max W) is 5e-3 on columns of curvature 5e-5, so the steps in those directions are
           * damped a hundredfold and the errors stalled after the first step (1.1e-2 -> 1.6e-8 -> 1.2e-8 -> 9.7e-9; on its
           * twin QCFIXQTWIN_2000 short of the tolerance, so the cones were solved a second time); with each column's own
           * term one step to 2.3e-11. The 83 QCQPs: no status or error worse,
           * QCSCALE_20000 109 -> 31 and QCFIXQTWIN_2000 175 -> 80 iterations (no second solve), QPLIB_2482's 2.6e-9 kept
           * (bench/socp_rotfix/res3_qcqp.txt). BRISK_QCSQPRCOL=0: the largest diagonal */
          static int rcol = -1; if (rcol < 0) { const char *e_ = getenv("BRISK_QCSQPRCOL"); rcol = e_ ? atoi(e_) : 1; }
          if (rcol) {
              double *wd = xz(sizeof(double) * (n + 1)); for (long q = 0; q < nt; q++) if (T[q].i == T[q].j) wd[T[q].j] += T[q].v;
              if (rho > 0) for (int j = 0; j < n; j++) ptrip_add(&T, &nt, &cap, j, j, rho * (1 + fabs(wd[j])));
              free(wd);
          } else
          if (rho > 0) for (int j = 0; j < n; j++) ptrip_add(&T, &nt, &cap, j, j, rho * (1 + wmax)); }
        qsort(T, nt, sizeof(PTrip), ptrip_cmp);
        { long u = 0; for (long q = 0; q < nt; q++) { if (u > 0 && T[u - 1].i == T[q].i && T[u - 1].j == T[q].j) T[u - 1].v += T[q].v; else T[u++] = T[q]; } nt = u; }
        /* J (rows by columns) and g(x) */
        PTrip *G = NULL; long ng = 0, cg = 0; double *gx = xz(sizeof(double) * (m + 1)), *qx = xz(sizeof(double) * (n + 1));
        for (int j = 0; j < n; j++) for (int p = P->Ap[j]; p < P->Ap[j + 1]; p++) { ptrip_add(&G, &ng, &cg, P->Ai[p], j, P->Ax[p]); gx[P->Ai[p]] += P->Ax[p] * xc[j]; }
        for (int k = 0; k < P->nqc; k++) {
            const int i = P->qrow[k];
            for (int e2 = P->qs[k]; e2 < P->qs[k + 1]; e2++) { qx[P->qi[e2]] = 0; qx[P->qj[e2]] = 0; }
            for (int e2 = P->qs[k]; e2 < P->qs[k + 1]; e2++) { const int a = P->qi[e2], b = P->qj[e2]; const double v = P->qv[e2];
                gx[i] += (a == b ? 0.5 : 1.0) * v * xc[a] * xc[b]; qx[a] += v * xc[b]; if (a != b) qx[b] += v * xc[a]; }
            for (int e2 = P->qs[k]; e2 < P->qs[k + 1]; e2++) {
                const int cols[2] = { P->qi[e2], P->qj[e2] };
                for (int t = 0; t < (cols[0] == cols[1] ? 1 : 2); t++) { const int a = cols[t]; if (qx[a] != qx[a]) continue; ptrip_add(&G, &ng, &cg, i, a, qx[a]); qx[a] = NAN; }
            }
        }
        qsort(G, ng, sizeof(PTrip), ptrip_cmp);
        { long u = 0; for (long q = 0; q < ng; q++) { if (u > 0 && G[u - 1].i == G[q].i && G[u - 1].j == G[q].j) G[u - 1].v += G[q].v; else G[u++] = G[q]; } ng = u; }
        QpProb Q; memset(&Q, 0, sizeof Q); Q.n = n; Q.m = m; snprintf(Q.name, sizeof Q.name, "%.59s-sqp", P->name);
        Q.Hp = xz(sizeof(int) * (n + 1)); Q.Hi = xm(sizeof(int) * (nt + 1)); Q.Hx = xm(sizeof(double) * (nt + 1));
        { long q = 0; for (int j = 0; j < n; j++) { Q.Hp[j] = (int)q; while (q < nt && T[q].j == j) { Q.Hi[q] = T[q].i; Q.Hx[q] = T[q].v; q++; } } Q.Hp[n] = (int)q; }
        Q.Ap = xz(sizeof(int) * (n + 1)); Q.Ai = xm(sizeof(int) * (ng + 1)); Q.Ax = xm(sizeof(double) * (ng + 1));
        { long q = 0; for (int j = 0; j < n; j++) { Q.Ap[j] = (int)q; while (q < ng && G[q].j == j) { Q.Ai[q] = G[q].i; Q.Ax[q] = G[q].v; q++; } } Q.Ap[n] = (int)q; }
        Q.c = xm(sizeof(double) * (n + 1)); qp_hmul(P, xc, Q.c); for (int j = 0; j < n; j++) Q.c[j] += P->c[j];
        Q.rl = xm(sizeof(double) * (m + 1)); Q.ru = xm(sizeof(double) * (m + 1)); Q.xl = xm(sizeof(double) * (n + 1)); Q.xu = xm(sizeof(double) * (n + 1));
        for (int i = 0; i < m; i++) { Q.rl[i] = fin(P->rl[i]) ? P->rl[i] - gx[i] : -QP_INF; Q.ru[i] = fin(P->ru[i]) ? P->ru[i] - gx[i] : QP_INF; }
        for (int j = 0; j < n; j++) { Q.xl[j] = fin(P->xl[j]) ? P->xl[j] - xc[j] : -QP_INF; Q.xu[j] = fin(P->xu[j]) ? P->xu[j] - xc[j] : QP_INF; }
        QpOpts o; qp_default_opts(&o); o.verbose = 0; o.tol = fmin(1e-10, 0.01 * tol); o.method = QP_METHOD_IPM;
        QpRes R; memset(&R, 0, sizeof R);
        const int st = qp_solve(&Q, &o, &R);
        double e1 = HUGE_VAL;
        if ((st == QP_OPTIMAL || st == QP_REDUCED) && R.x && R.y && R.z) {
            for (int j = 0; j < n; j++) xc[j] += R.x[j];
            memcpy(yc, R.y, sizeof(double) * m); memcpy(zc, R.z, sizeof(double) * n);
            e1 = maxerr(P, xc, yc, zc);
        }
        if (verbose) qp_printf("  a sequential quadratic programming step %d: the QP %s in %d iterations, the largest error %.1e -> %.1e\n", it + 1, qp_status_name(st), R.iter, *ebest, e1);
        qp_res_free(&R);
        free(Q.Hp); free(Q.Hi); free(Q.Hx); free(Q.Ap); free(Q.Ai); free(Q.Ax); free(Q.c); free(Q.rl); free(Q.ru); free(Q.xl); free(Q.xu);
        free(T); free(G); free(gx); free(qx);
        if (!(e1 < *ebest)) break;
        *ebest = e1; steps = it + 1; memcpy(x, xc, sizeof(double) * n); memcpy(y, yc, sizeof(double) * m); memcpy(z, zc, sizeof(double) * n);
        if (e1 <= 0.1 * tol) break;
    }
    if (verbose) qp_printf("sequential quadratic programming steps on the QCQP: %d of %d kept, the largest error %.1e -> %.1e, %.2f s\n", steps, tried, e_in, *ebest, now() - t0);
    free(xc); free(yc); free(zc);
}

static void qcqp_polish(const QpProb *P, double *x, double *y, double *z, double tol, int verbose) {
    const int n = P->n, m = P->m;
    double ebest = maxerr(P, x, y, z);
    const double e_in = ebest, tp0 = now();
    if (!(ebest > 1e-15)) return;
    int *qk = xm(sizeof(int) * (m + 1)); for (int i = 0; i < m; i++) qk[i] = -1;
    for (int k = 0; k < P->nqc; k++) qk[P->qrow[k]] = k;
    double *ax = xz(sizeof(double) * (m + 1));
    for (int j = 0; j < n; j++) for (int p = P->Ap[j]; p < P->Ap[j + 1]; p++) ax[P->Ai[p]] += P->Ax[p] * x[j];
    for (int k = 0; k < P->nqc; k++) for (int e = P->qs[k]; e < P->qs[k + 1]; e++) {
        const int a = P->qi[e], b = P->qj[e]; ax[P->qrow[k]] += (a == b ? 0.5 : 1.0) * P->qv[e] * x[a] * x[b];
    }
    double ymax = 0, zmax = 0;
    for (int i = 0; i < m; i++) ymax = fmax(ymax, fabs(y[i]));
    for (int j = 0; j < n; j++) zmax = fmax(zmax, fabs(z[j]));
    /* the active set: rows 0 inactive, 1 at rl, 2 at ru, 3 equality; columns 0 free, 1 at xl, 2 at xu, 3 fixed */
    signed char *ra = xz(m + 1), *ca = xz(n + 1);
    int nact = 0;
    for (int i = 0; i < m; i++) {
        if (fin(P->rl[i]) && P->rl[i] == P->ru[i]) ra[i] = 3;
        else if (y[i] > 0 && fin(P->rl[i]) && y[i] / (1 + ymax) > (ax[i] - P->rl[i]) / (1 + fabs(P->rl[i]))) ra[i] = 1;
        else if (y[i] < 0 && fin(P->ru[i]) && -y[i] / (1 + ymax) > (P->ru[i] - ax[i]) / (1 + fabs(P->ru[i]))) ra[i] = 2;
        nact += ra[i] != 0;
    }
    int nfree = 0;
    for (int j = 0; j < n; j++) {
        if (fin(P->xl[j]) && P->xl[j] == P->xu[j]) ca[j] = 3;
        else if (z[j] > 0 && fin(P->xl[j]) && z[j] / (1 + zmax) > (x[j] - P->xl[j]) / (1 + fabs(P->xl[j]))) ca[j] = 1;
        else if (z[j] < 0 && fin(P->xu[j]) && -z[j] / (1 + zmax) > (P->xu[j] - x[j]) / (1 + fabs(P->xu[j]))) ca[j] = 2;
        nfree += ca[j] == 0;
    }
    double *x1 = xm(sizeof(double) * (n + 1)), *y1 = xm(sizeof(double) * (m + 1)), *z1 = xm(sizeof(double) * (n + 1));
    memcpy(x1, x, sizeof(double) * n); memcpy(y1, y, sizeof(double) * m); memcpy(z1, z, sizeof(double) * n);
    for (int j = 0; j < n; j++) { if (ca[j] == 1 || ca[j] == 3) x1[j] = P->xl[j]; else if (ca[j] == 2) x1[j] = P->xu[j]; if (!ca[j]) z1[j] = 0; }
    for (int i = 0; i < m; i++) if (!ra[i]) y1[i] = 0;
    int *cid = xm(sizeof(int) * (n + 1)), *rid = xm(sizeof(int) * (m + 1));
    const size_t NM = (size_t)n + m + 1;
    double *g = xm(sizeof(double) * (m + 1)), *r = xm(sizeof(double) * (n + 1)), *qx = xm(sizeof(double) * (n + 1));
    double *rhs = xm(sizeof(double) * NM), *sol = xm(sizeof(double) * NM), *res = xm(sizeof(double) * NM), *e = xm(sizeof(double) * NM);
    double *work = xm(sizeof(double) * (2 * NM + 2));
    double *xb = xm(sizeof(double) * (n + 1)), *yb = xm(sizeof(double) * (m + 1)), *zb = xm(sizeof(double) * (n + 1));
    int steps = 0, nadded = 0, rounds = 0;
    int *qmk = xm(sizeof(int) * (n + 1)), stamp = 0; for (int j = 0; j < n; j++) qmk[j] = -1;
    /* the analysis kept while the active set (aver) and so the pattern stay */
    SChol *K = NULL; signed char *sgn = NULL; int aver = 0, Kaver = -1; long Knt = -1;
    for (int it = 0; it < 6; it++) {
        /* the unknowns: the free columns, then the active rows */
        nfree = 0; nact = 0;
        for (int j = 0; j < n; j++) cid[j] = ca[j] ? -1 : nfree++;
        for (int i = 0; i < m; i++) rid[i] = ra[i] ? nfree + nact++ : -1;
        const int N = nfree + nact;
        if (N == 0) break;
        memcpy(xb, x1, sizeof(double) * n); memcpy(yb, y1, sizeof(double) * m); memcpy(zb, z1, sizeof(double) * n);
        /* g(x1), r = Hx1 + c - A'y1 - sum_k y1_k Q_k x1, and the matrix's entries (lower triangle, by columns) */
        PTrip *T = NULL; long nt = 0, cap = 0;
        memset(g, 0, sizeof(double) * m);
        for (int j = 0; j < n; j++) for (int p = P->Ap[j]; p < P->Ap[j + 1]; p++) g[P->Ai[p]] += P->Ax[p] * x1[j];
        qp_hmul(P, x1, r);
        for (int j = 0; j < n; j++) {
            r[j] += P->c[j];
            for (int p = P->Ap[j]; p < P->Ap[j + 1]; p++) {
                const int i = P->Ai[p]; r[j] -= P->Ax[p] * y1[i];
                if (ra[i] && cid[j] >= 0 && qk[i] < 0) ptrip_add(&T, &nt, &cap, rid[i], cid[j], P->Ax[p]);
            }
        }
        double wmax = 0;
        if (P->Hp) for (int j = 0; j < n; j++) if (cid[j] >= 0) for (int p = P->Hp[j]; p < P->Hp[j + 1]; p++) if (cid[P->Hi[p]] >= 0) {
            ptrip_add(&T, &nt, &cap, cid[P->Hi[p]], cid[j], -P->Hx[p]); if (P->Hi[p] == j) wmax = fmax(wmax, fabs(P->Hx[p]));
        }
        for (int k = 0; k < P->nqc; k++) {
            const int i = P->qrow[k];
            for (int e2 = P->qs[k]; e2 < P->qs[k + 1]; e2++) { qx[P->qi[e2]] = 0; qx[P->qj[e2]] = 0; }
            for (int e2 = P->qs[k]; e2 < P->qs[k + 1]; e2++) {
                const int a = P->qi[e2], b = P->qj[e2]; const double v = P->qv[e2];
                g[i] += (a == b ? 0.5 : 1.0) * v * x1[a] * x1[b];
                qx[a] += v * x1[b]; if (a != b) qx[b] += v * x1[a];
                if (ra[i] && cid[a] >= 0 && cid[b] >= 0) { ptrip_add(&T, &nt, &cap, cid[a], cid[b], y1[i] * v); if (a == b) wmax = fmax(wmax, fabs(y1[i] * v)); }
            }
            /* the row's gradient a_i + Q_k x1: its linear part from A, the quadratic part here (marked once per column) */
            stamp++;
            for (int e2 = P->qs[k]; e2 < P->qs[k + 1]; e2++) {
                const int cols[2] = { P->qi[e2], P->qj[e2] };
                for (int t = 0; t < (cols[0] == cols[1] ? 1 : 2); t++) {
                    const int a = cols[t]; if (qmk[a] == stamp) continue;
                    qmk[a] = stamp;
                    r[a] -= y1[i] * qx[a];
                    if (ra[i] && cid[a] >= 0) ptrip_add(&T, &nt, &cap, rid[i], cid[a], qx[a]);   /* (also when 0: the pattern stays) */
                }
            }
        }
        /* the quadratic rows' linear parts in J */
        for (int j = 0; j < n; j++) if (cid[j] >= 0) for (int p = P->Ap[j]; p < P->Ap[j + 1]; p++) { const int i = P->Ai[p]; if (ra[i] && qk[i] >= 0) ptrip_add(&T, &nt, &cap, rid[i], cid[j], P->Ax[p]); }
        const double dlt = 1e-9 * (1 + wmax);
        for (int c = 0; c < nfree; c++) ptrip_add(&T, &nt, &cap, c, c, 0.0);
        for (int c = nfree; c < N; c++) ptrip_add(&T, &nt, &cap, c, c, 0.0);
        /* merge duplicates; T holds (i >= j) */
        for (long q = 0; q < nt; q++) if (T[q].i < T[q].j) { const int t = T[q].i; T[q].i = T[q].j; T[q].j = t; }
        qsort(T, nt, sizeof(PTrip), ptrip_cmp);
        long u = 0; for (long q = 0; q < nt; q++) { if (u > 0 && T[u - 1].i == T[q].i && T[u - 1].j == T[q].j) T[u - 1].v += T[q].v; else T[u++] = T[q]; }
        nt = u;
        /* the right-hand side: (r_F, t - g) */
        for (int j = 0; j < n; j++) if (cid[j] >= 0) rhs[cid[j]] = r[j];
        for (int i = 0; i < m; i++) if (ra[i]) rhs[rid[i]] = (ra[i] == 2 ? P->ru[i] : P->rl[i]) - g[i];
        double e0 = 0; for (int k = 0; k < N; k++) e0 = fmax(e0, fabs(rhs[k]));
        /* the factorization */
        int *deg = xz(sizeof(int) * (N + 1));
        for (long q = 0; q < nt; q++) if (T[q].i != T[q].j) { deg[T[q].i]++; deg[T[q].j]++; }
        int **nbr = xm(sizeof(int *) * (N + 1)); long tot = 0; for (int k = 0; k < N; k++) tot += deg[k];
        int *pool = xm(sizeof(int) * (tot + 1)); { long o2 = 0; for (int k = 0; k < N; k++) { nbr[k] = pool + o2; o2 += deg[k]; deg[k] = 0; } }
        for (long q = 0; q < nt; q++) if (T[q].i != T[q].j) { nbr[T[q].i][deg[T[q].i]++] = T[q].j; nbr[T[q].j][deg[T[q].j]++] = T[q].i; }
        if (K && (Kaver != aver || Knt != nt)) { schol_free(K); K = NULL; free(sgn); sgn = NULL; }
        if (!K) {
            /* AMD alone: the own orderings cost about one factorization and pay over many (ELLIPS_3000: 3.37e9 against
             * 3.32e9 flops), polishing makes two to six */
            const int amd_old = schol_set_amd(2);
            K = schol_analyze_adj(N, deg, nbr, (size_t)1 << 31);
            schol_set_amd(amd_old);
            if (K) { sgn = xm(N + 1); for (int k = 0; k < N; k++) sgn[k] = k < nfree ? -1 : 1; schol_set_signs(K, sgn); }
            Kaver = aver; Knt = nt;
        }
        free(deg); free(nbr); free(pool);
        if (!K) { free(T); break; }
        schol_set_dynpiv(K, 1e-8 * dlt);
        double *v = schol_values(K);
        memset(v, 0, sizeof(double) * (schol_pansz(K) ? schol_pansz(K) : 1));
        for (long q = 0; q < nt; q++) v[schol_offset(K, T[q].i, T[q].j)] += T[q].v + (T[q].i == T[q].j ? (T[q].i < nfree ? -dlt : dlt) : 0);
        /* the diagonal of the x block: -(H - sum y Q)_jj is the merged entry; the products below use T without delta */
        const int frc = schol_factor(K, 0.0);
        if (frc) { free(T); break; }
        /* solve with refinement against the unregularized matrix (T, symmetric, lower triangle) */
        memset(sol, 0, sizeof(double) * N); memcpy(res, rhs, sizeof(double) * N);
        double rn_prev = HUGE_VAL, rn = e0; int nref = 0;
        for (int ref = 0; ref < 30; ref++) {
            rn = 0; for (int k = 0; k < N; k++) rn = fmax(rn, fabs(res[k]));
            if (rn <= 1e-15 * (1 + e0) || rn >= 0.5 * rn_prev) break;
            rn_prev = rn; nref++;
            memcpy(e, res, sizeof(double) * N); schol_solve(K, e, work);
            for (int k = 0; k < N; k++) sol[k] += e[k];
            memcpy(res, rhs, sizeof(double) * N);
            for (long q = 0; q < nt; q++) { res[T[q].i] -= T[q].v * sol[T[q].j]; if (T[q].i != T[q].j) res[T[q].j] -= T[q].v * sol[T[q].i]; }
        }
        free(T);
        if (getenv("BRISK_QCPDBG")) { int bs = 0, br = 0, bh = 0; for (int k = 0; k < N; k++) { bs += !isfinite(sol[k]); br += !isfinite(res[k]); bh += !isfinite(rhs[k]); }
            qp_printf("    [polish dbg] N %d nfree %d: non-finite rhs %d sol %d res %d; rn %.2e e0 %.2e nref %d dlt %.1e\n", N, nfree, bh, bs, br, rn, e0, nref, dlt); }
        /* (22 October 2026) a step from a solve that did not converge is not used, neither to move the point nor to
         * change the active set: on QPLIB_2468 (QCQP, a linear objective) the system on the active set read off the
         * SOCP solver's point is singular where the quadratic rows do not reach, its solve stopped at 3 % of the
         * right-hand side, the step violated 1,876 inactive rows by up to 0.5, they were added at a point where their
         * slack is 0.5, and the factorization of the inconsistent system gave NaN (returned as OPTIMAL until qp_errors
         * made non-finite points' errors infinite). The polishing keeps the SOCP solver's point then */
        {   int fin_ok = 1; for (int k = 0; k < N && fin_ok; k++) fin_ok = isfinite(sol[k]);
            if (!fin_ok || !(rn <= 1e-3 * e0)) {
                if (verbose >= 2) { if (fin_ok) qp_printf("  polishing step %d: the system on this active set is not solved (residual %.1e of %.1e): stopped\n", it + 1, rn, e0);
                                    else qp_printf("  polishing step %d: the system on this active set is singular (its solve is not finite): stopped\n", it + 1); }
                break;
            } }
        /* the step: the K's rows read -(W) dx + J'dy = r and J dx = t - g */
        for (int j = 0; j < n; j++) if (cid[j] >= 0) x1[j] += sol[cid[j]];
        for (int i = 0; i < m; i++) if (ra[i]) y1[i] += sol[rid[i]];
        /* the multipliers' signs, and the fixed columns' z from stationarity */
        for (int i = 0; i < m; i++) { if (ra[i] == 1 && y1[i] < 0) y1[i] = 0; if (ra[i] == 2 && y1[i] > 0) y1[i] = 0; }
        qp_hmul(P, x1, r);
        for (int j = 0; j < n; j++) { r[j] += P->c[j]; for (int p = P->Ap[j]; p < P->Ap[j + 1]; p++) r[j] -= P->Ax[p] * y1[P->Ai[p]]; }
        for (int k = 0; k < P->nqc; k++) {
            const int i = P->qrow[k]; if (y1[i] == 0) continue;
            for (int e2 = P->qs[k]; e2 < P->qs[k + 1]; e2++) { const int a = P->qi[e2], b = P->qj[e2]; const double vv = P->qv[e2]; r[a] -= y1[i] * vv * x1[b]; if (a != b) r[b] -= y1[i] * vv * x1[a]; }
        }
        for (int j = 0; j < n; j++) {
            z1[j] = ca[j] ? r[j] : 0;
            if (ca[j] == 1 && z1[j] < 0) z1[j] = 0;
            if (ca[j] == 2 && z1[j] > 0) z1[j] = 0;
        }
        /* an inactive row or bound violated by the step belongs to the active set: add it and redo the step (at most
         * three rounds an iteration) */
        {   int nv = 0;
            double *g2 = xz(sizeof(double) * (m + 1));
            for (int j = 0; j < n; j++) for (int p = P->Ap[j]; p < P->Ap[j + 1]; p++) g2[P->Ai[p]] += P->Ax[p] * x1[j];
            for (int k = 0; k < P->nqc; k++) for (int e2 = P->qs[k]; e2 < P->qs[k + 1]; e2++) { const int a = P->qi[e2], b = P->qj[e2]; g2[P->qrow[k]] += (a == b ? 0.5 : 1.0) * P->qv[e2] * x1[a] * x1[b]; }
            if (rounds < 3) {
                for (int i = 0; i < m; i++) if (!ra[i]) {
                    if (fin(P->rl[i]) && g2[i] < P->rl[i] - 1e-9 * (1 + fabs(P->rl[i]))) { ra[i] = 1; nv++; }
                    else if (fin(P->ru[i]) && g2[i] > P->ru[i] + 1e-9 * (1 + fabs(P->ru[i]))) { ra[i] = 2; nv++; }
                }
                for (int j = 0; j < n; j++) if (!ca[j]) {
                    if (fin(P->xl[j]) && x1[j] < P->xl[j] - 1e-9 * (1 + fabs(P->xl[j]))) { ca[j] = 1; nv++; }
                    else if (fin(P->xu[j]) && x1[j] > P->xu[j] + 1e-9 * (1 + fabs(P->xu[j]))) { ca[j] = 2; nv++; }
                }
            }
            free(g2);
            if (nv) {
                if (verbose >= 2) qp_printf("  polishing step %d: %d inactive rows or bounds violated, added to the active set\n", it + 1, nv);
                memcpy(x1, xb, sizeof(double) * n); memcpy(y1, yb, sizeof(double) * m); memcpy(z1, zb, sizeof(double) * n);
                for (int j = 0; j < n; j++) { if (ca[j] == 1) x1[j] = P->xl[j]; else if (ca[j] == 2) x1[j] = P->xu[j]; }
                nadded += nv; rounds++; aver++; it--; continue;
            }
            rounds = 0;
        }
        const double e1 = maxerr(P, x1, y1, z1);
        if (verbose >= 2) { double sm = 0; for (int k = 0; k < N; k++) sm = fmax(sm, fabs(sol[k]));
            double sx = 0; for (int j = 0; j < n; j++) if (cid[j] >= 0) sx = fmax(sx, fabs(sol[cid[j]]));
            int vb = 0, vr = 0; for (int j = 0; j < n; j++) if (!ca[j] && ((fin(P->xl[j]) && x1[j] < P->xl[j] - 1e-9 * (1 + fabs(P->xl[j]))) || (fin(P->xu[j]) && x1[j] > P->xu[j] + 1e-9 * (1 + fabs(P->xu[j]))))) vb++;
            { double *g2 = xz(sizeof(double) * (m + 1)); for (int j = 0; j < n; j++) for (int p = P->Ap[j]; p < P->Ap[j + 1]; p++) g2[P->Ai[p]] += P->Ax[p] * x1[j];
              for (int k = 0; k < P->nqc; k++) for (int e2 = P->qs[k]; e2 < P->qs[k + 1]; e2++) { const int a = P->qi[e2], b = P->qj[e2]; g2[P->qrow[k]] += (a == b ? 0.5 : 1.0) * P->qv[e2] * x1[a] * x1[b]; }
              for (int i = 0; i < m; i++) if (!ra[i] && ((fin(P->rl[i]) && g2[i] < P->rl[i] - 1e-9 * (1 + fabs(P->rl[i]))) || (fin(P->ru[i]) && g2[i] > P->ru[i] + 1e-9 * (1 + fabs(P->ru[i]))))) vr++;
              free(g2); }
            qp_printf("    (inactive bounds violated %d, inactive rows violated %d)\n", vb, vr);
            double a1, a2, a3; qp_errors(P, x1, y1, z1, &a1, &a2, &a3); qp_printf("    (primal %.1e, dual %.1e, gap %.1e; |dx| %.1e)\n", a1, a2, a3, sx);
            qp_printf("  polishing step %d: %d free variables, %d active rows; the largest error %.1e -> %.1e (the system's residual %.1e -> %.1e in %d refinements, |step| %.1e)\n", it + 1, nfree, nact, ebest, e1, e0, rn, nref, sm); }
        /* the best point kept; a step that raises the error may still lead on (a quadratic row's curvature after a
         * long first step, QPLIB_2626), so the iteration goes on unless the error is far above the start's */
        if (!(e1 <= 1e3 * e_in)) break;
        if (e1 < ebest) { ebest = e1; steps = it + 1; memcpy(x, x1, sizeof(double) * n); memcpy(y, y1, sizeof(double) * m); memcpy(z, z1, sizeof(double) * n); }
        if (e1 <= 0.1 * tol) break;        /* (a tenth of the tolerance: a further step costs a factorization) */
    }
    if (verbose) qp_printf("polishing (Newton steps on the QCQP's optimality conditions, %d active rows, %d variables at bounds, %d added on the way): %d steps, the largest error %.1e -> %.1e, %.2f s\n",
                        nact, n - nfree, nadded, steps, e_in, ebest, now() - tp0);
    {   static int sqp = -1; if (sqp < 0) { const char *e_ = getenv("BRISK_QCSQP"); sqp = e_ ? atoi(e_) : 1; }   /* (0: off) */
        if (sqp && ebest > tol) qcqp_sqp_polish(P, x, y, z, tol, &ebest, verbose); }
    free(xb); free(yb); free(zb); free(qmk); if (K) schol_free(K); free(sgn);
    free(qk); free(ax); free(ra); free(ca); free(x1); free(y1); free(z1); free(cid); free(rid);
    free(g); free(r); free(qx); free(rhs); free(sol); free(res); free(e); free(work);
}

int qcqp_solve(const QpProb *P, const QpOpts *o, QpRes *Res) {
    const double t0 = now();
    QpOpts od; if (!o) { qp_default_opts(&od); o = &od; }
    memset(Res, 0, sizeof(*Res));
    const int n = P->n, m = P->m, verbose = o->verbose;
    Res->x = xz(sizeof(double) * (n + 1)); Res->y = xz(sizeof(double) * (m + 1)); Res->z = xz(sizeof(double) * (n + 1));
    if (!qcqp_socp) {
        fprintf(stderr, "brisk: a problem with quadratic constraints is solved by the SOCP solver: run it with brisk (not brisk_qp)\n");
        Res->status = QP_INPUT; return Res->status;
    }
    /* the quadratic rows: which side, and the factors */
    int *qk = xm(sizeof(int) * (m + 1)); for (int i = 0; i < m; i++) qk[i] = -1;
    for (int k = 0; k < P->nqc; k++) qk[P->qrow[k]] = k;
    Fac *FQ = xz(sizeof(Fac) * (P->nqc + 1)); double *qsg = xm(sizeof(double) * (P->nqc + 1));
    int bad = 0;
    int *floc = xm(sizeof(int) * (n + 1)); for (int j = 0; j < n; j++) floc[j] = -1;     /* (factor_psd's work array) */
    for (int k = 0; k < P->nqc && !bad; k++) {
        const int i = P->qrow[k];
        if (fin(P->rl[i]) && fin(P->ru[i])) {
            fprintf(stderr, "brisk: row %s: a quadratic constraint with two finite sides (or an equality) is not convex\n", P->rname && P->rname[i] ? P->rname[i] : "?");
            bad = 1; break;
        }
        if (!fin(P->rl[i]) && !fin(P->ru[i])) { qsg[k] = 0; continue; }         /* a free row: no constraint */
        qsg[k] = fin(P->ru[i]) ? 1.0 : -1.0;
        double neg; const int e0 = P->qs[k], ne = P->qs[k + 1] - e0;
        const int rc = factor_psd(n, ne, P->qi + e0, P->qj + e0, P->qv + e0, qsg[k], &FQ[k], &neg, floc);
        if (rc == 1) { fprintf(stderr, "brisk: row %s: the quadratic constraint is not convex (a pivot %.1e of its matrix%s)\n", P->rname && P->rname[i] ? P->rname[i] : "?", neg, qsg[k] < 0 ? ", negated for the lower side" : ""); bad = 1; }
        if (rc == 2) { fprintf(stderr, "brisk: row %s: a quadratic constraint with a connected block of more than %d variables (not yet supported)\n", P->rname && P->rname[i] ? P->rname[i] : "?", QC_DENSE_MAX); bad = 1; }
    }
    Fac F0; memset(&F0, 0, sizeof F0);
    const int hasH = P->Hp && P->Hp[n] > 0;
    if (!bad && hasH) {
        int ne = P->Hp[n]; int *hi = xm(sizeof(int) * (ne + 1)), *hj = xm(sizeof(int) * (ne + 1));
        for (int j = 0; j < n; j++) for (int p = P->Hp[j]; p < P->Hp[j + 1]; p++) { hi[p] = P->Hi[p]; hj[p] = j; }
        double neg; const int rc = factor_psd(n, ne, hi, hj, P->Hx, 1.0, &F0, &neg, floc);
        if (rc == 1) { fprintf(stderr, "brisk: the objective's quadratic part is not convex (a pivot %.1e)\n", neg); bad = 1; }
        if (rc == 2) { fprintf(stderr, "brisk: the objective's quadratic part has a connected block of more than %d variables (not yet supported with quadratic constraints)\n", QC_DENSE_MAX); bad = 1; }
        free(hi); free(hj);
    }
    free(floc);
    if (bad) {
        for (int k = 0; k < P->nqc; k++) fac_free(&FQ[k]);
        free(FQ); free(qsg); free(qk); fac_free(&F0);
        Res->status = QP_INPUT; Res->time = now() - t0; return Res->status;
    }
    /* the rows of A as lists (for the cone entries: a row of A is a column of SeDuMi's A) */
    int *rp = xz(sizeof(int) * (m + 2)), *rj; double *rv;
    for (int j = 0; j < n; j++) for (int p = P->Ap[j]; p < P->Ap[j + 1]; p++) rp[P->Ai[p] + 1]++;
    for (int i = 0; i < m; i++) rp[i + 1] += rp[i];
    rj = xm(sizeof(int) * (rp[m] + 1)); rv = xm(sizeof(double) * (rp[m] + 1));
    { int *f = xm(sizeof(int) * (m + 1)); memcpy(f, rp, sizeof(int) * m);
      for (int j = 0; j < n; j++) for (int p = P->Ap[j]; p < P->Ap[j + 1]; p++) { const int q = f[P->Ai[p]]++; rj[q] = j; rv[q] = P->Ax[p]; }
      free(f); }
    const int ny = n + (hasH ? 1 : 0);
    Cols C; memset(&C, 0, sizeof C);
    /* what each cone entry stands for, for the multipliers: kind 0 row, 1 bound; idx; sign (+1: the multiplier adds to
     * y_i or z_j, -1: subtracts) */
    int ecap = 1024, ne_ = 0; int *ek = xm(sizeof(int) * ecap), *ei = xm(sizeof(int) * ecap); signed char *es = xm(ecap);
#define ENTRY(K, I, S) do { if (ne_ == ecap) { ecap *= 2; ek = realloc(ek, sizeof(int) * ecap); ei = realloc(ei, sizeof(int) * ecap); es = realloc(es, ecap); } ek[ne_] = (K); ei[ne_] = (I); es[ne_] = (S); ne_++; } while (0)
    /* K.f: equality rows (linear), fixed variables: c - A'y = rl - a'x = 0 */
    int nf = 0;
    for (int i = 0; i < m; i++) if (qk[i] < 0 && fin(P->rl[i]) && P->rl[i] == P->ru[i]) {
        col_begin(&C, P->rl[i]); for (int q = rp[i]; q < rp[i + 1]; q++) col_add(&C, rj[q], rv[q]); ENTRY(0, i, -1); nf++;
    }
    for (int j = 0; j < n; j++) if (fin(P->xl[j]) && P->xl[j] == P->xu[j]) { col_begin(&C, P->xl[j]); col_add(&C, j, 1.0); ENTRY(1, j, -1); nf++; }
    /* K.l: the other linear rows' finite sides, the bounds */
    int nl = 0;
    for (int i = 0; i < m; i++) {
        if (qk[i] >= 0 || (fin(P->rl[i]) && P->rl[i] == P->ru[i])) continue;
        if (fin(P->rl[i])) { col_begin(&C, -P->rl[i]); for (int q = rp[i]; q < rp[i + 1]; q++) col_add(&C, rj[q], -rv[q]); ENTRY(0, i, 1); nl++; }
        if (fin(P->ru[i])) { col_begin(&C, P->ru[i]); for (int q = rp[i]; q < rp[i + 1]; q++) col_add(&C, rj[q], rv[q]); ENTRY(0, i, -1); nl++; }
    }
    for (int j = 0; j < n; j++) {
        if (fin(P->xl[j]) && P->xl[j] == P->xu[j]) continue;
        if (fin(P->xl[j])) { col_begin(&C, -P->xl[j]); col_add(&C, j, -1.0); ENTRY(1, j, 1); nl++; }
        if (fin(P->xu[j])) { col_begin(&C, P->xu[j]); col_add(&C, j, 1.0); ENTRY(1, j, -1); nl++; }
    }
    /* K.r: each quadratic row (t1 = u - s a'x, t2 = 1, w = F x; s = 1 for an upper side, -1 for a lower one), the
     * objective (t, 1, F0 x) */
    int nr = 0, *rsz = xm(sizeof(int) * (P->nqc + 2));
    for (int k = 0; k < P->nqc; k++) {
        if (qsg[k] == 0) continue;
        const int i = P->qrow[k]; const double sg = qsg[k], u = sg > 0 ? P->ru[i] : -P->rl[i];
        col_begin(&C, u); for (int q = rp[i]; q < rp[i + 1]; q++) col_add(&C, rj[q], sg * rv[q]); ENTRY(0, i, sg > 0 ? -1 : 1);
        col_begin(&C, 1.0); ENTRY(2, 0, 0);
        for (int r = 0; r < FQ[k].nr; r++) { col_begin(&C, 0.0); for (int q = FQ[k].rp[r]; q < FQ[k].rp[r + 1]; q++) col_add(&C, FQ[k].ri[q], -FQ[k].rv[q]); ENTRY(2, 0, 0); }
        rsz[nr++] = 2 + FQ[k].nr;
    }
    if (hasH) {
        col_begin(&C, 0.0); col_add(&C, n, -1.0); ENTRY(2, 0, 0);
        col_begin(&C, 1.0); ENTRY(2, 0, 0);
        for (int r = 0; r < F0.nr; r++) { col_begin(&C, 0.0); for (int q = F0.rp[r]; q < F0.rp[r + 1]; q++) col_add(&C, F0.ri[q], -F0.rv[q]); ENTRY(2, 0, 0); }
        rsz[nr++] = 2 + F0.nr;
    }
#undef ENTRY
    /* SeDuMi's problem: A (ny x C.n) by columns, b = -(c, 1) */
    SedumiProb S; memset(&S, 0, sizeof S);
    S.m = ny; S.n = C.n; S.Ap = C.cp; S.Ai = C.ri; S.Ax = C.rv; S.c = C.c;
    S.b = xz(sizeof(double) * (ny + 1)); for (int j = 0; j < n; j++) S.b[j] = -P->c[j]; if (hasH) S.b[n] = -1.0;
    S.nf = nf; S.nl = nl; S.nr = nr; S.r = rsz;
    if (!S.Ai) { S.Ai = xm(sizeof(int)); S.Ax = xm(sizeof(double)); }
    if (verbose) qp_printf("QCQP %s: n = %d, m = %d, %d quadratic constraints; as an SOCP: %d free, %d nonnegative, %d rotated cones, %d unknowns\n",
                        P->name, n, m, P->nqc, nf, nl, nr, ny);
    /* the SOCP solver */
    char *av[16]; int ac = 0; char tolbuf[32], itbuf[32], tlbuf[32];
    if (verbose == 0) av[ac++] = "-q"; else if (verbose >= 2) av[ac++] = "-v";
    /* (25 October 2026, the SOCP developer) no second attempt of the SOCP solver with shorter steps after a first attempt at
     * reduced accuracy (within 1e-6; one in numerical difficulties is still repeated): that point comes back as it is, since the polishing and the second pass below take it from there (the
     * second attempt started again from the beginning: QCFIXQTWIN_2000 32 + 48 iterations, QCTOUCH_2000 54 + 81, 16.6 s,
     * QCVAR_NEAR_2000 30 + 33 with the same results as 32, 54 (6.6 s) and 30 without it). BRISK_QCCONERETRY=1: as before */
    { static int cr = -1; if (cr < 0) { const char *e_ = getenv("BRISK_QCCONERETRY"); cr = e_ ? atoi(e_) : 0; }
      if (!cr) { av[ac++] = "-coneretry"; av[ac++] = "0"; } }
    /* the SOCP solver at the QCQP's tolerance; its errors are measured on the reformulation, so the status below is
     * set by the errors on the problem as read (QPLIB_3029: the dual error there 4.3e-6 at the SOCP solver's 1e-8; a
     * tighter SOCP tolerance costs much on some files, QPLIB_2468: 19, 35, 97 iterations for 1e-8, 1e-9, 1e-10 -- the
     * SOCP solver's slow convergence at high accuracy, ISSUES) */
    if (o->tol > 0) { snprintf(tolbuf, sizeof tolbuf, "%g", o->tol); av[ac++] = "-tol"; av[ac++] = tolbuf; }
    if (o->maxit > 0) { snprintf(itbuf, sizeof itbuf, "%d", o->maxit); av[ac++] = "-maxit"; av[ac++] = itbuf; }
    if (o->timelimit > 0) { snprintf(tlbuf, sizeof tlbuf, "%g", o->timelimit); av[ac++] = "-timelimit"; av[ac++] = tlbuf; }
    av[ac] = NULL;
    /* (24 October 2026, the SOCP developer) at most two passes: the SOCP solver's errors are measured on the reformulation,
     * and the map back to the QCQP can magnify them (QPLIB_2482: 5.8e-9 on the cones, a dual error of 7.1e-8 on the
     * QCQP after the polishing). When the first pass meets the tolerance on the cones but not on the problem as read,
     * the cones are solved again to e_cones (tol / e_QCQP)^2 (halved, at least 1e-12: the square law below), and the
     * better result is kept: the problems that pass the first time cost nothing more (a tighter
     * tolerance for all costs much: QPLIB_2468 19, 35, 97 iterations at 1e-8, 1e-9, 1e-10). BRISK_QCRETOL=0: one pass */
    static int retol = -1; if (retol < 0) { const char *e = getenv("BRISK_QCRETOL"); retol = e ? atoi(e) : 1; }
    const double qtol = o->tol > 0 ? o->tol : 1e-8;
    double *bx = NULL, *by = NULL, *bz = NULL; QpRes B0; memset(&B0, 0, sizeof B0); int have_b = 0, iters_tot = 0; double t_pass1 = 0;
    double tol_pass = o->tol > 0 ? o->tol : 0;
    for (int pass = 0; pass < 2; pass++) {
    if (pass == 1) {          /* the second pass's tolerance replaces the -tol of the first */
        int k = 0; for (int a = 0; a < ac; a++) { if (!strcmp(av[a], "-tol")) { a++; continue; } av[k++] = av[a]; }
        ac = k; snprintf(tolbuf, sizeof tolbuf, "%g", tol_pass); av[ac++] = "-tol"; av[ac++] = tolbuf; av[ac] = NULL;
        for (int j = 0; j < n; j++) { Res->x[j] = 0; Res->z[j] = 0; }
        for (int i = 0; i < m; i++) Res->y[i] = 0;
    }
    SedumiRes SR; memset(&SR, 0, sizeof SR);
    const double ts0 = now();
    if (verbose && pass == 0) qp_printf("the reformulation: %.2f s\n", ts0 - t0);
    qcqp_socp(&S, ac, av, &SR);
    if (verbose) qp_printf("the SOCP solver%s: %.2f s\n", pass ? " (again, at the tolerance the map back needs)" : "", now() - ts0);
    /* back: x = y(1:n); the rows' and bounds' multipliers from the cone entries' (SeDuMi's x) */
    if (SR.y) for (int j = 0; j < n; j++) Res->x[j] = SR.y[j];
    if (SR.x) for (int e = 0; e < ne_ && e < S.n; e++) {
        if (ek[e] == 0) Res->y[ei[e]] += es[e] * SR.x[e];
        else if (ek[e] == 1) Res->z[ei[e]] += es[e] * SR.x[e];
    }
    iters_tot += SR.iters; Res->iter = iters_tot;
    switch (SR.status) {
        case 0: Res->status = QP_OPTIMAL; break;
        case 1: Res->status = QP_DUAL_INFEASIBLE; break;      /* SeDuMi's primal infeasible: our problem (its dual) unbounded */
        case 2: Res->status = QP_PRIMAL_INFEASIBLE; break;
        case 3: Res->status = QP_MAXIT; break;
        case 5: Res->status = QP_REDUCED; break;
        case 6: Res->status = QP_TIME; break;
        default: Res->status = QP_NUMERICAL; break;
    }
    { static int pol = -1; if (pol < 0) { const char *e = getenv("BRISK_QCPOLISH"); pol = e ? atoi(e) : 1; }
      if (pol && (SR.status == 0 || SR.status == 3 || SR.status == 5 || SR.status == 4 || SR.status == 6)) qcqp_polish(P, Res->x, Res->y, Res->z, o->tol > 0 ? o->tol : 1e-8, verbose); }
    Res->obj = qp_objective(P, Res->x);
    qp_errors(P, Res->x, Res->y, Res->z, &Res->pres, &Res->dres, &Res->gap);
    /* the status of a solved problem by its errors on the problem as read */
    if (Res->status == QP_OPTIMAL || Res->status == QP_REDUCED || Res->status == QP_MAXIT || Res->status == QP_NUMERICAL || Res->status == QP_TIME) {
        const double e = fmax(Res->pres, fmax(Res->dres, Res->gap)), tol = o->tol > 0 ? o->tol : 1e-8;
        if (e <= tol) Res->status = QP_OPTIMAL;
        else if (e <= fmax(1e-6, 100 * tol)) Res->status = QP_REDUCED;
        else if (Res->status == QP_OPTIMAL || Res->status == QP_REDUCED) Res->status = QP_NUMERICAL;
    }
    const double e_q = fmax(Res->pres, fmax(Res->dres, Res->gap));
    double e_c = 0; for (int k = 0; k < 6; k++) if (isfinite(SR.err[k])) e_c = fmax(e_c, SR.err[k]);
    free(SR.x); free(SR.y); free(SR.z); for (int b = 0; b < 2; b++) { free(SR.bnd[b].x); free(SR.bnd[b].y); }   /* (sedumi_result_free: not linked into brisk_qp) */
    if (pass == 1) {          /* the better of the two passes */
        const double e_b = fmax(B0.pres, fmax(B0.dres, B0.gap));
        if (!(e_q < e_b)) { memcpy(Res->x, bx, sizeof(double) * n); memcpy(Res->y, by, sizeof(double) * m); memcpy(Res->z, bz, sizeof(double) * n);
            Res->status = B0.status; Res->obj = B0.obj; Res->pres = B0.pres; Res->dres = B0.dres; Res->gap = B0.gap; }
        if (verbose) qp_printf("the second pass: the largest error %.1e -> %.1e, %s\n", e_b, e_q, e_q < e_b ? "kept" : "the first kept");
        Res->resolves = 2; Res->tol2 = tol_pass; Res->kept2 = e_q < e_b; Res->t_first = t_pass1;
        break;
    }
    /* a second pass only when the cones met their tolerance and the problem as read did not */
    if (!retol || SR.status != 0 || !(e_q > qtol) || !(e_c > 0) || !(e_c <= qtol) || !isfinite(e_q)) break;
    /* (the QCQP's error behaves as the square root of the cones', not in proportion: QPLIB_2482 5.8e-9 -> 7.1e-8 and
     * 3.2e-10 -> 1.8e-8 on the cones and the QCQP; two vectors of a cone with u'v = eps lie at an angle of order
     * sqrt(eps), so F_k'v_k differs from y_k Q_k x by about sqrt(eps): IMPLEMENTATION 28.4, the polishing) */
    tol_pass = fmax(1e-12, 0.5 * e_c * (qtol / e_q) * (qtol / e_q));
    if (!(tol_pass < 0.99 * e_c)) break;
    if (o->timelimit > 0 && now() - t0 > 0.5 * o->timelimit) break;
    bx = xm(sizeof(double) * (n + 1)); by = xm(sizeof(double) * (m + 1)); bz = xm(sizeof(double) * (n + 1));
    memcpy(bx, Res->x, sizeof(double) * n); memcpy(by, Res->y, sizeof(double) * m); memcpy(bz, Res->z, sizeof(double) * n);
    B0 = *Res; have_b = 1; t_pass1 = now() - t0;
    if (verbose) qp_printf("the errors on the QCQP %.1e against %.1e on the cones (tolerance %.1e): the cones again at %.1e\n", e_q, e_c, qtol, tol_pass);
    }
    if (have_b) { free(bx); free(by); free(bz); }
    free(S.b); free(C.c); free(C.cp); free(C.ri); free(C.rv); free(rsz);
    free(ek); free(ei); free(es); free(rp); free(rj); free(rv);
    for (int k = 0; k < P->nqc; k++) fac_free(&FQ[k]);
    free(FQ); free(qsg); free(qk); fac_free(&F0);
    Res->time = now() - t0;
    return Res->status;
}
