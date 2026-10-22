/* simplex_lu.c - the factorization of the simplex basis and its updates (simplex_int.h).
 *
 * Factorization: right-looking sparse Gaussian elimination with Markowitz pivot choice and threshold
 * partial pivoting (Suhl and Suhl 1990: columns and rows searched by increasing count, a candidate
 * accepted when |a_ij| >= u max_k |a_kj|, the search stopped after a few candidates). Logical columns
 * are pivoted first in bulk; singletons have Markowitz count 0 and are taken before the kernel without
 * a special path (no fill). Columns that cannot be pivoted (structurally or numerically singular basis)
 * are replaced by the logicals of the rows left over.
 *
 * Solves: column etas of L forward, the row etas R of the updates, U backward by columns (FTRAN);
 * U^T forward by rows, R^T backward, L^T backward by rows (BTRAN). When the right-hand side is sparse
 * and the result is expected sparse, the nonzero pattern of the result is found first by a depth-first
 * search over the graph of the factor (Gilbert and Peierls 1988), and only the reached pivots are
 * processed (hypersparse solves, Hall and McKinnon 2005).
 *
 * Updates: Forrest and Tomlin (1972). The entering column's partial FTRAN (after L and R: the spike)
 * replaces the column of the leaving position in U; that position's pivot moves to the end of the
 * order, and its row is eliminated by a row eta whose entries come from the partial BTRAN of the
 * leaving row (after U^T), which the simplex iteration computes anyway. The new diagonal entry is
 * checked against u_rr * alpha_rq (the determinant identity). */
#include "simplex_int.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define SX_TINY 1e-14          /* entries below this are dropped from solve results */
#define SX_PIVTOL 1e-11        /* absolute pivot tolerance of the factorization */
#define SX_DENSE_MIN 48        /* the dense kernel: at least this many active rows and columns ... */
#define SX_DENSE_FRAC 0.7      /* ... and this fraction of their entries nonzero */

static void *xmalloc(size_t n) { void *p = malloc(n ? n : 1); if (!p) { fprintf(stderr, "brisk: out of memory (simplex)\n"); exit(1); } return p; }
static void *xcalloc(size_t n, size_t s) { void *p = calloc(n ? n : 1, s ? s : 1); if (!p) { fprintf(stderr, "brisk: out of memory (simplex)\n"); exit(1); } return p; }
static void *xrealloc(void *q, size_t n) { void *p = realloc(q, n ? n : 1); if (!p) { fprintf(stderr, "brisk: out of memory (simplex)\n"); exit(1); } return p; }

/* ---------------------------------------------------------------- vectors */
void sxv_init(SxVec *v, int n) {
    v->n = n; v->cnt = 0;
    v->idx = xmalloc(sizeof(int) * ((size_t)n + 1));
    v->val = xcalloc((size_t)n + 1, sizeof(double));
    v->mark = xcalloc((size_t)n + 1, 1);
}
void sxv_free(SxVec *v) { free(v->idx); free(v->val); free(v->mark); memset(v, 0, sizeof *v); }
void sxv_clear(SxVec *v) {
    if (v->cnt >= 0 && v->cnt < v->n / 8) for (int k = 0; k < v->cnt; k++) v->val[v->idx[k]] = 0;
    else memset(v->val, 0, sizeof(double) * (size_t)v->n);
    v->cnt = 0;
}
void sxv_reindex(SxVec *v) {
    int c = 0;
    for (int i = 0; i < v->n; i++) {
        if (v->val[i] == 0) continue;
        if (fabs(v->val[i]) < SX_TINY) { v->val[i] = 0; continue; }
        v->idx[c++] = i;
    }
    v->cnt = c;
}
void sxv_copy(SxVec *dst, const SxVec *src) {
    sxv_clear(dst);
    if (src->cnt >= 0) { for (int k = 0; k < src->cnt; k++) { const int i = src->idx[k]; dst->val[i] = src->val[i]; dst->idx[k] = i; } dst->cnt = src->cnt; }
    else { memcpy(dst->val, src->val, sizeof(double) * (size_t)src->n); sxv_reindex(dst); }
}
/* drop tiny entries from the list of a vector solved in hypersparse mode */
static void sxv_prune(SxVec *v) {
    int c = 0;
    for (int k = 0; k < v->cnt; k++) {
        const int i = v->idx[k];
        if (fabs(v->val[i]) < SX_TINY) { v->val[i] = 0; continue; }
        v->idx[c++] = i;
    }
    v->cnt = c;
}

/* ---------------------------------------------------------------- the active matrix of the elimination */
typedef struct {
    int m;
    int *cs, *cl, *cc, *ci; double *cv; int cend, ccap;     /* columns (basis positions): rows and values */
    int *rs, *rl, *rc, *ri; int rend, rcap;                 /* rows: positions (pattern) */
    int *chead, *cnext, *cprev, *rhead, *rnext, *rprev;     /* count lists */
    double *cmax;                                           /* largest |entry| of a column, < 0: stale */
    char *cact, *ract;
} Act;

static void lst_ins(int *head, int *next, int *prev, int k, int j) { next[j] = head[k]; prev[j] = -1; if (head[k] >= 0) prev[head[k]] = j; head[k] = j; }
static void lst_del(int *head, int *next, int *prev, int k, int j) {
    if (prev[j] >= 0) next[prev[j]] = next[j]; else head[k] = next[j];
    if (next[j] >= 0) prev[next[j]] = prev[j];
}
static void col_compact(Act *A, int extra) {
    long tot = 0;
    for (int j = 0; j < A->m; j++) if (A->cact[j]) tot += A->cl[j] + 2;
    long cap = 2 * tot + extra + 4 * (long)A->m + 16;
    if (cap < A->ccap) cap = A->ccap;
    int *ni = xmalloc(sizeof(int) * cap); double *nv = xmalloc(sizeof(double) * cap);
    int e = 0;
    for (int j = 0; j < A->m; j++) {
        if (!A->cact[j]) { A->cs[j] = 0; A->cc[j] = 0; continue; }
        memcpy(ni + e, A->ci + A->cs[j], sizeof(int) * A->cl[j]); memcpy(nv + e, A->cv + A->cs[j], sizeof(double) * A->cl[j]);
        A->cs[j] = e; A->cc[j] = A->cl[j] + 2; e += A->cc[j];
    }
    free(A->ci); free(A->cv); A->ci = ni; A->cv = nv; A->cend = e; A->ccap = (int)cap;
}
static void col_reserve(Act *A, int j, int need) {
    if (A->cl[j] + need <= A->cc[j]) return;
    const int nc = 2 * (A->cl[j] + need) + 4;
    if ((long)A->cend + nc > A->ccap) col_compact(A, nc);
    memmove(A->ci + A->cend, A->ci + A->cs[j], sizeof(int) * A->cl[j]);
    memmove(A->cv + A->cend, A->cv + A->cs[j], sizeof(double) * A->cl[j]);
    A->cs[j] = A->cend; A->cc[j] = nc; A->cend += nc;
}
static void row_compact(Act *A, int extra) {
    long tot = 0;
    for (int i = 0; i < A->m; i++) if (A->ract[i]) tot += A->rl[i] + 2;
    long cap = 2 * tot + extra + 4 * (long)A->m + 16;
    if (cap < A->rcap) cap = A->rcap;
    int *ni = xmalloc(sizeof(int) * cap);
    int e = 0;
    for (int i = 0; i < A->m; i++) {
        if (!A->ract[i]) { A->rs[i] = 0; A->rc[i] = 0; continue; }
        memcpy(ni + e, A->ri + A->rs[i], sizeof(int) * A->rl[i]);
        A->rs[i] = e; A->rc[i] = A->rl[i] + 2; e += A->rc[i];
    }
    free(A->ri); A->ri = ni; A->rend = e; A->rcap = (int)cap;
}
static void row_reserve(Act *A, int i, int need) {
    if (A->rl[i] + need <= A->rc[i]) return;
    const int nc = 2 * (A->rl[i] + need) + 4;
    if ((long)A->rend + nc > A->rcap) row_compact(A, nc);
    memmove(A->ri + A->rend, A->ri + A->rs[i], sizeof(int) * A->rl[i]);
    A->rs[i] = A->rend; A->rc[i] = nc; A->rend += nc;
}
static double col_max(Act *A, int j) {
    if (A->cmax[j] >= 0) return A->cmax[j];
    double mx = 0; const int s = A->cs[j], e = s + A->cl[j];
    for (int p = s; p < e; p++) { const double a = fabs(A->cv[p]); if (a > mx) mx = a; }
    return A->cmax[j] = mx;
}
static int col_find(const Act *A, int j, int i) {
    const int s = A->cs[j], e = s + A->cl[j];
    for (int p = s; p < e; p++) if (A->ci[p] == i) return p;
    return -1;
}
static void row_remove(Act *A, int i, int j) {   /* position j out of row i's pattern */
    const int s = A->rs[i], e = s + A->rl[i];
    for (int p = s; p < e; p++) if (A->ri[p] == j) { A->ri[p] = A->ri[e - 1]; A->rl[i]--; return; }
}

/* Markowitz search: returns 1 and (r, c) or 0 when no acceptable pivot remains */
static int find_pivot(Act *A, double u, int *pr, int *pc) {
    long best = -1; double bestv = 0; int br = -1, bc = -1, nsearch = 0;
    const int lim = 4;
    for (int k = 1; k <= A->m; k++) {
        for (int j = A->chead[k]; j >= 0; j = A->cnext[j]) {
            const double cm = col_max(A, j);
            const int s = A->cs[j], e = s + A->cl[j];
            for (int p = s; p < e; p++) {
                const double a = fabs(A->cv[p]);
                if (a < u * cm || a < SX_PIVTOL) continue;
                const long mer = (long)(k - 1) * (A->rl[A->ci[p]] - 1);
                if (best < 0 || mer < best || (mer == best && a > bestv)) { best = mer; bestv = a; br = A->ci[p]; bc = j; }
            }
            if (best >= 0 && (best <= (long)(k - 1) * (k - 1) || ++nsearch >= lim)) goto done;
        }
        for (int i = A->rhead[k]; i >= 0; i = A->rnext[i]) {
            const int s = A->rs[i], e = s + A->rl[i];
            for (int p = s; p < e; p++) {
                const int j = A->ri[p];
                const int q = col_find(A, j, i);
                const double a = fabs(A->cv[q]);
                if (a < u * col_max(A, j) || a < SX_PIVTOL) continue;
                const long mer = (long)(k - 1) * (A->cl[j] - 1);
                if (best < 0 || mer < best || (mer == best && a > bestv)) { best = mer; bestv = a; br = i; bc = j; }
            }
            if (best >= 0 && (best <= (long)k * (k - 1) || ++nsearch >= lim)) goto done;
        }
        if (best >= 0 && best <= (long)k * k) goto done;
    }
done:
    if (best < 0) return 0;
    *pr = br; *pc = bc; return 1;
}

/* the U entries recorded during the elimination: (row r pivoted, position j, value) */
typedef struct { int *r, *j; double *v; int n, cap; } Trip;
static void trip_add(Trip *T, int r, int j, double v) {
    if (T->n == T->cap) { T->cap = 2 * T->cap + 1024; T->r = xrealloc(T->r, sizeof(int) * T->cap); T->j = xrealloc(T->j, sizeof(int) * T->cap); T->v = xrealloc(T->v, sizeof(double) * T->cap); }
    T->r[T->n] = r; T->j[T->n] = j; T->v[T->n] = v; T->n++;
}
static void l_add(SxLU *F, int i, double v) {   /* to the column being built: lstart[nl] its start, lstart[nl + 1] its end */
    if (F->lstart[F->nl + 1] + 1 > F->lcap) { F->lcap = 2 * F->lcap + 1024; F->lidx = xrealloc(F->lidx, sizeof(int) * F->lcap); F->lval = xrealloc(F->lval, sizeof(double) * F->lcap); }
    const int p = F->lstart[F->nl + 1]++;
    F->lidx[p] = i; F->lval[p] = v;
}

void sxlu_free(SxLU *F) {
    free(F->lrow); free(F->lstart); free(F->lidx); free(F->lval); free(F->lrstart); free(F->lridx); free(F->lrval); free(F->lpos);
    free(F->uord); free(F->uordpos); free(F->uz); free(F->uzpos); free(F->udiag); free(F->ucs); free(F->ucl); free(F->ucc); free(F->uci); free(F->ucv);
    free(F->urs); free(F->url); free(F->urc); free(F->uri); free(F->urv);
    free(F->rpiv); free(F->rstart); free(F->ridx); free(F->rval);
    free(F->stack); free(F->wmark); free(F->wlist); free(F->wval); free(F->dfs_pos);
    free(F->lcs_row); free(F->lcl_row); free(F->lrorder); free(F->lrlen);
    memset(F, 0, sizeof *F);
}

/* the dense kernel: once the active matrix is dense, it is copied into a dense array (row-major) and factorized by
 * LU with partial pivoting, column by column; the pivots, the L columns and the U entries are recorded as in the
 * sparse elimination. A column without an entry above the absolute pivot tolerance is left (its position gets a
 * logical at the end, as in the sparse case). */
static void dense_kernel(SxLU *F, Act *A, Trip *T, double u, int *rowof, int *prow, int *pcol, double *pval, int *np) {
    (void)u;
    const int m = A->m;
    int nr = 0, nc = 0;
    int *rid = xmalloc(sizeof(int) * (m + 1)), *cid = xmalloc(sizeof(int) * (m + 1)), *rmap = xmalloc(sizeof(int) * (m + 1));
    for (int i = 0; i < m; i++) { rmap[i] = -1; if (A->ract[i]) { rmap[i] = nr; rid[nr++] = i; } }
    for (int j = 0; j < m; j++) if (A->cact[j]) cid[nc++] = j;
    double *D = xcalloc((size_t)nr * nc, sizeof(double));
    for (int jj = 0; jj < nc; jj++) { const int j = cid[jj]; for (int p = A->cs[j]; p < A->cs[j] + A->cl[j]; p++) D[(size_t)rmap[A->ci[p]] * nc + jj] = A->cv[p]; }
    int *left = xmalloc(sizeof(int) * (nr + 1)); int nleft = nr;
    for (int k = 0; k < nr; k++) left[k] = k;
    int *lr = xmalloc(sizeof(int) * (nr + 1)); double *lv = xmalloc(sizeof(double) * (nr + 1));
    for (int jj = 0; jj < nc && nleft > 0; jj++) {
        int kb = -1; double best = 0;
        for (int k = 0; k < nleft; k++) { const double a = fabs(D[(size_t)left[k] * nc + jj]); if (a > best) { best = a; kb = k; } }
        if (kb < 0 || best < SX_PIVTOL) continue;
        const int ii = left[kb]; left[kb] = left[--nleft];
        const double *prw = D + (size_t)ii * nc;
        const double piv = prw[jj];
        /* L column */
        int nl = 0;
        for (int k = 0; k < nleft; k++) { const int i = left[k]; const double a = D[(size_t)i * nc + jj]; if (a != 0) { lr[nl] = i; lv[nl] = a / piv; nl++; } }
        if (nl) {
            F->lrow[F->nl] = rid[ii]; F->lstart[F->nl + 1] = F->lstart[F->nl];
            for (int k = 0; k < nl; k++) l_add(F, rid[lr[k]], lv[k]);
            F->nl++;
        }
        /* U row: the later columns */
        for (int kk = jj + 1; kk < nc; kk++) if (prw[kk] != 0) trip_add(T, rid[ii], cid[kk], prw[kk]);
        /* the update of the rows left */
        for (int k = 0; k < nl; k++) {
            double *row = D + (size_t)lr[k] * nc; const double l = lv[k];
            for (int kk = jj + 1; kk < nc; kk++) row[kk] -= l * prw[kk];
        }
        F->factor_work += (double)nl * (nc - jj);
        const int r = rid[ii], c = cid[jj];
        A->ract[r] = 0; A->cact[c] = 0;
        rowof[c] = r; prow[*np] = r; pcol[*np] = c; pval[*np] = piv; (*np)++;
    }
    free(rid); free(cid); free(rmap); free(D); free(left); free(lr); free(lv);
}

int sxlu_factor(SxLU *F, int m, int n, const int *Ap, const int *Ai, const double *Ax, int *var, int *out) {
    if (F->m != m) { sxlu_free(F); F->m = m; }
    const double u = F->pivthr > 0 ? F->pivthr : 0.1;
    /* work arrays kept between factorizations */
    if (!F->stack) {
        F->stack = xmalloc(sizeof(int) * (2 * (size_t)m + 2)); F->wmark = xcalloc((size_t)m + 1, sizeof(int));
        F->wlist = xmalloc(sizeof(int) * ((size_t)m + 1)); F->wval = xcalloc((size_t)m + 1, sizeof(double)); F->dfs_pos = xmalloc(sizeof(int) * ((size_t)m + 1));
        F->lpos = xmalloc(sizeof(int) * ((size_t)m + 1)); F->lcs_row = xmalloc(sizeof(int) * ((size_t)m + 1)); F->lcl_row = xmalloc(sizeof(int) * ((size_t)m + 1));
        F->lrstart = xmalloc(sizeof(int) * ((size_t)m + 2)); F->lrlen = xmalloc(sizeof(int) * ((size_t)m + 1)); F->lrorder = xmalloc(sizeof(int) * ((size_t)m + 1));
        F->lrow = xmalloc(sizeof(int) * ((size_t)m + 1)); F->lstart = xmalloc(sizeof(int) * ((size_t)m + 2));
        F->uordpos = xmalloc(sizeof(int) * ((size_t)m + 1)); F->udiag = xmalloc(sizeof(double) * ((size_t)m + 1));
        F->ucs = xmalloc(sizeof(int) * ((size_t)m + 1)); F->ucl = xmalloc(sizeof(int) * ((size_t)m + 1)); F->ucc = xmalloc(sizeof(int) * ((size_t)m + 1));
        F->urs = xmalloc(sizeof(int) * ((size_t)m + 1)); F->url = xmalloc(sizeof(int) * ((size_t)m + 1)); F->urc = xmalloc(sizeof(int) * ((size_t)m + 1));
    }
    int *prow = xmalloc(sizeof(int) * m), *pcol = xmalloc(sizeof(int) * m); double *pval = xmalloc(sizeof(double) * m);
    int *rowof = xmalloc(sizeof(int) * m);       /* pivot row of each position (-1: none yet) */
    char *rdone = xcalloc(m, 1);
    int np = 0;
    F->nl = 0; F->lstart[0] = 0; F->nr = 0; F->nupd = 0; F->rank_deficient = 0; F->factor_work = 0;
    Trip T = {0};
    for (int j = 0; j < m; j++) rowof[j] = -1;
    /* 1. logicals first (bulk): their rows leave the active matrix; entries of other columns there go to U */
    for (int j = 0; j < m; j++) {
        if (var[j] < n) continue;
        const int i = var[j] - n;
        if (rdone[i]) continue;          /* a second logical of the same row: singular, left to the end */
        rdone[i] = 1; rowof[j] = i; prow[np] = i; pcol[np] = j; pval[np] = 1.0; np++;
    }
    /* 1b. column singletons (structural columns with one entry in the rows left), in bulk like the logicals: no
     * elimination, their rows leave the active matrix (entries of other columns there go to U, the column's own entries
     * in rows pivoted before it too), and the counts of the columns with an entry in such a row drop, which can make
     * more singletons. Counts and a row-wise copy of the structural entries make it linear in the basis' entries. */
    {
        int *cnt = xcalloc(m, sizeof(int)), *rptr = xcalloc(m + 2, sizeof(int)), *stk = xmalloc(sizeof(int) * (m + 1)); int nst = 0;
        long nzb = 0;
        for (int j = 0; j < m; j++) {
            if (var[j] >= n) continue;
            for (int p = Ap[var[j]]; p < Ap[var[j] + 1]; p++) if (Ax[p] != 0) { rptr[Ai[p] + 1]++; nzb++; if (!rdone[Ai[p]]) cnt[j]++; }
        }
        for (int i = 0; i < m; i++) rptr[i + 1] += rptr[i];
        int *rj = xmalloc(sizeof(int) * (nzb + 1)), *fill = xmalloc(sizeof(int) * (m + 1));
        memcpy(fill, rptr, sizeof(int) * m);
        for (int j = 0; j < m; j++) {
            if (var[j] >= n) continue;
            for (int p = Ap[var[j]]; p < Ap[var[j] + 1]; p++) if (Ax[p] != 0) rj[fill[Ai[p]]++] = j;
            if (cnt[j] == 1) stk[nst++] = j;
        }
        while (nst > 0) {
            const int j = stk[--nst];
            if (rowof[j] >= 0 || cnt[j] != 1) continue;
            const int v = var[j];
            int i = -1; double a = 0;
            for (int p = Ap[v]; p < Ap[v + 1]; p++) if (Ax[p] != 0 && !rdone[Ai[p]]) { i = Ai[p]; a = Ax[p]; break; }
            if (i < 0 || fabs(a) < SX_PIVTOL) continue;    /* (left to the elimination) */
            for (int p = Ap[v]; p < Ap[v + 1]; p++) if (Ax[p] != 0 && Ai[p] != i) trip_add(&T, Ai[p], j, Ax[p]);
            rdone[i] = 1; rowof[j] = i; prow[np] = i; pcol[np] = j; pval[np] = a; np++;
            for (int p = rptr[i]; p < rptr[i + 1]; p++) { const int k = rj[p]; if (rowof[k] < 0 && --cnt[k] == 1) stk[nst++] = k; }
        }
        free(cnt); free(rptr); free(stk); free(rj); free(fill);
    }
    /* 2. the active matrix: structural columns, entries in rows not yet pivoted */
    Act A; memset(&A, 0, sizeof A); A.m = m;
    long nzs = 0;
    for (int j = 0; j < m; j++) if (var[j] < n) nzs += Ap[var[j] + 1] - Ap[var[j]];
    A.ccap = (int)(2 * nzs + 4 * (long)m + 16); A.ci = xmalloc(sizeof(int) * A.ccap); A.cv = xmalloc(sizeof(double) * A.ccap);
    A.rcap = A.ccap; A.ri = xmalloc(sizeof(int) * A.rcap);
    A.cs = xmalloc(sizeof(int) * m); A.cl = xcalloc(m, sizeof(int)); A.cc = xcalloc(m, sizeof(int));
    A.rs = xmalloc(sizeof(int) * m); A.rl = xcalloc(m, sizeof(int)); A.rc = xcalloc(m, sizeof(int));
    A.chead = xmalloc(sizeof(int) * (m + 1)); A.cnext = xmalloc(sizeof(int) * m); A.cprev = xmalloc(sizeof(int) * m);
    A.rhead = xmalloc(sizeof(int) * (m + 1)); A.rnext = xmalloc(sizeof(int) * m); A.rprev = xmalloc(sizeof(int) * m);
    A.cmax = xmalloc(sizeof(double) * m); A.cact = xcalloc(m, 1); A.ract = xcalloc(m, 1);
    for (int k = 0; k <= m; k++) A.chead[k] = A.rhead[k] = -1;
    int e = 0;
    for (int j = 0; j < m; j++) {
        A.cs[j] = e; A.cmax[j] = -1;
        if (rowof[j] >= 0) continue;
        if (var[j] >= n) { /* a duplicate logical: an empty active column */ A.cact[j] = 1; A.cc[j] = 2; e += 2; continue; }
        A.cact[j] = 1;
        const int v = var[j];
        for (int p = Ap[v]; p < Ap[v + 1]; p++) {
            const int i = Ai[p];
            if (Ax[p] == 0) continue;
            if (rdone[i]) trip_add(&T, i, j, Ax[p]);
            else { A.ci[e] = i; A.cv[e] = Ax[p]; e++; A.cl[j]++; A.rl[i]++; }
        }
        A.cc[j] = A.cl[j] + 2; e += 2;
    }
    A.cend = e;
    e = 0;
    for (int i = 0; i < m; i++) { A.rs[i] = e; if (!rdone[i]) { A.ract[i] = 1; A.rc[i] = A.rl[i] + 2; e += A.rc[i]; } A.rl[i] = 0; }
    A.rend = e;
    for (int j = 0; j < m; j++) if (A.cact[j]) for (int p = A.cs[j]; p < A.cs[j] + A.cl[j]; p++) { const int i = A.ci[p]; A.ri[A.rs[i] + A.rl[i]++] = j; }
    for (int j = 0; j < m; j++) if (A.cact[j]) lst_ins(A.chead, A.cnext, A.cprev, A.cl[j], j);
    for (int i = 0; i < m; i++) if (A.ract[i]) lst_ins(A.rhead, A.rnext, A.rprev, A.rl[i], i);
    /* 3. elimination */
    int *wpos = F->wmark;          /* row -> position in the column being updated (+1; 0 none) */
    int *lrows = F->wlist; double *lmul = F->wval; int *ucols = F->lrorder;   /* (lrorder is rebuilt after the elimination) */
    int r, c;
    /* the entries, rows and columns of the active matrix, for the switch to the dense kernel */
    long nzact = 0; int nca = 0, nra = 0;
    for (int j = 0; j < m; j++) if (A.cact[j]) { nca++; nzact += A.cl[j]; }
    for (int i = 0; i < m; i++) nra += A.ract[i];
    for (;;) {
        static double dfrac = -1; if (dfrac < 0) dfrac = getenv("SXDENSE") ? atof(getenv("SXDENSE")) : SX_DENSE_FRAC;   /* (experiments) */
        if (nca >= SX_DENSE_MIN && nra >= SX_DENSE_MIN && (double)nzact >= dfrac * (double)nca * nra) {
            dense_kernel(F, &A, &T, u, rowof, prow, pcol, pval, &np);
            break;
        }
        if (!find_pivot(&A, u, &r, &c)) break;
        nca--; nra--; nzact -= A.cl[c];
        const int pp = col_find(&A, c, r);
        const double piv = A.cv[pp];
        /* out of the count lists: the pivot row and column, the rows of column c, the columns of row r */
        lst_del(A.chead, A.cnext, A.cprev, A.cl[c], c); lst_del(A.rhead, A.rnext, A.rprev, A.rl[r], r);
        int nlr = 0;
        for (int p = A.cs[c]; p < A.cs[c] + A.cl[c]; p++) {
            const int i = A.ci[p];
            if (i == r) continue;
            lst_del(A.rhead, A.rnext, A.rprev, A.rl[i], i);
            lrows[nlr] = i; lmul[nlr] = A.cv[p] / piv; nlr++;
            row_remove(&A, i, c);
        }
        A.cact[c] = 0;            /* (row r stays active until its pattern has been used: a compaction keeps it) */
        /* L column */
        if (nlr) {
            F->lrow[F->nl] = r; F->lstart[F->nl + 1] = F->lstart[F->nl];
            for (int k = 0; k < nlr; k++) l_add(F, lrows[k], lmul[k]);
            F->nl++;
        }
        /* U row and the update of the columns of row r (its pattern copied: compactions move the row file) */
        int nur = 0;
        for (int p = A.rs[r]; p < A.rs[r] + A.rl[r]; p++) if (A.ri[p] != c) ucols[nur++] = A.ri[p];
        for (int t0 = 0; t0 < nur; t0++) {
            const int j = ucols[t0];
            lst_del(A.chead, A.cnext, A.cprev, A.cl[j], j);
            const int q = col_find(&A, j, r);
            const double urv = A.cv[q];
            A.ci[q] = A.ci[A.cs[j] + A.cl[j] - 1]; A.cv[q] = A.cv[A.cs[j] + A.cl[j] - 1]; A.cl[j]--; nzact--;
            trip_add(&T, r, j, urv);
            F->factor_work += nlr;
            if (nlr) {
                col_reserve(&A, j, nlr);
                const int s = A.cs[j];
                for (int t = s; t < s + A.cl[j]; t++) wpos[A.ci[t]] = t + 1;
                for (int k = 0; k < nlr; k++) {
                    const int i = lrows[k];
                    const double d = lmul[k] * urv;
                    if (wpos[i]) A.cv[wpos[i] - 1] -= d;
                    else {
                        const int t = A.cs[j] + A.cl[j]++; nzact++;
                        A.ci[t] = i; A.cv[t] = -d;
                        row_reserve(&A, i, 1); A.ri[A.rs[i] + A.rl[i]++] = j;
                    }
                }
                for (int t = A.cs[j]; t < A.cs[j] + A.cl[j]; t++) wpos[A.ci[t]] = 0;
            }
            A.cmax[j] = -1;
            lst_ins(A.chead, A.cnext, A.cprev, A.cl[j], j);
        }
        A.ract[r] = 0;
        for (int k = 0; k < nlr; k++) lst_ins(A.rhead, A.rnext, A.rprev, A.rl[lrows[k]], lrows[k]);
        rowof[c] = r; prow[np] = r; pcol[np] = c; pval[np] = piv; np++;
    }
    /* 4. a singular basis: the positions left get the logicals of the rows left */
    int nout = 0;
    char *repl = NULL;
    if (np < m) {
        /* rows left: neither a logical's row (rdone) nor pivoted in the elimination (inactive) */
        repl = xcalloc(m, 1);
        int ir = 0;
        for (int j = 0; j < m; j++) {
            if (rowof[j] >= 0) continue;
            while (ir < m && (rdone[ir] || !A.ract[ir])) ir++;
            if (ir == m) break;
            const int i = ir++;
            out[nout++] = var[j]; repl[j] = 1;
            var[j] = n + i; rowof[j] = i; A.ract[i] = 0;
            prow[np] = i; pcol[np] = j; pval[np] = 1.0; np++;
        }
        F->rank_deficient = nout;
    }
    /* 5. L by rows, and the per-row access of L columns (for the searches) */
    for (int i = 0; i < m; i++) { F->lpos[i] = -1; F->lcs_row[i] = 0; F->lcl_row[i] = 0; }
    for (int k = 0; k < F->nl; k++) { const int rr = F->lrow[k]; F->lpos[rr] = k; F->lcs_row[rr] = F->lstart[k]; F->lcl_row[rr] = F->lstart[k + 1] - F->lstart[k]; }
    const int nlz = F->lstart[F->nl];
    F->lridx = xrealloc(F->lridx, sizeof(int) * (nlz + 1)); F->lrval = xrealloc(F->lrval, sizeof(double) * (nlz + 1));
    {
        int *cnt = F->dfs_pos;
        for (int i = 0; i <= m; i++) F->lrstart[i] = 0;
        for (int p = 0; p < nlz; p++) F->lrstart[F->lidx[p] + 1]++;
        for (int i = 0; i < m; i++) F->lrstart[i + 1] += F->lrstart[i];
        for (int i = 0; i < m; i++) F->lrlen[i] = F->lrstart[i + 1] - F->lrstart[i];
        for (int i = 0; i < m; i++) cnt[i] = F->lrstart[i];
        for (int k = 0; k < F->nl; k++)
            for (int p = F->lstart[k]; p < F->lstart[k + 1]; p++) { const int i = F->lidx[p]; F->lridx[cnt[i]] = F->lrow[k]; F->lrval[cnt[i]] = F->lval[p]; cnt[i]++; }
        /* rows with entries in L, by decreasing pivot step */
        int nro = 0;
        for (int k = np - 1; k >= 0; k--) { const int rr = prow[k]; if (F->lrstart[rr + 1] > F->lrstart[rr]) F->lrorder[nro++] = rr; }
        F->nlrorder = nro;
    }
    /* 6. U: the order, the diagonal, columns and rows keyed by the pivot rows */
    if (F->ordcap < 2 * m + 16) { F->ordcap = 2 * m + 16; F->uord = xrealloc(F->uord, sizeof(int) * F->ordcap); }
    F->nord = np;
    for (int k = 0; k < np; k++) { F->uord[k] = prow[k]; F->uordpos[prow[k]] = k; F->udiag[prow[k]] = pval[k]; }
    int nu = 0;
    {
        /* drop the entries of replaced positions; translate positions to their pivot rows */
        for (int t = 0; t < T.n; t++) {
            const int j = T.j[t];
            if (repl && repl[j]) continue;      /* a replaced column: the logical has no entries in other rows */
            T.j[nu] = rowof[j]; T.r[nu] = T.r[t]; T.v[nu] = T.v[t]; nu++;
        }
    }
    {
        for (int i = 0; i < m; i++) { F->ucl[i] = 0; F->url[i] = 0; }
        for (int t = 0; t < nu; t++) { F->ucl[T.j[t]]++; F->url[T.r[t]]++; }
        const int need = 2 * nu + 8 * m + 64;
        if (F->ucap < need) { F->ucap = need; F->uci = xrealloc(F->uci, sizeof(int) * need); F->ucv = xrealloc(F->ucv, sizeof(double) * need); }
        if (F->urcap < need) { F->urcap = need; F->uri = xrealloc(F->uri, sizeof(int) * need); F->urv = xrealloc(F->urv, sizeof(double) * need); }
        int ec = 0, er = 0;
        for (int i = 0; i < m; i++) {
            F->ucs[i] = ec; F->ucc[i] = F->ucl[i] + 4; ec += F->ucc[i]; F->ucl[i] = 0;
            F->urs[i] = er; F->urc[i] = F->url[i] + 4; er += F->urc[i]; F->url[i] = 0;
        }
        F->ucend = ec; F->urend = er;
        for (int t = 0; t < nu; t++) {
            const int rc = T.j[t], rr = T.r[t];
            int p = F->ucs[rc] + F->ucl[rc]++; F->uci[p] = rr; F->ucv[p] = T.v[t];
            p = F->urs[rr] + F->url[rr]++; F->uri[p] = rc; F->urv[p] = T.v[t];
        }
    }
    /* 7. the basis reordered: the position of each column is its pivot row */
    {
        int *nv = F->dfs_pos;
        for (int j = 0; j < m; j++) nv[rowof[j]] = var[j];
        memcpy(var, nv, sizeof(int) * m);
    }
    F->nnz_l = nlz; F->nnz_u0 = nu; F->nnz_u = nu; F->nnz_r = 0;
    /* the pivots that do something in the forward solve with U (a column or a diagonal other than 1), in pivot order */
    if (F->uzcap < 2 * m + 16) { F->uzcap = 2 * m + 16; F->uz = xrealloc(F->uz, sizeof(int) * F->uzcap); }
    if (!F->uzpos) F->uzpos = xmalloc(sizeof(int) * ((size_t)m + 1));
    F->nuz = 0;
    for (int i = 0; i < m; i++) F->uzpos[i] = -1;
    for (int k = 0; k < np; k++) { const int r = F->uord[k]; if (F->ucl[r] > 0 || F->udiag[r] != 1.0) { F->uzpos[r] = F->nuz; F->uz[F->nuz++] = r; } }
    /* the work of the factorization in entries touched: the elimination's updates (counted above), and the passes
     * over the basis' entries and over the rows that every factorization makes (setting up the active matrix and the
     * count lists, the transposes of L and U, the reordering: about a dozen per row and four per entry) */
    F->factor_work += 4.0 * (nzs + nlz + nu) + 12.0 * m;
    if (!F->rstart) { F->rcapn = 256; F->rpiv = xmalloc(sizeof(int) * F->rcapn); F->rstart = xmalloc(sizeof(int) * (F->rcapn + 1)); }
    F->rstart[0] = 0;
    if (!F->ridx) { F->rcap = 4 * m + 1024; F->ridx = xmalloc(sizeof(int) * F->rcap); F->rval = xmalloc(sizeof(double) * F->rcap); }
    for (int i = 0; i < m; i++) F->wmark[i] = 0;
    free(T.r); free(T.j); free(T.v);
    free(A.cs); free(A.cl); free(A.cc); free(A.ci); free(A.cv); free(A.rs); free(A.rl); free(A.rc); free(A.ri);
    free(A.chead); free(A.cnext); free(A.cprev); free(A.rhead); free(A.rnext); free(A.rprev); free(A.cmax); free(A.cact); free(A.ract);
    free(prow); free(pcol); free(pval); free(rowof); free(rdone); free(repl);
    return nout;
}

/* ---------------------------------------------------------------- solves */
/* depth-first search from the nonzeros of x over the graph (start/len/adj keyed by row id); the reached
 * rows in postorder in F->wlist (count returned); F->wmark used and cleared by the caller */
static int dfs(SxWs *W, const SxVec *x, const int *start, const int *len, const int *adj) {
    int *mark = W->wmark, *stk = W->stack, *pos = W->dfs_pos, *outl = W->wlist;
    int nout = 0;
    for (int k = 0; k < x->cnt; k++) {
        const int r0 = x->idx[k];
        if (mark[r0]) continue;
        int top = 0; stk[0] = r0; pos[0] = 0; mark[r0] = 1;
        while (top >= 0) {
            const int r = stk[top];
            const int s = start[r], l = len[r];
            int p = pos[top];
            while (p < l && mark[adj[s + p]]) p++;
            if (p < l) {
                const int nx = adj[s + p];
                pos[top] = p + 1;
                mark[nx] = 1; top++; stk[top] = nx; pos[top] = 0;
            } else { outl[nout++] = r; top--; }
        }
    }
    return nout;
}
static void unmark(SxWs *W, int nout) { for (int k = 0; k < nout; k++) W->wmark[W->wlist[k]] = 0; }
/* the factorization's own scratch space, as a workspace */
static SxWs own_ws(SxLU *F) { SxWs W; W.m = F->m; W.stack = F->stack; W.wmark = F->wmark; W.wlist = F->wlist; W.dfs_pos = F->dfs_pos; W.nflops = 0; return W; }
void sxws_init(SxWs *W, int m) {
    W->m = m; W->stack = malloc(sizeof(int) * (2 * (size_t)m + 2)); W->wmark = calloc((size_t)m + 1, sizeof(int));
    W->wlist = malloc(sizeof(int) * ((size_t)m + 1)); W->dfs_pos = malloc(sizeof(int) * ((size_t)m + 1)); W->nflops = 0;
    if (!W->stack || !W->wmark || !W->wlist || !W->dfs_pos) { fprintf(stderr, "brisk: out of memory (simplex)\n"); exit(1); }
}
void sxws_free(SxWs *W) { free(W->stack); free(W->wmark); free(W->wlist); free(W->dfs_pos); memset(W, 0, sizeof *W); }

/* the forward solve keeps its list of nonzeros as it goes when the pivots of U with work are fewer than half the rows
 * (a basis mostly of logicals: a pass over all rows to find the nonzeros would cost more than the solve); otherwise
 * one pass at the end (cheaper than a test per update) */
#define TRACK(F) ((F)->nuz < (F)->m / 2)
/* x <- L^{-1} x */
static void ftran_l(SxLU *F, SxWs *W, SxVec *x, int hyper) {
    double *v = x->val;
    if (hyper) {
        const int nr = dfs(W, x, F->lcs_row, F->lcl_row, F->lidx);
        for (int k = nr - 1; k >= 0; k--) {
            const int r = W->wlist[k];
            const double xr = v[r];
            if (xr != 0 && F->lpos[r] >= 0) {
                const int l = F->lpos[r];
                W->nflops += F->lstart[l + 1] - F->lstart[l];
            for (int p = F->lstart[l]; p < F->lstart[l + 1]; p++) v[F->lidx[p]] -= F->lval[p] * xr;
            }
        }
        for (int k = 0; k < nr; k++) x->idx[k] = W->wlist[k];
        x->cnt = nr; unmark(W, nr);
        W->nflops += nr;
    } else if (TRACK(F)) {
        /* (the list of nonzeros kept as entries appear; one that cancels keeps 1e-300 and is pruned afterwards) */
        int *idx = x->idx, cnt = x->cnt;
        for (int l = 0; l < F->nl; l++) {
            const double xr = v[F->lrow[l]];
            if (xr == 0) continue;
            W->nflops += F->lstart[l + 1] - F->lstart[l];
            for (int p = F->lstart[l]; p < F->lstart[l + 1]; p++) {
                const int i = F->lidx[p];
                if (v[i] == 0) idx[cnt++] = i;
                v[i] -= F->lval[p] * xr;
                if (v[i] == 0) v[i] = 1e-300;
            }
        }
        x->cnt = cnt;
        W->nflops += F->nl;
    } else {
        for (int l = 0; l < F->nl; l++) {
            const double xr = v[F->lrow[l]];
            if (xr == 0) continue;
            W->nflops += F->lstart[l + 1] - F->lstart[l];
            for (int p = F->lstart[l]; p < F->lstart[l + 1]; p++) v[F->lidx[p]] -= F->lval[p] * xr;
        }
        x->cnt = -1;
        W->nflops += F->nl;
    }
}
/* x <- R_k ... R_1 x */
static void ftran_r(SxLU *F, SxWs *W, SxVec *x) {
    double *v = x->val;
    for (int e = 0; e < F->nr; e++) {
        double s = 0;
        for (int p = F->rstart[e]; p < F->rstart[e + 1]; p++) s += F->rval[p] * v[F->ridx[p]];
        if (s != 0) {
            const int r = F->rpiv[e];
            if (x->cnt >= 0 && v[r] == 0) x->idx[x->cnt++] = r;
            v[r] -= s;
            if (v[r] == 0) v[r] = 1e-300;   /* keeps the index valid; pruned at the end */
        }
    }
    W->nflops += F->rstart[F->nr];
}
/* x <- U^{-1} x */
static void ftran_u(SxLU *F, SxWs *W, SxVec *x, int hyper) {
    double *v = x->val;
    if (hyper) {
        const int nr = dfs(W, x, F->ucs, F->ucl, F->uci);
        for (int k = nr - 1; k >= 0; k--) {
            const int r = W->wlist[k];
            if (v[r] == 0) continue;
            const double xr = v[r] / F->udiag[r];
            v[r] = xr;
            W->nflops += F->ucl[r];
            for (int p = F->ucs[r]; p < F->ucs[r] + F->ucl[r]; p++) v[F->uci[p]] -= F->ucv[p] * xr;
        }
        for (int k = 0; k < nr; k++) x->idx[k] = W->wlist[k];
        x->cnt = nr; unmark(W, nr); sxv_prune(x);
        W->nflops += nr;
    } else if (TRACK(F) && x->cnt >= 0) {
        /* the pivots with an empty column and a unit diagonal (the logicals) change nothing: only the others (uz) */
        int *idx = x->idx, cnt = x->cnt;
        for (int k = F->nuz - 1; k >= 0; k--) {
            const int r = F->uz[k];
            if (r < 0 || v[r] == 0) continue;
            const double xr = v[r] / F->udiag[r];
            v[r] = xr;
            W->nflops += F->ucl[r];
            for (int p = F->ucs[r]; p < F->ucs[r] + F->ucl[r]; p++) {
                const int i = F->uci[p];
                if (v[i] == 0) idx[cnt++] = i;
                v[i] -= F->ucv[p] * xr;
                if (v[i] == 0) v[i] = 1e-300;
            }
        }
        x->cnt = cnt; sxv_prune(x);
        W->nflops += F->nuz;
    } else {
        for (int k = F->nuz - 1; k >= 0; k--) {
            const int r = F->uz[k];
            if (r < 0 || v[r] == 0) continue;
            const double xr = v[r] / F->udiag[r];
            v[r] = xr;
            W->nflops += F->ucl[r];
            for (int p = F->ucs[r]; p < F->ucs[r] + F->ucl[r]; p++) v[F->uci[p]] -= F->ucv[p] * xr;
        }
        sxv_reindex(x);
        W->nflops += F->nuz;
    }
}
void sxlu_ftran_ws(SxLU *F, SxWs *W, SxVec *x, double expdens, SxVec *spike) {
    const int m = F->m;
    if (x->cnt < 0) sxv_reindex(x);
    const int hyper = x->cnt < 0.1 * m && expdens < 0.1;
    ftran_l(F, W, x, hyper);
    if (x->cnt >= 0) sxv_prune(x);
    if (F->nr && x->cnt < 0) sxv_reindex(x);
    if (F->nr) ftran_r(F, W, x);
    if (x->cnt < 0) sxv_reindex(x);
    else sxv_prune(x);
    if (spike) sxv_copy(spike, x);
    ftran_u(F, W, x, hyper && x->cnt < 0.1 * m);
}

void sxlu_ftran(SxLU *F, SxVec *x, double expdens, SxVec *spike) {
    SxWs W = own_ws(F);
    sxlu_ftran_ws(F, &W, x, expdens, spike);
    F->nflops += W.nflops;
}
/* a depth-first search that can be advanced by steps (sxlu_ftran_part runs two at the same pace) */
typedef struct {
    const int *start, *len, *adj, *seed;
    int nseed, k, top, nout;
    int *mark, *stk, *pos, *out;
    long edges;
} Dfs;
static void dfs_init(Dfs *D, SxWs *W, const int *start, const int *len, const int *adj, const int *seed, int nseed) {
    D->start = start; D->len = len; D->adj = adj; D->seed = seed; D->nseed = nseed; D->k = 0; D->top = -1; D->nout = 0;
    D->mark = W->wmark; D->stk = W->stack; D->pos = W->dfs_pos; D->out = W->wlist; D->edges = 0;
}
/* advances by about `steps` entries examined; returns 1 when the search is complete (the reached nodes in postorder in
 * out[0..nout-1]) */
static int dfs_step(Dfs *D, long steps) {
    int *mark = D->mark, *stk = D->stk, *pos = D->pos;
    const int *adj = D->adj;
    long budget = D->edges + steps;
    for (;;) {
        if (D->top < 0) {
            while (D->k < D->nseed && mark[D->seed[D->k]]) D->k++;
            if (D->k == D->nseed) return 1;
            const int r0 = D->seed[D->k++];
            D->top = 0; stk[0] = r0; pos[0] = 0; mark[r0] = 1;
        }
        while (D->top >= 0) {
            const int r = stk[D->top];
            const int s = D->start[r], l = D->len[r];
            int p = pos[D->top];
            while (p < l && mark[adj[s + p]]) p++;
            D->edges += p - pos[D->top] + 1;
            if (p < l) {
                const int nx = adj[s + p];
                pos[D->top] = p + 1;
                mark[nx] = 1; D->top++; stk[D->top] = nx; pos[D->top] = 0;
            } else { D->out[D->nout++] = r; D->top--; }
            if (D->edges >= budget) return D->top < 0 && D->k == D->nseed;
        }
    }
}
/* the marks of an unfinished search out */
static void dfs_unmark(Dfs *D) {
    for (int k = 0; k < D->nout; k++) D->mark[D->out[k]] = 0;
    for (int t = 0; t <= D->top; t++) D->mark[D->stk[t]] = 0;
}
/* a solve wanted at a few positions only (the steepest-edge weights read B^{-1} rho_r where the entering column is
 * nonzero). L and R as in the full solve; then, in U x = y, x_i = (y_i - sum_{j after i} u_ij x_j) / u_ii depends only on
 * the entries of row i of U: the wanted positions and the rows they reach through the rows of U (a depth-first search
 * over the row lists, in postorder: each row after the rows it depends on) can be computed by dot products, and nothing
 * else. Which of the two is cheaper depends on the basis: the full solve reaches the rows that y's nonzeros reach through
 * the columns of U, the restricted one the rows the wanted positions reach through the rows of U (a fifth of the rows
 * against a few hundred on buildingenergy; the other way round on the long chains of a network basis). With a sparse y
 * both searches run at the same pace and the one that ends first is used (the work is at most twice the cheaper one's);
 * with a dense y the restricted solve is used. W2: the second search's scratch space. */
int sxlu_ftran_part(SxLU *F, SxWs *W2, SxVec *x, double expdens, const int *pat, int npat) {
    SxWs W0 = own_ws(F), *W = &W0;
    const int m = F->m;
    double *v = x->val;
    if (x->cnt < 0) sxv_reindex(x);
    const int hyper = x->cnt < 0.1 * m && expdens < 0.1;
    ftran_l(F, W, x, hyper);
    if (x->cnt >= 0) sxv_prune(x);
    if (F->nr && x->cnt < 0) sxv_reindex(x);
    if (F->nr) ftran_r(F, W, x);
    if (x->cnt >= 0) sxv_prune(x);
    const int nlr = x->cnt;
    Dfs A, B;       /* A: the rows the wanted positions depend on; B: the reach of y's nonzeros (the full solve) */
    dfs_init(&A, W, F->urs, F->url, F->uri, pat, npat);
    int full = 0;
    if (x->cnt >= 0 && x->cnt < 0.1 * m) {
        dfs_init(&B, W2, F->ucs, F->ucl, F->uci, x->idx, x->cnt);
        for (;;) {
            if (dfs_step(&A, 256)) { dfs_unmark(&B); break; }
            if (dfs_step(&B, 256)) { dfs_unmark(&A); full = 1; break; }
        }
        W->nflops += B.edges;
    } else dfs_step(&A, (long)1 << 62);
    W->nflops += A.edges;
    if (full) {
        /* the full solve on B's reach (as ftran_u's hypersparse solve) */
        const int nr = B.nout;
        long fl = nr;
        for (int k = nr - 1; k >= 0; k--) {
            const int r = B.out[k];
            if (v[r] == 0) continue;
            const double xr = v[r] / F->udiag[r];
            v[r] = xr;
            fl += F->ucl[r];
            for (int p = F->ucs[r]; p < F->ucs[r] + F->ucl[r]; p++) v[F->uci[p]] -= F->ucv[p] * xr;
        }
        for (int k = 0; k < nr; k++) { x->idx[k] = B.out[k]; B.mark[B.out[k]] = 0; }
        x->cnt = nr; sxv_prune(x);
        W->nflops += fl;
    } else {
        const int nr = A.nout;
        long fl = nr;
        for (int k = 0; k < nr; k++) {
            const int r = A.out[k];
            double sum = v[r];
            const int e = F->urs[r] + F->url[r];
            for (int p = F->urs[r]; p < e; p++) sum -= F->urv[p] * v[F->uri[p]];
            fl += F->url[r];
            v[r] = sum / F->udiag[r];
        }
        /* the other entries of y out (the marks of the search tell the computed ones) */
        if (x->cnt >= 0) { for (int k = 0; k < x->cnt; k++) { const int i = x->idx[k]; if (!A.mark[i]) v[i] = 0; } }
        else for (int i = 0; i < m; i++) if (!A.mark[i]) v[i] = 0;
        for (int k = 0; k < nr; k++) { x->idx[k] = A.out[k]; A.mark[A.out[k]] = 0; }
        x->cnt = nr;
        W->nflops += fl;
    }
    F->nflops += W0.nflops;
    return nlr;
}

/* y <- U^{-T} y */
static void btran_u(SxLU *F, SxWs *W, SxVec *y, int hyper) {
    double *v = y->val;
    if (hyper) {
        const int nr = dfs(W, y, F->urs, F->url, F->uri);
        for (int k = nr - 1; k >= 0; k--) {
            const int r = W->wlist[k];
            if (v[r] == 0) continue;
            const double yr = v[r] / F->udiag[r];
            v[r] = yr;
            W->nflops += F->url[r];
            for (int p = F->urs[r]; p < F->urs[r] + F->url[r]; p++) v[F->uri[p]] -= F->urv[p] * yr;
        }
        for (int k = 0; k < nr; k++) y->idx[k] = W->wlist[k];
        y->cnt = nr; unmark(W, nr); sxv_prune(y);
        W->nflops += nr;
    } else {
        for (int k = 0; k < F->nord; k++) {
            const int r = F->uord[k];
            if (r < 0 || v[r] == 0) continue;
            const double yr = v[r] / F->udiag[r];
            v[r] = yr;
            W->nflops += F->url[r];
            for (int p = F->urs[r]; p < F->urs[r] + F->url[r]; p++) v[F->uri[p]] -= F->urv[p] * yr;
        }
        sxv_reindex(y);
        W->nflops += F->nord;
    }
}
/* y <- R_1^T ... R_k^T y */
static void btran_r(SxLU *F, SxWs *W, SxVec *y) {
    double *v = y->val;
    for (int e = F->nr - 1; e >= 0; e--) {
        const double yr = v[F->rpiv[e]];
        if (yr == 0) continue;
        for (int p = F->rstart[e]; p < F->rstart[e + 1]; p++) {
            const int j = F->ridx[p];
            if (y->cnt >= 0 && v[j] == 0) y->idx[y->cnt++] = j;
            v[j] -= F->rval[p] * yr;
            if (v[j] == 0) v[j] = 1e-300;
        }
    }
    W->nflops += F->rstart[F->nr];
}
/* y <- L^{-T} y */
static void btran_l(SxLU *F, SxWs *W, SxVec *y, int hyper) {
    double *v = y->val;
    if (hyper) {
        const int nr = dfs(W, y, F->lrstart, F->lrlen, F->lridx);
        for (int k = nr - 1; k >= 0; k--) {
            const int r = W->wlist[k];
            const double yr = v[r];
            if (yr == 0) continue;
            W->nflops += F->lrstart[r + 1] - F->lrstart[r];
            for (int p = F->lrstart[r]; p < F->lrstart[r + 1]; p++) v[F->lridx[p]] -= F->lrval[p] * yr;
        }
        for (int k = 0; k < nr; k++) y->idx[k] = W->wlist[k];
        y->cnt = nr; unmark(W, nr); sxv_prune(y);
        W->nflops += nr;
    } else {
        for (int k = 0; k < F->nlrorder; k++) {
            const int r = F->lrorder[k];
            const double yr = v[r];
            if (yr == 0) continue;
            W->nflops += F->lrstart[r + 1] - F->lrstart[r];
            for (int p = F->lrstart[r]; p < F->lrstart[r + 1]; p++) v[F->lridx[p]] -= F->lrval[p] * yr;
        }
        sxv_reindex(y);
        W->nflops += F->nlrorder;
    }
}
void sxlu_btran(SxLU *F, SxVec *y, double expdens, SxVec *upart) {
    SxWs W0 = own_ws(F), *W = &W0;
    const int m = F->m;
    if (y->cnt < 0) sxv_reindex(y);
    const int hyper = y->cnt < 0.1 * m && expdens < 0.1;
    btran_u(F, W, y, hyper);
    if (upart) sxv_copy(upart, y);
    if (F->nr) btran_r(F, W, y);
    if (y->cnt >= 0) sxv_prune(y);
    btran_l(F, W, y, hyper && y->cnt >= 0 && y->cnt < 0.1 * m);
    F->nflops += W0.nflops;
}

/* ---------------------------------------------------------------- the Forrest-Tomlin update */
static void ucol_del(SxLU *F, int c, int i) {   /* entry of row i out of column c */
    const int s = F->ucs[c], e = s + F->ucl[c];
    for (int p = s; p < e; p++) if (F->uci[p] == i) { F->uci[p] = F->uci[e - 1]; F->ucv[p] = F->ucv[e - 1]; F->ucl[c]--; return; }
}
static void urow_del(SxLU *F, int r, int j) {
    const int s = F->urs[r], e = s + F->url[r];
    for (int p = s; p < e; p++) if (F->uri[p] == j) { F->uri[p] = F->uri[e - 1]; F->urv[p] = F->urv[e - 1]; F->url[r]--; return; }
}
static void ucompact(SxLU *F, int rows, int extra) {   /* rows: the row lists, else the column lists */
    const int m = F->m;
    int *s = rows ? F->urs : F->ucs, *l = rows ? F->url : F->ucl, *c = rows ? F->urc : F->ucc;
    int *idx = rows ? F->uri : F->uci; double *val = rows ? F->urv : F->ucv;
    long tot = 0; for (int i = 0; i < m; i++) tot += l[i] + 4;
    long cap = 2 * tot + extra + 64; const int oldcap = rows ? F->urcap : F->ucap; if (cap < oldcap) cap = oldcap;
    int *ni = xmalloc(sizeof(int) * cap); double *nv = xmalloc(sizeof(double) * cap);
    int e = 0;
    for (int i = 0; i < m; i++) { memcpy(ni + e, idx + s[i], sizeof(int) * l[i]); memcpy(nv + e, val + s[i], sizeof(double) * l[i]); s[i] = e; c[i] = l[i] + 4; e += c[i]; }
    free(idx); free(val);
    if (rows) { F->uri = ni; F->urv = nv; F->urend = e; F->urcap = (int)cap; } else { F->uci = ni; F->ucv = nv; F->ucend = e; F->ucap = (int)cap; }
}
static void urow_add(SxLU *F, int r, int j, double v) {
    if (F->url[r] == F->urc[r]) {
        const int nc = 2 * F->urc[r] + 4;
        if ((long)F->urend + nc > F->urcap) ucompact(F, 1, nc);
        memmove(F->uri + F->urend, F->uri + F->urs[r], sizeof(int) * F->url[r]); memmove(F->urv + F->urend, F->urv + F->urs[r], sizeof(double) * F->url[r]);
        F->urs[r] = F->urend; F->urc[r] = nc; F->urend += nc;
    }
    const int p = F->urs[r] + F->url[r]++;
    F->uri[p] = j; F->urv[p] = v;
}
int sxlu_update(SxLU *F, int r, SxVec *spike, SxVec *upart, double alpha) {
    const int m = F->m;
    const double urr = F->udiag[r];
    /* the row eta: eta_j = -u_rr z_j (z: the partial btran of e_r), j != r */
    if (F->nr == F->rcapn) { F->rcapn *= 2; F->rpiv = xrealloc(F->rpiv, sizeof(int) * F->rcapn); F->rstart = xrealloc(F->rstart, sizeof(int) * (F->rcapn + 1)); }
    if (upart->cnt < 0) sxv_reindex(upart);
    if (spike->cnt < 0) sxv_reindex(spike);
    int p0 = F->rstart[F->nr];
    if (p0 + upart->cnt + 1 > F->rcap) { F->rcap = 2 * (p0 + upart->cnt) + 1024; F->ridx = xrealloc(F->ridx, sizeof(int) * F->rcap); F->rval = xrealloc(F->rval, sizeof(double) * F->rcap); }
    double newd = spike->val[r];
    int p = p0;
    for (int k = 0; k < upart->cnt; k++) {
        const int j = upart->idx[k];
        if (j == r) continue;
        const double z = upart->val[j];
        if (fabs(z) < SX_TINY) continue;
        const double eta = -urr * z;
        F->ridx[p] = j; F->rval[p] = eta; p++;
        newd -= eta * spike->val[j];
    }
    F->rpiv[F->nr] = r; F->rstart[F->nr + 1] = p; F->nr++;
    F->nnz_r += p - p0;
    /* U: column r and row r out */
    F->nnz_u -= F->ucl[r] + F->url[r];
    for (int q = F->ucs[r]; q < F->ucs[r] + F->ucl[r]; q++) urow_del(F, F->uci[q], r);
    F->ucl[r] = 0;
    for (int q = F->urs[r]; q < F->urs[r] + F->url[r]; q++) ucol_del(F, F->uri[q], r);
    F->url[r] = 0;
    /* the spike as the new column r (the last pivot) */
    int ns = 0;
    for (int k = 0; k < spike->cnt; k++) { const int i = spike->idx[k]; if (i != r && fabs(spike->val[i]) >= SX_TINY) ns++; }
    if (F->ucc[r] < ns) {
        const int nc = 2 * ns + 4;
        if ((long)F->ucend + nc > F->ucap) ucompact(F, 0, nc);
        F->ucs[r] = F->ucend; F->ucc[r] = nc; F->ucend += nc;
    }
    for (int k = 0; k < spike->cnt; k++) {
        const int i = spike->idx[k];
        const double s = spike->val[i];
        if (i == r || fabs(s) < SX_TINY) continue;
        const int q = F->ucs[r] + F->ucl[r]++;
        F->uci[q] = i; F->ucv[q] = s;
        urow_add(F, i, r, s);
    }
    F->udiag[r] = newd;
    F->nnz_u += F->ucl[r];
    /* the order: r moves to the end */
    F->uord[F->uordpos[r]] = -1;
    if (F->nord == F->ordcap) {   /* compact the order */
        int e = 0; for (int k = 0; k < F->nord; k++) if (F->uord[k] >= 0) { F->uord[e] = F->uord[k]; F->uordpos[F->uord[e]] = e; e++; }
        F->nord = e;
        if (F->nord == F->ordcap) { F->ordcap *= 2; F->uord = xrealloc(F->uord, sizeof(int) * F->ordcap); }
    }
    F->uord[F->nord] = r; F->uordpos[r] = F->nord; F->nord++;
    if (F->uzpos[r] >= 0) F->uz[F->uzpos[r]] = -1;
    if (F->nuz == F->uzcap) {
        int e = 0; for (int k = 0; k < F->nuz; k++) if (F->uz[k] >= 0) { F->uz[e] = F->uz[k]; F->uzpos[F->uz[e]] = e; e++; }
        F->nuz = e;
        if (F->nuz == F->uzcap) { F->uzcap *= 2; F->uz = xrealloc(F->uz, sizeof(int) * F->uzcap); }
    }
    F->uz[F->nuz] = r; F->uzpos[r] = F->nuz; F->nuz++;
    F->nupd++;
    (void)m;
    /* stability: the new diagonal against u_rr alpha_rq */
    const double pred = urr * alpha;
    if (!(fabs(newd) > 0) || fabs(newd - pred) > 1e-8 * (1 + fabs(pred))) return 1;
    return 0;
}
