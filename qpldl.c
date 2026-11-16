/* qpldl.c (the QP developer, 15 November 2026): the QP interior-point method's column-by-column LDL' factorization for
 * very sparse factors.
 *
 * Why. The shared sparse factorization (sparsechol.c) works by supernodes on dense panels: where the columns of the
 * factor are long it runs at the processor's rate, and that is where the time of a large problem goes. Where the factor
 * has two or three entries a column (a banded QP: a tridiagonal H with difference rows), a supernode is one or two columns
 * wide and the work per supernode (its panel's set-up, the list of the nodes that update it, the assembly's lookups, the
 * calls) is many times its arithmetic: 16,383 unknowns, 40,956 entries, 106,000 multiply-adds took 1 ms a factorization and
 * 0.25 ms a solve, a fortieth of the processor's rate.
 *
 * What. The row-by-row ("up-looking") LDL' factorization of Davis's LDL package (T. A. Davis, Algorithm 849: a concise
 * sparse Cholesky factorization package, ACM Trans. Math. Softw. 31 (2005) 587-591), the method of QDLDL (the
 * factorization of OSQP: B. Stellato, G. Banjac, P. Goulart, A. Bemporad, S. Boyd, OSQP: an operator splitting solver for
 * quadratic programs, Math. Program. Comput. 12 (2020) 637-672): row k of L is the solution of a sparse triangular system
 * with the rows before it, whose pattern is read from the elimination tree (here once, at the analysis: it does not
 * change from one factorization to the next); nothing but the entries of L is touched. The
 * matrix is quasi-definite (the sign of each pivot is known), so any order is a valid pivot order (R. J. Vanderbei,
 * Symmetric quasidefinite matrices, SIAM J. Optim. 5 (1995) 100-113); a pivot that loses its sign to rounding is replaced
 * by a small one of the right sign, the shared factorization's rule. The reciprocals of the pivots are kept, so a solve
 * has no division.
 *
 * When. A supernodal factorization pays a fixed price for every update of one supernode by another, whatever the
 * update's size, and does its arithmetic at the processor's rate; this one pays nothing per update and does its arithmetic
 * through index arrays. The caller decides by the factor's work per update, both read from the shared analysis
 * (lk_build in qpipm.c).
 *
 * One thread, a fixed order of the operations: the same bits whatever the thread count. */
#include <stdlib.h>
#include <string.h>
#include "qpldl.h"

struct QpLdl {
    int N;
    int *perm;                  /* perm[k]: the unknown at position k */
    signed char *sg;            /* the pivots' signs, by position */
    size_t *Up; int *Ui; double *Ux;   /* the upper triangle of P K P' by columns (the diagonal last in each) */
    int *parent;                /* the elimination tree */
    size_t *Lp; int *Li; double *Lx;   /* L below the diagonal by columns, the rows increasing */
    double *D, *Dinv;
    int *lnz, *flag, *pat;      /* work: the columns' counts so far, the row's marks and its pattern */
    int *ip;                    /* ip[i]: the position of unknown i */
    int sorted;                 /* the rows of each column of the upper triangle are increasing (qp_ldl_offset) */
    double *y;
    size_t *Rp; int *Ri, *Rc;   /* row k of L: its columns Ri[Rp[k] .. Rp[k+1]) in the order of their elimination, and each entry's rank Rc in its column (its place in Lx is Lp[column] + rank) */
    void *ib, *zb, *db;         /* the blocks the arrays above are cut from */
};

void qp_ldl_free(QpLdl *F) {
    if (!F) return;
    free(F->ib); free(F->zb); free(F->db); free(F->sg); free(F->Li); free(F->Lx); free(F->Rp); free(F->Rc); free(F->Ri); free(F);
}
double *qp_ldl_values(QpLdl *F) { return F->Ux; }
/* the place of the entry (r, c) of the matrix in the value array (an analysis made without places; SIZE_MAX: no such entry) */
size_t qp_ldl_offset(const QpLdl *F, int r, int c) {
    const int a = F->ip[r], b = F->ip[c], col = a > b ? a : b, row = a > b ? b : a;
    size_t lo = F->Up[col], hi = F->Up[col + 1] - 1;      /* (the diagonal is the column's last entry) */
    if (row == col) return hi;
    if (!F->sorted) { for (size_t p = lo; p < hi; p++) if (F->Ui[p] == row) return p; return (size_t)-1; }
    while (lo < hi) { const size_t mid = lo + (hi - lo) / 2; if (F->Ui[mid] < row) lo = mid + 1; else hi = mid; }
    return lo < F->Up[col + 1] - 1 && F->Ui[lo] == row ? lo : (size_t)-1;
}
size_t qp_ldl_nnz(const QpLdl *F) { return F->Lp[F->N]; }

QpLdl *qp_ldl_build(int N, const int *perm, size_t ne, const int *er, const int *ec, const signed char *sgn, size_t *pos, size_t *dpos) {
    QpLdl *F = calloc(1, sizeof(QpLdl)); if (!F) return NULL;
    const size_t n1 = (size_t)N + 1;
    F->N = N;
    int *ib = malloc(sizeof(int) * (6 * n1 + ne + n1)); size_t *zb = calloc(3 * n1 + 2, sizeof(size_t)); double *db = calloc(3 * n1 + ne + n1, sizeof(double));
    F->ib = ib; F->zb = zb; F->db = db; F->sg = malloc(n1);
    if (!ib || !zb || !db || !F->sg) { qp_ldl_free(F); return NULL; }
    F->perm = ib; F->parent = ib + n1; F->lnz = ib + 2 * n1; F->flag = ib + 3 * n1; F->pat = ib + 4 * n1; int *ip = ib + 5 * n1; F->ip = ip; F->Ui = ib + 6 * n1;
    F->Up = zb; F->Lp = zb + n1 + 1; size_t *nx = zb + 2 * n1 + 2;
    F->D = db; F->Dinv = db + n1; F->y = db + 2 * n1; F->Ux = db + 3 * n1;
    for (int k = 0; k < N; k++) { F->perm[k] = perm[k]; ip[perm[k]] = k; F->sg[k] = sgn[perm[k]]; }
    /* the upper triangle by columns: the entry (r, c) goes to the column of the later position; the diagonal last */
    size_t *Up = F->Up;
    for (size_t e = 0; e < ne; e++) { const int a = ip[er[e]], b = ip[ec[e]]; Up[(a > b ? a : b) + 1]++; }
    for (int k = 0; k < N; k++) Up[k + 1] += Up[k] + 1;
    for (int k = 0; k < N; k++) { const size_t d = Up[k + 1] - 1; F->Ui[d] = k; if (dpos) dpos[perm[k]] = d; nx[k] = Up[k]; }
    for (size_t e = 0; e < ne; e++) { const int a = ip[er[e]], b = ip[ec[e]]; const int c = a > b ? a : b, r = a > b ? b : a; const size_t p = nx[c]++; F->Ui[p] = r; if (pos) pos[e] = p; }
    if (!pos) {      /* (no places asked for: the caller looks them up, qp_ldl_offset; the rows of each column increasing) */
        for (int k = 0; k < N; k++) {
            int *u = F->Ui + Up[k]; const size_t len = Up[k + 1] - 1 - Up[k];
            for (size_t a = 1; a < len; a++) { const int v = u[a]; size_t b = a; while (b > 0 && u[b - 1] > v) { u[b] = u[b - 1]; b--; } u[b] = v; }
        }
        F->sorted = 1;
    }
    /* the elimination tree and the columns' counts (Davis's symbolic step: each entry (i, k), i < k, is followed up the
     * tree from i until a node already seen for row k) */
    int *parent = F->parent, *flag = F->flag, *lnz = F->lnz;
    for (int k = 0; k < N; k++) {
        parent[k] = -1; flag[k] = k; lnz[k] = 0;
        for (size_t p = Up[k]; p + 1 < Up[k + 1]; p++)
            for (int i = F->Ui[p]; flag[i] != k; i = parent[i]) { if (parent[i] < 0) parent[i] = k; lnz[i]++; flag[i] = k; }
    }
    F->Lp[0] = 0; for (int k = 0; k < N; k++) F->Lp[k + 1] = F->Lp[k] + (size_t)lnz[k];
    F->Li = malloc(sizeof(int) * (F->Lp[N] + 1)); F->Lx = malloc(sizeof(double) * (F->Lp[N] + 1));
    F->Rp = malloc(sizeof(size_t) * (n1 + 1)); F->Rc = malloc(sizeof(int) * (F->Lp[N] + 1)); F->Ri = malloc(sizeof(int) * (F->Lp[N] + 1));
    if (!F->Li || !F->Lx || !F->Rp || !F->Rc || !F->Ri) { qp_ldl_free(F); return NULL; }
    /* The rows of L, once: the pattern of row k is found from the tree as Davis's factorization finds it at every row (the
     * entries of column k of the upper triangle followed up the tree, in an order in which each column comes after those
     * it depends on), and kept with the place of each entry in its column. The pattern does not change from one
     * factorization to the next, so the factorization below only does arithmetic, in the same order. */
    {   int *pat = F->pat, *cnt = lnz; size_t r = 0;
        for (int k = 0; k < N; k++) cnt[k] = 0;
        F->Rp[0] = 0;
        for (int k = 0; k < N; k++) {
            int top = N; flag[k] = k;
            for (size_t p = Up[k]; p + 1 < Up[k + 1]; p++) {
                int i = F->Ui[p], len = 0;
                for (; flag[i] != k; i = parent[i]) { pat[len++] = i; flag[i] = k; }
                while (len > 0) pat[--top] = pat[--len];
            }
            for (; top < N; top++) { const int i = pat[top]; const int c = cnt[i]++; F->Ri[r] = i; F->Rc[r] = c; F->Li[F->Lp[i] + (size_t)c] = k; r++; }
            F->Rp[k + 1] = r;
        }
    }
    return F;
}

/* (15 November 2026) The plan of a factorization in the order perm, before it is built: the elimination tree and the
 * columns' counts of the matrix given as for qp_ldl_build, then perm composed with the tree's postorder (each subtree's
 * columns contiguous, a parent after its children: the same factor, with the fundamental supernodes contiguous, as the
 * shared analysis orders it), *flops the sum of the squared column lengths with the diagonal, and *ubound an upper
 * bound on the number of updates of one fundamental supernode by another (the sum of the supernodes' rows below them).
 * 0, or 1 when memory is short (perm unchanged). */
int qp_ldl_plan(int N, int *perm, size_t ne, const int *er, const int *ec, double *flops, double *ubound, double *entries) {
    const size_t n1 = (size_t)N + 1;
    int *ib = malloc(sizeof(int) * (8 * n1 + ne)); size_t *Up = calloc(2 * n1 + 2, sizeof(size_t));
    if (!ib || !Up) { free(ib); free(Up); return 1; }
    int *ip = ib, *parent = ib + n1, *lnz = ib + 2 * n1, *flag = ib + 3 * n1, *head = ib + 4 * n1, *next = ib + 5 * n1, *post = ib + 6 * n1, *stack = ib + 7 * n1, *Ui = ib + 8 * n1;
    size_t *nx = Up + n1 + 1;
    for (int k = 0; k < N; k++) ip[perm[k]] = k;
    for (size_t e = 0; e < ne; e++) { const int a = ip[er[e]], b = ip[ec[e]]; Up[(a > b ? a : b) + 1]++; }
    for (int k = 0; k < N; k++) { Up[k + 1] += Up[k]; nx[k] = Up[k]; }
    for (size_t e = 0; e < ne; e++) { const int a = ip[er[e]], b = ip[ec[e]]; Ui[nx[a > b ? a : b]++] = a > b ? b : a; }
    for (int k = 0; k < N; k++) {
        parent[k] = -1; flag[k] = k; lnz[k] = 0;
        for (size_t p = Up[k]; p < Up[k + 1]; p++)
            for (int i = Ui[p]; flag[i] != k; i = parent[i]) { if (parent[i] < 0) parent[i] = k; lnz[i]++; flag[i] = k; }
    }
    /* the postorder: the children of a node in increasing order, the roots in increasing order */
    for (int k = 0; k < N; k++) head[k] = -1;
    for (int k = N - 1; k >= 0; k--) if (parent[k] >= 0) { next[k] = head[parent[k]]; head[parent[k]] = k; }
    int np = 0;
    for (int r = 0; r < N; r++) {
        if (parent[r] >= 0) continue;
        int top = 0; stack[0] = r;
        while (top >= 0) {
            const int v = stack[top], c = head[v];
            if (c >= 0) { head[v] = next[c]; stack[++top] = c; }
            else { post[np++] = v; top--; }
        }
    }
    /* the counts and the supernodes in the new order (flag: a node's new place) */
    for (int k = 0; k < N; k++) flag[post[k]] = k;
    /* (15 November 2026, evening) the bound on the updates of a supernode: the supernodes its rows below it lie in are
     * among its ancestors in the tree of supernodes, so they number at most its depth there as well as its rows. With
     * the rows alone a dense last block made the bound 15 times the count (RQP_100, QP: 30,000 against 2,000), the
     * factor was built, counted and thrown away, 6 to 10 % of the solve */
    double fl = 0, ub = 0, nzl = 0;
    int *snof = head, *sdepth = next, *slast = stack, ns = 0;      /* (free after the postorder: the supernode of a place, a supernode's depth, its last place) */
    for (int k = 0; k < N; k++) {
        const int v = post[k]; const double c = (double)lnz[v] + 1.0; fl += c * c; nzl += c;
        const int last = k + 1 == N || !(parent[v] >= 0 && flag[parent[v]] == k + 1 && lnz[v] == lnz[post[k + 1]] + 1);     /* (the supernode ends here) */
        snof[k] = ns;
        if (last) slast[ns++] = k;
    }
    for (int s_ = ns - 1; s_ >= 0; s_--) {      /* (a parent comes later in the postorder: its depth is known) */
        const int v = post[slast[s_]], pv = parent[v];
        sdepth[s_] = pv >= 0 ? sdepth[snof[flag[pv]]] + 1 : 0;
        ub += lnz[v] < sdepth[s_] ? lnz[v] : sdepth[s_];
    }
    for (int k = 0; k < N; k++) stack[k] = perm[post[k]];
    for (int k = 0; k < N; k++) perm[k] = stack[k];
    if (flops) *flops = fl;
    if (ubound) *ubound = ub;
    if (entries) *entries = nzl;
    free(ib); free(Up);
    return 0;
}
/* the number of updates of one fundamental supernode by another in a built factor (the changes of supernode along the
 * rows below each supernode's last column): the count that the shared analysis gives as schol_nupdates */
double qp_ldl_nupdates(const QpLdl *F) {
    const int N = F->N; int *sn = malloc(sizeof(int) * ((size_t)N + 1)); if (!sn) return 0;
    int s_ = 0;
    for (int j = 0; j < N; j++) { if (j > 0 && !(F->parent[j - 1] == j && F->Lp[j] - F->Lp[j - 1] == F->Lp[j + 1] - F->Lp[j] + 1)) s_++; sn[j] = s_; }
    double u = 0;
    for (int j = 0; j < N; j++) if (j + 1 == N || sn[j + 1] != sn[j]) {
        int last = sn[j];
        for (size_t p = F->Lp[j]; p < F->Lp[j + 1]; p++) { const int t = sn[F->Li[p]]; if (t != last) { u += 1; last = t; } }
    }
    free(sn);
    return u;
}
double qp_ldl_flops(const QpLdl *F) { double fl = 0; for (int j = 0; j < F->N; j++) { const double c = (double)(F->Lp[j + 1] - F->Lp[j]) + 1.0; fl += c * c; } return fl; }

/* (the two functions below start on a 64-byte boundary: their inner loops are a few instructions long, and their speed
 * followed where the linker happened to put them, 7 to 9 % between two builds that differed only in other files) */
#if defined(__GNUC__)
#define QP_LDL_HOT __attribute__((aligned(64)))
#else
#define QP_LDL_HOT
#endif
QP_LDL_HOT int qp_ldl_factor(QpLdl *F, double dynpiv, int *nrep, double tinypiv) {
    const int N = F->N;
    const size_t *Up = F->Up, *Lp = F->Lp, *Rp = F->Rp; const int *Ui = F->Ui, *Li = F->Li, *Ri = F->Ri, *Rc = F->Rc; const double *Ux = F->Ux;
    double *Lx = F->Lx, *D = F->D, *Dinv = F->Dinv, *y = F->y;
    for (int k = 0; k < N; k++) {
        /* row k of L: the solution of the triangular system with the rows before it, column by column of its pattern */
        const size_t pd = Up[k + 1] - 1;
        for (size_t p = Up[k]; p < pd; p++) y[Ui[p]] += Ux[p];
        double d = Ux[pd];
        for (size_t q = Rp[k]; q < Rp[k + 1]; q++) {
            const int i = Ri[q]; const size_t p0 = Lp[i], p2 = p0 + (size_t)Rc[q]; const double yi = y[i]; y[i] = 0;
            for (size_t p = p0; p < p2; p++) y[Li[p]] -= Lx[p] * yi;
            const double l = yi * Dinv[i];
            d -= l * yi;
            Lx[p2] = l;
        }
        if (tinypiv > 0) {      /* (a positive definite matrix: a pivot that has lost all but tinypiv of its diagonal entry is dropped) */
            if (!(d > tinypiv * Ux[pd]) || !(d > 0)) { d = 1e128; if (nrep) (*nrep)++; }
        } else {
            const double s = F->sg[k];
            if (!(d * s > 0)) { if (!(dynpiv > 0)) return 1; d = s * dynpiv; if (nrep) (*nrep)++; }
        }
        D[k] = d; Dinv[k] = 1.0 / d;
    }
    return 0;
}

QP_LDL_HOT void qp_ldl_solve(const QpLdl *F, double *b, double *x) {
    const int N = F->N; const int *perm = F->perm, *Li = F->Li; const size_t *Lp = F->Lp; const double *Lx = F->Lx, *Dinv = F->Dinv;
    for (int k = 0; k < N; k++) x[k] = b[perm[k]];
    for (int j = 0; j < N; j++) { const double xj = x[j]; for (size_t p = Lp[j]; p < Lp[j + 1]; p++) x[Li[p]] -= Lx[p] * xj; }
    for (int j = N - 1; j >= 0; j--) { double s = x[j] * Dinv[j]; for (size_t p = Lp[j]; p < Lp[j + 1]; p++) s -= Lx[p] * x[Li[p]]; x[j] = s; }
    for (int k = 0; k < N; k++) b[perm[k]] = x[k];
}
