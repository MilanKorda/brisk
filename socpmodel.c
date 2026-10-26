/* socpmodel.c - an SOCP kept for re-solves after bound changes, each solve warm-started from the last one (the SOCP
 * developer, 23 October 2026; SOCP_WARMSTART.md, IMPLEMENTATION 25.7). For branch and bound on a mixed-integer SOCP.
 *
 * The problem is SeDuMi's primal form, min c'x, A x = b, x in K (free, nonnegative, second-order and rotated cones in
 * this order, socp.h). A bound lo <= x_j <= hi on any column enters as a row of its own with a new nonnegative slack:
 * x_j + s = hi, x_j - s = lo. The slacks are appended to the nonnegative block, so the node problem's column of an original
 * column j is j before the cones and j + (the number of slacks) from the cones on. The first solve is the usual one
 * (presolve and all); after an optimal solve the next one starts from its point (SocpOpts.wx, wy, wz, wlam): the original
 * columns and rows from the last point, a bound row that existed then from its own slack and multiplier, a new bound row
 * from its slack u - x_j (x_j - l) clipped at 0 and a multiplier 0. A solve that is not optimal (infeasible, stopped by the
 * cutoff, failed) leaves the stored point as it was. Measured on 123 mixed-integer SOCPs of CBLIB (tests/misocp_bb.c): the
 * warm node solves take 0.705 of the cold iterations, 0.615 with the incumbent as cutoff. */
#include "socpmodel.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

typedef struct { int col, side; double v; } SmBound;     /* side +1: x_col <= v, -1: x_col >= v (each with a slack), 0: x_col = v (none) */

struct SocpModel {
    int m, n, nf, nl, nq, nr;
    int *Ap, *Ai, *q, *r; double *Ax, *b, *c;
    SocpOpts o;
    double cutoff;
    SmBound *bd; int nb, cap;                           /* the bound rows, in their order */
    /* the last optimal point, over the node problem it was solved on: its bound rows (keys), x, y, z */
    int have, pnb; SmBound *pbd; double *px, *py, *pz;
};

static void *xmal(size_t s) { void *p = malloc(s ? s : 1); return p; }

SocpModel *socpm_create(const SocpProb *P, const SocpOpts *o) {
    if (!P || P->m < 0 || P->n < 1 || !P->Ap || !P->c) return NULL;
    SocpModel *M = calloc(1, sizeof *M);
    if (!M) return NULL;
    M->m = P->m; M->n = P->n; M->nf = P->nf; M->nl = P->nl; M->nq = P->nq; M->nr = P->nr;
    const int nz = P->Ap[P->n];
    M->Ap = xmal(sizeof(int) * (P->n + 1)); M->Ai = xmal(sizeof(int) * (nz + 1)); M->Ax = xmal(sizeof(double) * (nz + 1));
    M->b = xmal(sizeof(double) * (P->m + 1)); M->c = xmal(sizeof(double) * (P->n + 1));
    M->q = xmal(sizeof(int) * (P->nq + 1)); M->r = xmal(sizeof(int) * (P->nr + 1));
    if (!M->Ap || !M->Ai || !M->Ax || !M->b || !M->c || !M->q || !M->r) { socpm_destroy(M); return NULL; }
    memcpy(M->Ap, P->Ap, sizeof(int) * (P->n + 1)); memcpy(M->Ai, P->Ai, sizeof(int) * nz); memcpy(M->Ax, P->Ax, sizeof(double) * nz);
    for (int i = 0; i < P->m; i++) M->b[i] = P->b ? P->b[i] : 0.0;
    memcpy(M->c, P->c, sizeof(double) * P->n);
    if (P->nq) memcpy(M->q, P->q, sizeof(int) * P->nq);
    if (P->nr) memcpy(M->r, P->r, sizeof(int) * P->nr);
    if (o) M->o = *o; else socp_default_opts(&M->o);
    M->o.wx = M->o.wy = M->o.wz = NULL; M->o.wsave = 0;
    if (!(M->o.wlam > 0 && M->o.wlam < 1)) M->o.wlam = 0.99;
    M->cutoff = INFINITY;
    return M;
}

void socpm_destroy(SocpModel *M) {
    if (!M) return;
    free(M->Ap); free(M->Ai); free(M->Ax); free(M->b); free(M->c); free(M->q); free(M->r);
    free(M->bd); free(M->pbd); free(M->px); free(M->py); free(M->pz); free(M);
}

static int bound_find(const SmBound *bd, int nb, int col, int side) { for (int k = 0; k < nb; k++) if (bd[k].col == col && bd[k].side == side) return k; return -1; }

/* sets (an infinite value: removes) the bound row of one side */
static int bound_set(SocpModel *M, int j, int side, double v) {
    const int k = bound_find(M->bd, M->nb, j, side);
    if (!isfinite(v)) { if (k >= 0) { memmove(M->bd + k, M->bd + k + 1, sizeof(SmBound) * (size_t)(M->nb - k - 1)); M->nb--; } return 0; }
    if (k >= 0) { M->bd[k].v = v; return 0; }
    if (M->nb == M->cap) { const int c2 = M->cap ? 2 * M->cap : 16; SmBound *t = realloc(M->bd, sizeof(SmBound) * (size_t)c2); if (!t) return -1; M->bd = t; M->cap = c2; }
    M->bd[M->nb].col = j; M->bd[M->nb].side = side; M->bd[M->nb].v = v; M->nb++;
    return 0;
}

int socpm_set_var_bounds(SocpModel *M, int j, double lo, double hi) {
    if (!M || j < 0 || j >= M->n || isnan(lo) || isnan(hi)) return 1;
    if (lo > hi) return 1;
    /* a fixed column (lo = hi) is one equality row without a slack: two rows with slacks would force both slacks to the
     * boundary of their cone */
    if (lo == hi && isfinite(lo)) { if (bound_set(M, j, -1, -INFINITY) || bound_set(M, j, 1, INFINITY) || bound_set(M, j, 0, lo)) return -1; return 0; }
    if (bound_set(M, j, 0, INFINITY) || bound_set(M, j, -1, lo > -1e30 ? lo : -INFINITY) || bound_set(M, j, 1, hi < 1e30 ? hi : INFINITY)) return -1;
    return 0;
}

int socpm_set_cost(SocpModel *M, const double *c) {
    if (!M || !c) return 1;
    memcpy(M->c, c, sizeof(double) * M->n);
    return 0;
}

void socpm_set_cutoff(SocpModel *M, double cutoff) { if (M) M->cutoff = isnan(cutoff) ? INFINITY : cutoff; }

/* the node problem: the original columns with the bound rows' slacks after the nonnegative block */
typedef struct { SocpProb P; int *Ap, *Ai; double *Ax, *b, *c; } Node;
static int nslack(const SmBound *bd, int nb) { int s = 0; for (int k = 0; k < nb; k++) s += bd[k].side != 0; return s; }
static int node_build(const SocpModel *M, Node *N) {
    const int ns = nslack(M->bd, M->nb), n = M->n + ns, m = M->m + M->nb, e0 = M->nf + M->nl;
    const long nz = (long)M->Ap[M->n] + M->nb + (long)ns;
    N->Ap = xmal(sizeof(int) * ((size_t)n + 1)); N->Ai = xmal(sizeof(int) * ((size_t)nz + 1)); N->Ax = xmal(sizeof(double) * ((size_t)nz + 1));
    N->b = xmal(sizeof(double) * ((size_t)m + 1)); N->c = calloc((size_t)n + 1, sizeof(double));
    if (!N->Ap || !N->Ai || !N->Ax || !N->b || !N->c) return -1;
    long w = 0;
    for (int jn = 0; jn < n; jn++) {
        N->Ap[jn] = (int)w;
        const int j = jn < e0 ? jn : jn >= e0 + ns ? jn - ns : -1;
        if (j >= 0) {
            for (int p = M->Ap[j]; p < M->Ap[j + 1]; p++) { N->Ai[w] = M->Ai[p]; N->Ax[w] = M->Ax[p]; w++; }
            for (int k = 0; k < M->nb; k++) if (M->bd[k].col == j) { N->Ai[w] = M->m + k; N->Ax[w] = 1.0; w++; }
            N->c[jn] = M->c[j];
        } else {      /* the slack of the (jn - e0)-th bound row with a side */
            int k = -1; for (int t = jn - e0; t >= 0; ) { k++; if (M->bd[k].side != 0) t--; }
            N->Ai[w] = M->m + k; N->Ax[w] = M->bd[k].side > 0 ? 1.0 : -1.0; w++;
        }
    }
    N->Ap[n] = (int)w;
    memcpy(N->b, M->b, sizeof(double) * (size_t)M->m);
    for (int k = 0; k < M->nb; k++) N->b[M->m + k] = M->bd[k].v;
    SocpProb *P = &N->P;
    P->m = m; P->n = n; P->Ap = N->Ap; P->Ai = N->Ai; P->Ax = N->Ax; P->b = N->b; P->c = N->c;
    P->nf = M->nf; P->nl = M->nl + ns; P->nq = M->nq; P->q = M->q; P->nr = M->nr; P->r = M->r;
    return 0;
}
static void node_free(Node *N) { free(N->Ap); free(N->Ai); free(N->Ax); free(N->b); free(N->c); }

int socpm_solve(SocpModel *M, SocpRes *R) {
    if (!M || !R) return SOCP_INPUT;
    memset(R, 0, sizeof *R);
    Node N; memset(&N, 0, sizeof N);
    if (node_build(M, &N)) { node_free(&N); R->status = SOCP_INPUT; return SOCP_INPUT; }
    const int ns = nslack(M->bd, M->nb), n = M->n + ns, m = M->m + M->nb, e0 = M->nf + M->nl;
    SocpOpts o = M->o; o.cutoff = M->cutoff;
    double *wx = NULL, *wy = NULL, *wz = NULL;
    if (M->have) {
        /* the last point mapped to this node: original columns and rows, then the bound rows by their keys */
        const int pns = nslack(M->pbd, M->pnb);
        wx = calloc((size_t)n + 1, sizeof(double)); wz = calloc((size_t)n + 1, sizeof(double)); wy = calloc((size_t)m + 1, sizeof(double));
        if (!wx || !wy || !wz) { free(wx); free(wy); free(wz); node_free(&N); R->status = SOCP_INPUT; return SOCP_INPUT; }
        for (int j = 0; j < M->n; j++) { const int jp = j < e0 ? j : j + pns, jn = j < e0 ? j : j + ns; wx[jn] = M->px[jp]; wz[jn] = M->pz[jp]; }
        memcpy(wy, M->py, sizeof(double) * (size_t)M->m);
        for (int k = 0, sk = 0; k < M->nb; k++) {      /* sk: this bound row's slack, if it has one */
            const int kp = bound_find(M->pbd, M->pnb, M->bd[k].col, M->bd[k].side), col = M->bd[k].col, side = M->bd[k].side;
            const double xc = M->px[col < e0 ? col : col + pns];
            const int same = kp >= 0 && M->pbd[kp].v == M->bd[k].v;
            wy[M->m + k] = same ? M->py[M->m + kp] : 0.0;
            if (side == 0) continue;
            if (same) { const int skp = nslack(M->pbd, kp); wx[e0 + sk] = M->px[e0 + skp]; wz[e0 + sk] = M->pz[e0 + skp]; }
            else { wx[e0 + sk] = fmax(0.0, side > 0 ? M->bd[k].v - xc : xc - M->bd[k].v); wz[e0 + sk] = 0.0; }
            sk++;
        }
        o.wx = wx; o.wy = wy; o.wz = wz;
    }
    SocpRes Rn; memset(&Rn, 0, sizeof Rn);
    int st = socp_solve(&N.P, &o, &Rn);
    /* a warm start that broke down: the usual start, once (the warm-started iteration's failure is not the node's) */
    if (M->have && (st == SOCP_NUMERR || st == SOCP_MAXIT)) {
        socp_result_free(&Rn); memset(&Rn, 0, sizeof Rn);
        o.wx = o.wy = o.wz = NULL;
        st = socp_solve(&N.P, &o, &Rn);
    }
    free(wx); free(wy); free(wz);
    /* keep the point after an optimal solve */
    if (st == SOCP_OPTIMAL && Rn.x && Rn.y && Rn.z) {
        free(M->px); free(M->py); free(M->pz); free(M->pbd);
        M->px = Rn.x; M->py = Rn.y; M->pz = Rn.z; Rn.x = Rn.y = Rn.z = NULL;
        M->pbd = xmal(sizeof(SmBound) * (size_t)(M->nb + 1)); if (M->nb) memcpy(M->pbd, M->bd, sizeof(SmBound) * (size_t)M->nb);
        M->pnb = M->nb; M->have = 1;
        R->x = malloc(sizeof(double) * (size_t)M->n); R->z = malloc(sizeof(double) * (size_t)M->n); R->y = malloc(sizeof(double) * (size_t)M->m + 1);
        if (R->x && R->y && R->z) {
            for (int j = 0; j < M->n; j++) { const int jn = j < e0 ? j : j + ns; R->x[j] = M->px[jn]; R->z[j] = M->pz[jn]; }
            memcpy(R->y, M->py, sizeof(double) * (size_t)M->m);
        }
    } else if (Rn.x && Rn.y && Rn.z) {     /* not optimal: the solver's point over the original columns (a certificate when infeasible) */
        R->x = malloc(sizeof(double) * (size_t)M->n); R->z = malloc(sizeof(double) * (size_t)M->n); R->y = malloc(sizeof(double) * (size_t)M->m + 1);
        if (R->x && R->y && R->z) {
            for (int j = 0; j < M->n; j++) { const int jn = j < e0 ? j : j + ns; R->x[j] = Rn.x[jn]; R->z[j] = Rn.z[jn]; }
            memcpy(R->y, Rn.y, sizeof(double) * (size_t)M->m);
        }
    }
    R->status = st; R->iters = Rn.iters; R->pobj = Rn.pobj; R->dobj = Rn.dobj; R->maxerr = Rn.maxerr; R->time = Rn.time; R->lnz = Rn.lnz;
    memcpy(R->err, Rn.err, sizeof R->err);
    socp_result_free(&Rn);
    node_free(&N);
    return st;
}

struct SocpmPoint { int nb, nn, mm; SmBound *bd; double *x, *y, *z; };
SocpmPoint *socpm_save_point(const SocpModel *M) {
    if (!M || !M->have) return NULL;
    SocpmPoint *p = calloc(1, sizeof *p); if (!p) return NULL;
    const int ns = nslack(M->pbd, M->pnb);
    p->nb = M->pnb; p->nn = M->n + ns; p->mm = M->m + M->pnb;
    p->bd = xmal(sizeof(SmBound) * (size_t)(p->nb + 1)); p->x = xmal(sizeof(double) * (size_t)p->nn); p->z = xmal(sizeof(double) * (size_t)p->nn); p->y = xmal(sizeof(double) * ((size_t)p->mm + 1));
    if (!p->bd || !p->x || !p->y || !p->z) { socpm_point_free(p); return NULL; }
    if (p->nb) memcpy(p->bd, M->pbd, sizeof(SmBound) * (size_t)p->nb);
    memcpy(p->x, M->px, sizeof(double) * (size_t)p->nn); memcpy(p->z, M->pz, sizeof(double) * (size_t)p->nn); memcpy(p->y, M->py, sizeof(double) * (size_t)p->mm);
    return p;
}
void socpm_restore_point(SocpModel *M, const SocpmPoint *p) {
    if (!M || !p) return;
    double *x = xmal(sizeof(double) * (size_t)p->nn), *z = xmal(sizeof(double) * (size_t)p->nn), *y = xmal(sizeof(double) * ((size_t)p->mm + 1));
    SmBound *bd = xmal(sizeof(SmBound) * (size_t)(p->nb + 1));
    if (!x || !y || !z || !bd) { free(x); free(y); free(z); free(bd); return; }
    memcpy(x, p->x, sizeof(double) * (size_t)p->nn); memcpy(z, p->z, sizeof(double) * (size_t)p->nn); memcpy(y, p->y, sizeof(double) * (size_t)p->mm);
    if (p->nb) memcpy(bd, p->bd, sizeof(SmBound) * (size_t)p->nb);
    free(M->px); free(M->py); free(M->pz); free(M->pbd);
    M->px = x; M->py = y; M->pz = z; M->pbd = bd; M->pnb = p->nb; M->have = 1;
}
void socpm_point_free(SocpmPoint *p) { if (!p) return; free(p->bd); free(p->x); free(p->y); free(p->z); free(p); }
