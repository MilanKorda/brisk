/* simplex_pre.c - the presolve of the simplex method, and the postsolve that gives back a basis of the problem as read.
 *
 * The problem (LpProb, taken as a minimization) is reduced by these rules, repeated while they find something:
 *   - an empty row is dropped (infeasible if 0 is outside its bounds);
 *   - a row with one entry becomes bounds on its column;
 *   - a fixed column (l = u) and an empty column (at the bound its cost prefers) are substituted;
 *   - dual fixing: a column whose decrease (increase) never hurts a row, with a cost that prefers it, goes to that bound;
 *   - a column with one entry in an equality row, free or implied free (the row and the other columns' bounds keep it
 *     within its own bounds), is defined by the row: row and column dropped, the row's other columns take its cost;
 *   - the same in an inequality whose bound the column's cost makes binding at every optimum (the row always gives the
 *     column's bound in the direction its cost prefers, and keeps it within its other bound): the row is an equation;
 *   - otherwise a column with one entry and cost zero is a slack of its row: dropped, the row's bounds widen by its range;
 *   - parallel rows (proportional entries) become one row with the intersection of their bounds, and parallel columns
 *     (proportional entries and costs) one column for their sum (when the others find nothing more);
 *   - a doubleton equation a x_j + b x_k = r: x_j is substituted by (r - b x_k) / a (its bounds become bounds on x_k,
 *     its other entries are added to column k's);
 *   - activity bounds: a row that its columns' bounds keep within its bounds is dropped; a row whose bound equals the
 *     extreme of its activity forces each of its columns to the bound that attains it.
 * The matrix is kept by columns and by rows with room to change (the doubleton substitution adds entries).
 *
 * The postsolve restores, in the reverse order, the values of the dropped columns and a basic status for each dropped
 * row or column, with dual values where a choice depends on them (which of a forcing row's columns is basic, whether a
 * singleton row is active). Each record keeps the columns' entries and costs as they were when it was made, so these
 * dual values are those of the problem at that stage. The caller then solves the problem as read from that basis: it
 * is optimal, or a few iterations finish (the postsolve need not be exact for the result to be). */
#include "simplex_pre.h"
#include "simplex.h"
#include "lpsolve.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>
/* (27 October 2026) the settings through a table of the BRISK_ and SX entries of the environment (simplex_lu.c) */
const char *sx_getenv(const char *name);
void sx_env_refresh(void);
#define getenv(n) sx_getenv(n)

#define PINF 1e30
#define PHUGE 1e29
#define FTOL 1e-9               /* feasibility tolerance of the reductions */
#define ZTOL 1e-12              /* entries below this (relative) vanish when the substitution sums them */

static void *xmalloc(size_t n) { void *p = malloc(n ? n : 1); if (!p) { fprintf(stderr, "brisk: out of memory (presolve)\n"); exit(1); } return p; }
static void *xcalloc(size_t n, size_t s) { void *p = calloc(n ? n : 1, s ? s : 1); if (!p) { fprintf(stderr, "brisk: out of memory (presolve)\n"); exit(1); } return p; }
static void *xrealloc(void *q, size_t n) { void *p = realloc(q, n ? n : 1); if (!p) { fprintf(stderr, "brisk: out of memory (presolve)\n"); exit(1); } return p; }

enum { R_EMPTYROW = 1, R_REDROW, R_SINGROW, R_FIXCOL, R_FREESING, R_FORCING, R_DOUBLETON, R_AGG, R_SLACK, R_PARROW, R_PARCOL, R_COSTSING, R_AGGB, R_SLACK2, R_SLACKREST, R_AGGI, R_ZFREE, R_ROWDIFF, R_NEGL };

struct SxPre {
    int m, n;                       /* the problem as read */
    int mr, nr;                     /* the reduced problem */
    int *rmap, *cmap;               /* reduced -> original */
    double *l, *u, *rl, *ru;        /* the bounds as read */
    int *Ap, *Ai; double *Ax, *c;   /* the columns and costs as read (as minimization): records of unchanged columns */
    /* the records */
    int nrec, caprec;
    int *rtype; long *ristart, *rdstart;
    int *ibuf; double *dbuf; long ni, nd, capi, capd;
    /* (24 October 2026) the open run of slack columns of one row (slack_singleton): run_i the row (-1: none), its other
     * active columns and entries (run_col, run_val, run_n; run_pos: a column's place in them, -1 when not there) */
    int run_i, run_n, run_cap, *run_col, *run_pos; double *run_val;
    int nsparsify;                  /* (31 October 2026) equations replaced by their difference with another (sparsify_rows) */
};
int sxpre_nsparsify(const SxPre *P) { return P ? P->nsparsify : 0; }
static void run_close(SxPre *P);

static void rec_begin(SxPre *P, int type) {
    if (P->run_i >= 0 && type != R_SLACK2 && type != R_SLACKREST) run_close(P);    /* (any other record ends the run) */
    if (P->nrec == P->caprec) {
        P->caprec = 2 * P->caprec + 64;
        P->rtype = xrealloc(P->rtype, sizeof(int) * P->caprec); P->ristart = xrealloc(P->ristart, sizeof(long) * (P->caprec + 1));
        P->rdstart = xrealloc(P->rdstart, sizeof(long) * (P->caprec + 1));
    }
    P->rtype[P->nrec] = type; P->ristart[P->nrec] = P->ni; P->rdstart[P->nrec] = P->nd; P->nrec++;
}
static void rec_i(SxPre *P, int v) { if (P->ni == P->capi) { P->capi = 2 * P->capi + 1024; P->ibuf = xrealloc(P->ibuf, sizeof(int) * P->capi); } P->ibuf[P->ni++] = v; }
static void rec_d(SxPre *P, double v) { if (P->nd == P->capd) { P->capd = 2 * P->capd + 1024; P->dbuf = xrealloc(P->dbuf, sizeof(double) * P->capd); } P->dbuf[P->nd++] = v; }
/* the end of a run of slack columns: the row's other active columns and entries as at the run's last slack (the
 * postsolve, undoing the records in reverse, meets this record first and sums the row's rest once) */
static void run_close(SxPre *P) {
    const int i = P->run_i; if (i < 0) return;
    P->run_i = -1;
    rec_begin(P, R_SLACKREST); rec_i(P, i); rec_i(P, P->run_n);
    for (int t = 0; t < P->run_n; t++) { rec_i(P, P->run_col[t]); rec_d(P, P->run_val[t]); P->run_pos[P->run_col[t]] = -1; }
    P->run_n = 0;
}

/* (27 October 2026, the night) the calling thread's stop flag: the dual simplex method beside the interior-point method
 * (main.c's default LP route) is stopped when that method ends first, and the presolve and the symmetry search did not
 * look at the flag: osa-60 (LP) went on for 0.28 s after the stop, neos-4984115-canche's presolve (LP) takes 9 s */
static __thread const volatile int *g_sxpre_stop;
void sxpre_set_stop(const volatile int *stop) { g_sxpre_stop = stop; }
const volatile int *sxpre_get_stop(void) { return g_sxpre_stop; }
#define SXPRE_STOPPED() (brisk_flag_get(g_sxpre_stop))
/* (5 November 2026, experiments) the reduced problem's rows and columns as indices of the problem presolved */
void sxpre_maps(const SxPre *P, const int **rmap, const int **cmap) { *rmap = P->rmap; *cmap = P->cmap; }
void sxpre_free(SxPre *P) {
    if (!P) return;
    free(P->rmap); free(P->cmap); free(P->l); free(P->u); free(P->rl); free(P->ru);
    free(P->Ap); free(P->Ai); free(P->Ax); free(P->c);
    free(P->rtype); free(P->ristart); free(P->rdstart); free(P->ibuf); free(P->dbuf);
    free(P->run_col); free(P->run_val); free(P->run_pos);
    free(P);
}

/* ---------------------------------------------------------------- the working problem */
typedef struct {
    int m, n;
    int **ci; double **cv; int *clen, *ccap;   /* columns: rows and values (entries of dropped rows are kept, skipped) */
    int **ri; double **rv; int *rlen, *rcap;   /* rows: columns and values */
    double *c, *l, *u, *rl, *ru, off;
    double *rmag;                               /* (4 November 2026) the magnitude of each row's sides: |side| as read plus every
                                                 * |term| moved into them, the scale of their rounding (the empty-row tests) */
    char *rowon, *colon;
    char *cmod;                                 /* the column's entries or cost changed since it was read */
    long *crec_i, *crec_d; int *cver, *crec_ver, *crec_skip;   /* the column's changes counted; its last full record (rec_col) */
    int *rcnt, *ccnt;                           /* entries in active rows and columns */
    int *ipool; double *vpool; size_t npool;    /* (the lists' first blocks, one allocation each: 27 October 2026) */
    /* (2 November 2026) the position of an entry in a long row (at least RIDX_MIN entries), by an open-addressing table of
     * (row, column) -> position, kept by wrow_push and wrow_del: substitutions into rows of tens of thousands of entries
     * searched them once per entry added (neos-4984115-canche, LP: 541,771 searches of rows of 68,000 entries, 9.5 of its
     * 11 s of presolve) */
    char *ridx; long *hkey; int *hpos; long hcap, hcnt, hdel;
} W;
#define RIDX_MIN 256

/* (27 October 2026) the rows' and columns' lists start in two blocks (ipool, vpool), cut at the setup; a list that grows
 * past its room moves to an allocation of its own. Before, each list was two allocations of its own: about 800 per
 * presolve on sc205 (LP), whose allocations and frees were 5 % of its solve */
static int win_pool(const W *w, const int *p) { return w->ipool && p >= w->ipool && p < w->ipool + w->npool; }
static void wpush(const W *w, int **idx, double **val, int *len, int *cap, int k, int t, double v) {
    if (len[k] == cap[k]) {
        const int nc = 2 * cap[k] + 4;
        if (win_pool(w, idx[k])) {
            int *ni = xmalloc(sizeof(int) * nc); double *nv = xmalloc(sizeof(double) * nc);
            memcpy(ni, idx[k], sizeof(int) * len[k]); memcpy(nv, val[k], sizeof(double) * len[k]);
            idx[k] = ni; val[k] = nv;
        } else { idx[k] = xrealloc(idx[k], sizeof(int) * nc); val[k] = xrealloc(val[k], sizeof(double) * nc); }
        cap[k] = nc;
    }
    idx[k][len[k]] = t; val[k][len[k]] = v; len[k]++;
}
static int wfind(int *const *idx, const int *len, int k, int t) { for (int p = 0; p < len[k]; p++) if (idx[k][p] == t) return p; return -1; }
static void wdel(int **idx, double **val, int *len, int k, int p) { len[k]--; idx[k][p] = idx[k][len[k]]; val[k][p] = val[k][len[k]]; }
/* the long rows' position table: key r * n + c (-1 empty, -2 deleted); its own growth at half load */
static long hslot(const W *w, long key) { unsigned long h = (unsigned long)key * 0x9E3779B97F4A7C15UL; return (long)(h >> 20) & (w->hcap - 1); }
static void hput(W *w, long key, int pos);
static void hgrow(W *w) {
    long oc = w->hcap; long *ok = w->hkey; int *op = w->hpos;
    w->hcap = oc ? 2 * oc : 1 << 16; while (w->hcnt * 4 > w->hcap) w->hcap *= 2;
    w->hkey = xmalloc(sizeof(long) * (size_t)w->hcap); w->hpos = xmalloc(sizeof(int) * (size_t)w->hcap);
    for (long t = 0; t < w->hcap; t++) w->hkey[t] = -1;
    w->hcnt = 0; w->hdel = 0;
    for (long t = 0; t < oc; t++) if (ok[t] >= 0) hput(w, ok[t], op[t]);
    free(ok); free(op);
}
static void hput(W *w, long key, int pos) {
    if (2 * (w->hcnt + w->hdel + 1) > w->hcap) hgrow(w);
    long t = hslot(w, key), d = -1;
    while (w->hkey[t] != -1) { if (w->hkey[t] == key) { w->hpos[t] = pos; return; } if (w->hkey[t] == -2 && d < 0) d = t; t = (t + 1) & (w->hcap - 1); }
    if (d >= 0) { t = d; w->hdel--; }
    w->hkey[t] = key; w->hpos[t] = pos; w->hcnt++;
}
static long hfind(const W *w, long key) {
    if (!w->hcap) return -1;
    long t = hslot(w, key);
    while (w->hkey[t] != -1) { if (w->hkey[t] == key) return t; t = (t + 1) & (w->hcap - 1); }
    return -1;
}
static void hdelk(W *w, long key) { const long t = hfind(w, key); if (t >= 0) { w->hkey[t] = -2; w->hcnt--; w->hdel++; } }
/* the row lists' changes, with the table of the long rows */
static void wrow_push(W *w, int r, int k, double v) {
    wpush(w, w->ri, w->rv, w->rlen, w->rcap, r, k, v);
    if (w->ridx[r]) hput(w, (long)r * w->n + k, w->rlen[r] - 1);
    else if (w->rlen[r] >= RIDX_MIN) { w->ridx[r] = 1; for (int p = 0; p < w->rlen[r]; p++) hput(w, (long)r * w->n + w->ri[r][p], p); }
}
static void wrow_del(W *w, int r, int p) {
    if (w->ridx[r]) {
        hdelk(w, (long)r * w->n + w->ri[r][p]);
        if (p != w->rlen[r] - 1) hput(w, (long)r * w->n + w->ri[r][w->rlen[r] - 1], p);
    }
    wdel(w->ri, w->rv, w->rlen, r, p);
}
static void ridx_build(W *w, int r) { w->ridx[r] = 1; for (int p = 0; p < w->rlen[r]; p++) hput(w, (long)r * w->n + w->ri[r][p], p); }
static int wrow_find(W *w, int r, int k) {
    if (!w->ridx[r] && w->rlen[r] >= RIDX_MIN) ridx_build(w, r);
    if (w->ridx[r]) { const long t = hfind(w, (long)r * w->n + k); return t < 0 ? -1 : w->hpos[t]; }
    return wfind(w->ri, w->rlen, r, k);
}

/* (27 October 2026, the night) a row forces its columns to the bounds of its extreme activity only when the points this
 * removes lie within each column's tolerance of those bounds: they are the points whose activity lies between the row's
 * side and the extreme activity (gap apart), where column j is at most gap / |a_ij| from its bound. The test of the gap
 * against the row's tolerance alone depended on the row's scale: dell (LP, MIPLIB 2017 collection) reached the row
 * -1.3e-10 x84 + 1.6e-11 x530 >= 0 (the file's 8.424954716 against 1/0.118695, left after two substitutions), whose
 * largest activity 1.6e-11 was within 1e-9 of its side; x530 was fixed at 1 although x84 = x530 = 0 meet the row, and
 * the presolve declared the LP infeasible (Gurobi: optimal, -2302) */
static int forcing_tight(const W *w, int i, int side, double gap) {
    if (gap <= 0) return 1;
    for (int p = 0; p < w->rlen[i]; p++) {
        const int j = w->ri[i][p];
        if (!w->colon[j]) continue;
        const double a = w->rv[i][p], v = (side > 0) == (a > 0) ? w->u[j] : w->l[j];
        if (gap > fabs(a) * FTOL * (1 + fabs(v))) return 0;
    }
    return 1;
}
static void drop_row(W *w, int i) {
    w->rowon[i] = 0;
    for (int p = 0; p < w->rlen[i]; p++) if (w->colon[w->ri[i][p]]) w->ccnt[w->ri[i][p]]--;
}
/* fix column j at value v: the rows' bounds take its part, the objective its cost */
static void drop_col(W *w, int j, double v) {
    w->colon[j] = 0;
    w->off += w->c[j] * v;
    for (int p = 0; p < w->clen[j]; p++) {
        const int i = w->ci[j][p];
        if (!w->rowon[i]) continue;
        w->rcnt[i]--;
        const double d = w->cv[j][p] * v;
        if (w->rl[i] > -PHUGE) w->rl[i] -= d;
        if (w->ru[i] < PHUGE) w->ru[i] -= d;
        w->rmag[i] += fabs(d);
    }
}
/* the status of a column fixed at v, relative to its bounds at that time (l, u) and as read (l0, u0): at a bound that
 * a row gave, the column is nonbasic at that bound here; that row's record makes it basic if the row is active */
static int fixstat(double v, double l, double u, double l0, double u0) {
    if (l0 == u0) return SX_FIXED;
    if (l > -PHUGE && fabs(v - l) <= 1e-12 * (1 + fabs(l))) return SX_LOWER;
    if (u < PHUGE && fabs(v - u) <= 1e-12 * (1 + fabs(u))) return SX_UPPER;
    return SX_ZERO;
}
static void rec_col(SxPre *P, W *w, int j, int skip);
#define ISFIXED(w, j) ((w)->l[j] == (w)->u[j] || ((w)->l[j] > -PHUGE && fabs((w)->u[j] - (w)->l[j]) <= 1e-12 * (1 + fabs((w)->l[j]))))
/* a row with one active entry: bounds on its column (*jo), the row dropped; 1 when those bounds are empty */
static int sing_row(W *w, SxPre *P, int i, int *jo) {
    int j = -1; double a = 0;
    for (int p = 0; p < w->rlen[i]; p++) if (w->colon[w->ri[i][p]]) { j = w->ri[i][p]; a = w->rv[i][p]; break; }
    *jo = j;
    const double lo = a > 0 ? (w->rl[i] > -PHUGE ? w->rl[i] / a : -PINF) : (w->ru[i] < PHUGE ? w->ru[i] / a : -PINF);
    const double up = a > 0 ? (w->ru[i] < PHUGE ? w->ru[i] / a : PINF) : (w->rl[i] > -PHUGE ? w->rl[i] / a : PINF);
    int fromrow = 0;
    if (lo > w->l[j]) fromrow |= 1;
    if (up < w->u[j]) fromrow |= 2;
    if (w->rl[i] == w->ru[i]) fromrow |= 4;      /* (the row an equation now: its bounds as read may be another's) */
    rec_begin(P, R_SINGROW); rec_i(P, i); rec_i(P, j); rec_i(P, fromrow); rec_d(P, a);
    rec_d(P, fromrow & 1 ? lo : w->l[j]); rec_d(P, fromrow & 2 ? up : w->u[j]);
    rec_col(P, w, j, i);
    if (fromrow & 1) w->l[j] = lo;
    if (fromrow & 2) w->u[j] = up;
    if (w->l[j] > w->u[j]) {
        if (w->l[j] > w->u[j] + FTOL * (1 + fabs(w->u[j]) + w->rmag[i] / fabs(a))) return 1;   /* (4 November 2026: the side's rounding, rmag, carried to the bound) */
        const double mid = 0.5 * (w->l[j] + w->u[j]); w->l[j] = w->u[j] = mid;
    }
    drop_row(w, i);
    return 0;
}
/* a column whose bounds fix it */
static void fix_col(W *w, SxPre *P, int j) {
    const double v = w->l[j];
    rec_begin(P, R_FIXCOL); rec_i(P, j); rec_i(P, fixstat(v, w->l[j], w->u[j], P->l[j], P->u[j])); rec_d(P, v);
    drop_col(w, j, v);
}
/* 21 October 2026: the cheap reductions to their end, through a queue: empty rows, rows with one entry, columns fixed by
 * their bounds, empty columns at the bound their cost prefers. A full pass moves a chain of these by one link when the
 * rows come in the wrong order, and the passes end when they remove little (the rule below), so a chain could be left
 * half done: irish-electricity kept rows x_j <= 0 with x_j >= 0, a variable fixed at zero through a row, which leaves
 * the interior-point method no strictly feasible point (it stalled at a gap of 5.6e-4 with both sides feasible to 1e-9);
 * HiGHS's basic rules on that output removed 2,581 rows and 3,803 columns more. Each item costs its entries; returns the
 * rows and columns removed, *status 1 when a row or a column has no feasible value. */
static int cascade(W *w, SxPre *P, int *nsing, int *nfix, int *nempty, int *status, char *why, size_t nwhy) {
    const int m = w->m, n = w->n;
    int *qr = xmalloc(sizeof(int) * (m + 1)), *qc = xmalloc(sizeof(int) * (n + 1)), nr = 0, nc = 0, done = 0;
    char *inr = xcalloc(m + 1, 1), *inc = xcalloc(n + 1, 1);
    for (int i = 0; i < m; i++) if (w->rowon[i] && w->rcnt[i] <= 1) { qr[nr++] = i; inr[i] = 1; }
    for (int j = 0; j < n; j++) if (w->colon[j] && (ISFIXED(w, j) || w->ccnt[j] == 0)) { qc[nc++] = j; inc[j] = 1; }
    while ((nr || nc) && *status == 0) {
        if (nc) {
            const int j = qc[--nc]; inc[j] = 0;
            if (!w->colon[j]) continue;
            if (ISFIXED(w, j)) { fix_col(w, P, j); (*nfix)++; }
            else if (w->ccnt[j] == 0) {        /* (as the pass: the bound the cost prefers, else zero; unbounded left to it) */
                const double cj = w->c[j]; double v; int st;
                if (cj > 0 || (cj == 0 && w->l[j] > -PHUGE)) { if (w->l[j] <= -PHUGE) continue; v = w->l[j]; st = P->l[j] == P->u[j] ? SX_FIXED : SX_LOWER; }
                else if (cj < 0 || w->u[j] < PHUGE) { if (w->u[j] >= PHUGE) continue; v = w->u[j]; st = P->l[j] == P->u[j] ? SX_FIXED : SX_UPPER; }
                else { v = 0.0; st = SX_ZERO; }
                rec_begin(P, R_FIXCOL); rec_i(P, j); rec_i(P, st); rec_d(P, v);
                drop_col(w, j, v); (*nempty)++;
            } else continue;
            done++;
            for (int p = 0; p < w->clen[j]; p++) { const int i = w->ci[j][p]; if (w->rowon[i] && w->rcnt[i] <= 1 && !inr[i]) { qr[nr++] = i; inr[i] = 1; } }
            continue;
        }
        const int i = qr[--nr]; inr[i] = 0;
        if (!w->rowon[i] || w->rcnt[i] > 1) continue;
        if (w->rcnt[i] == 0) {
            /* (4 November 2026) relative to the magnitude of the terms the sides took (rmag), not to what is left of them: the
             * side of a row whose columns were all fixed is a difference of those terms, and its rounding is theirs.
             * gasprod1-1 (LP) with its rows multiplied by 10^u, u in [-2, 2] (a copy that must cost the same): a row left
             * empty at a side of the order of its terms' rounding was declared infeasible, the problem as read was then
             * solved without the presolve, 0.7 -> 21.6 s */
            if (w->rl[i] > FTOL * (1 + fabs(w->rl[i]) + w->rmag[i]) || w->ru[i] < -FTOL * (1 + fabs(w->ru[i]) + w->rmag[i])) { *status = 1; snprintf(why, nwhy, "row %d is empty and excludes 0", i); break; }
            rec_begin(P, R_EMPTYROW); rec_i(P, i);
            w->rowon[i] = 0; (*nempty)++; done++;
            continue;
        }
        int j;
        if (sing_row(w, P, i, &j)) { *status = 1; snprintf(why, nwhy, "column %d: the bounds from row %d are empty", j, i); break; }
        (*nsing)++; done++;
        if (!inc[j] && (ISFIXED(w, j) || w->ccnt[j] == 0)) { qc[nc++] = j; inc[j] = 1; }
    }
    free(qr); free(qc); free(inr); free(inc);
    return done;
}

/* the record of column j as it is now: its cost, then its active entries except in row skip (count, rows, values);
 * a column whose entries and cost are those as read is recorded as -1 (the postsolve takes it from the problem as read:
 * the rows that were not active then are not restored yet at its turn). Before, a long column bounded by many singleton
 * rows was written out for each: 238 million entries and 13.7 s of presolve on map10. */
static void rec_col(SxPre *P, W *w, int j, int skip) {
    if (!w->cmod[j]) { rec_i(P, -1); return; }
    /* (27 October 2026, the night) a column that has not changed since its last full record refers to it (-2 and the
     * record's offsets): the postsolve reads that record, whose rows not present at this record's turn are skipped as
     * they are here (rows only go: those dropped since, and the row this record removes, are not restored yet then).
     * Allowed when the row that record left out is gone. A long column changed by a substitution and bounded by many
     * singleton rows was written out for each: neos-5041822-cockle (LP, MIPLIB 2017 collection) 31,186 records of 3,200
     * entries, the presolve out of memory past 8 GB (the race killed, no answer) */
    if (w->crec_i[j] >= 0 && w->crec_ver[j] == w->cver[j] && (w->crec_skip[j] < 0 || !w->rowon[w->crec_skip[j]])) {
        rec_i(P, -2);
        rec_i(P, (int)(w->crec_i[j] & 0x7fffffff)); rec_i(P, (int)(w->crec_i[j] >> 31));
        rec_i(P, (int)(w->crec_d[j] & 0x7fffffff)); rec_i(P, (int)(w->crec_d[j] >> 31));
        return;
    }
    w->crec_i[j] = (long)P->ni; w->crec_d[j] = (long)P->nd; w->crec_ver[j] = w->cver[j]; w->crec_skip[j] = skip;
    int cnt = 0;
    for (int p = 0; p < w->clen[j]; p++) if (w->rowon[w->ci[j][p]] && w->ci[j][p] != skip) cnt++;
    rec_i(P, cnt); rec_d(P, w->c[j]);
    for (int p = 0; p < w->clen[j]; p++) { const int r = w->ci[j][p]; if (w->rowon[r] && r != skip) { rec_i(P, r); rec_d(P, w->cv[j][p]); } }
}
/* c_k -= d, a cancellation to the rounding level of the two terms taken as an exact zero, as add_entry does for the
 * entries (27 October 2026, the night). neos-4984115-canche (LP, MIPLIB 2017 collection): a free column's cost became
 * -0.0307400974 + 0.0307400974 = 1.7e-18 through a doubleton substitution, its rows went, and the presolve declared the LP
 * unbounded below (Gurobi: optimal, -1.0415e7). Every substitution's cost update goes through here */
static void cost_sub(W *w, int k, double d) {
    const double v = w->c[k] - d;
    w->c[k] = fabs(v) <= ZTOL * fmax(fabs(w->c[k]), fabs(d)) ? 0 : v;
    w->cmod[k] = 1; w->cver[k]++;
}
/* a_rk += d for an active row r and column k (an entry added, changed, or removed when it cancels) */
static double add_entry(W *w, int r, int k, double d, double scale) {   /* (returns |a_rk| after the change, 0 when it is gone) */
    w->cmod[k] = 1; w->cver[k]++;
    /* (26 October 2026) whether the entry is there is read from the shorter of the two lists (only this function
     * changes them, so they hold the same entries): substitutions into one long row searched it once per entry added
     * (neos-1445765, LP: its knapsack row grows to 13,000 entries; the presolve 0.031 -> 0.093 s with the net fill) */
    int pr, pc;
    if (!w->ridx[r] && w->rlen[r] >= RIDX_MIN) ridx_build(w, r);
    if (w->ridx[r]) { pr = wrow_find(w, r, k); pc = pr < 0 ? -1 : wfind(w->ci, w->clen, k, r); }
    else if (w->clen[k] <= w->rlen[r]) { pc = wfind(w->ci, w->clen, k, r); pr = pc < 0 ? -1 : wfind(w->ri, w->rlen, r, k); }
    else { pr = wfind(w->ri, w->rlen, r, k); pc = pr < 0 ? -1 : wfind(w->ci, w->clen, k, r); }
    if (pr < 0) {
        wrow_push(w, r, k, d);
        wpush(w, w->ci, w->cv, w->clen, w->ccap, k, r, d);
        w->rcnt[r]++; w->ccnt[k]++;
        return fabs(d);
    }
    const double v = w->rv[r][pr] + d;
    if (fabs(v) <= ZTOL * scale) {
        wrow_del(w, r, pr); wdel(w->ci, w->cv, w->clen, k, pc);
        w->rcnt[r]--; w->ccnt[k]--;
        return 0;
    }
    w->rv[r][pr] = v; w->cv[k][pc] = v;
    return fabs(v);
}
/* the bounds of row i's activity without column skip: min and max, with the counts of infinite contributions */

static void activity(const W *w, int i, int skip, double *lmin, double *lmax, int *nimin, int *nimax) {
    double a0 = 0, a1 = 0; int n0 = 0, n1 = 0;
    for (int p = 0; p < w->rlen[i]; p++) {
        const int j = w->ri[i][p];
        if (!w->colon[j] || j == skip) continue;
        const double a = w->rv[i][p];
        const double bl = a > 0 ? w->l[j] : w->u[j], bu = a > 0 ? w->u[j] : w->l[j];
        if (fabs(bl) >= PHUGE) n0++; else a0 += a * bl;
        if (fabs(bu) >= PHUGE) n1++; else a1 += a * bu;
    }
    *lmin = a0; *lmax = a1; *nimin = n0; *nimax = n1;
}

/* (27 October 2026) the activity bounds of long rows kept over a pass of the column step: a column singleton with a cost
 * asked for its row's bounds without itself (the row its cost makes tight, the implied free check) scanned the whole row,
 * once per such column: neos-787933 (LP) had 1,764 of them in rows of up to tens of thousands of entries, 0.47 s of its
 * 0.56 s presolve. A row of at least 64 entries keeps its sums over the active columns (computed at the first question
 * of the pass), a column's part is taken out when it leaves the row, and a question subtracts the asking column's part.
 * Not where a column's part exceeds 1e6 in size (the subtraction could cancel digits the tests need: the exact sum) */
typedef struct { double *a0, *a1, *big; int *n0, *n1; char *val; double *a0l, *a1l; } ActCache;
/* (15 November 2026) the kept sums in twice the precision: each term added by an error-free transformation (TwoSum), its
 * rounding error gathered in the low part (Ogita, Rump and Oishi's cascaded summation), so that taking a column's part out
 * loses no digits of the others' sum however large the part. Before, a row whose terms passed 1e6 was rescanned in full at
 * every question: neos-4984115-canche (LP), 1,762 questions a pass on rows of about 130,000 entries, 230 million entries a
 * pass, 0.42 s of its 1.6 s. A term above 1e15 still rescans (the sum's own magnitude then exceeds what double-double
 * keeps exact for the tests). BRISK_SXPRE_ACTDD=0: the rescan above 1e6, as before */
static inline void act_dd_add(double *hi, double *lo, double x) { const double s = *hi + x, bp = s - *hi; *lo += (*hi - (s - bp)) + (x - bp); *hi = s; }
static int act_dd_on(void) { static int on = -1; if (on < 0) { const char *e_ = getenv("BRISK_SXPRE_ACTDD"); on = !(e_ && atoi(e_) == 0); } return on; }
static void act_get(const W *w, ActCache *C, int i, int j, double a, double *lmin, double *lmax, int *nimin, int *nimax) {
    if (w->rlen[i] < 64) { activity(w, i, j, lmin, lmax, nimin, nimax); return; }
    const int dd = act_dd_on();
    if (!C->val[i]) {
        double a0 = 0, a1 = 0, l0 = 0, l1 = 0, big = 0; int n0 = 0, n1 = 0;
        for (int p = 0; p < w->rlen[i]; p++) {
            const int k = w->ri[i][p]; if (!w->colon[k]) continue;
            const double e = w->rv[i][p], bl = e > 0 ? w->l[k] : w->u[k], bu = e > 0 ? w->u[k] : w->l[k];
            if (fabs(bl) >= PHUGE) n0++; else { if (dd) act_dd_add(&a0, &l0, e * bl); else a0 += e * bl; big = fmax(big, fabs(e * bl)); }
            if (fabs(bu) >= PHUGE) n1++; else { if (dd) act_dd_add(&a1, &l1, e * bu); else a1 += e * bu; big = fmax(big, fabs(e * bu)); }
        }
        C->a0[i] = a0; C->a1[i] = a1; C->a0l[i] = l0; C->a1l[i] = l1; C->n0[i] = n0; C->n1[i] = n1; C->big[i] = big; C->val[i] = 1;
    }
    if (C->big[i] > (dd ? 1e15 : 1e6)) { activity(w, i, j, lmin, lmax, nimin, nimax); return; }
    double a0 = C->a0[i], a1 = C->a1[i], l0 = C->a0l[i], l1 = C->a1l[i]; int n0 = C->n0[i], n1 = C->n1[i];
    const double bl = a > 0 ? w->l[j] : w->u[j], bu = a > 0 ? w->u[j] : w->l[j];
    if (dd) {
        if (fabs(bl) >= PHUGE) n0--; else act_dd_add(&a0, &l0, -(a * bl));
        if (fabs(bu) >= PHUGE) n1--; else act_dd_add(&a1, &l1, -(a * bu));
        a0 += l0; a1 += l1;
    } else {
        if (fabs(bl) >= PHUGE) n0--; else a0 -= a * bl;
        if (fabs(bu) >= PHUGE) n1--; else a1 -= a * bu;
    }
    *lmin = a0; *lmax = a1; *nimin = n0; *nimax = n1;
}
/* column j (still active) leaves its rows: its part out of the kept sums */
static void act_leave(const W *w, ActCache *C, int j) {
    for (int p = 0; p < w->clen[j]; p++) {
        const int i = w->ci[j][p]; if (!w->rowon[i] || !C->val[i]) continue;
        const double e = w->cv[j][p], bl = e > 0 ? w->l[j] : w->u[j], bu = e > 0 ? w->u[j] : w->l[j];
        if (act_dd_on()) {
            if (fabs(bl) >= PHUGE) C->n0[i]--; else act_dd_add(&C->a0[i], &C->a0l[i], -(e * bl));
            if (fabs(bu) >= PHUGE) C->n1[i]--; else act_dd_add(&C->a1[i], &C->a1l[i], -(e * bu));
        } else {
            if (fabs(bl) >= PHUGE) C->n0[i]--; else C->a0[i] -= e * bl;
            if (fabs(bu) >= PHUGE) C->n1[i]--; else C->a1[i] -= e * bu;
        }
    }
}

/* the bounds the rows imply for each column (the tightest over its rows, from the other columns' bounds) */
/* sl, su (may be NULL): the row that gives each implied bound (-1: none) */
static void implied_col_bounds(const W *w, double *ilo, double *iup, int *sl, int *su) {
    for (int j = 0; j < w->n; j++) { ilo[j] = -PINF; iup[j] = PINF; }
    if (sl) for (int j = 0; j < w->n; j++) sl[j] = su[j] = -1;
    for (int i = 0; i < w->m; i++) {
        if (!w->rowon[i]) continue;
        const int hasl = w->rl[i] > -PHUGE, hasu = w->ru[i] < PHUGE;
        if (!hasl && !hasu) continue;
        double lmin, lmax; int nmin, nmax;
        activity(w, i, -1, &lmin, &lmax, &nmin, &nmax);
        if (nmin > 1 && nmax > 1) continue;
        for (int p = 0; p < w->rlen[i]; p++) {
            const int j = w->ri[i][p];
            if (!w->colon[j]) continue;
            const double a = w->rv[i][p];
            const double bl = a > 0 ? w->l[j] : w->u[j], bu = a > 0 ? w->u[j] : w->l[j];
            const int il = fabs(bl) >= PHUGE, iu = fabs(bu) >= PHUGE;
            /* the activity of the others: min (for the row's upper bound) and max (for its lower bound) */
            if (hasu && nmin - il == 0) {
                const double v = (w->ru[i] - (lmin - (il ? 0 : a * bl))) / a;
                if (a > 0) { if (v < iup[j]) { iup[j] = v; if (su) su[j] = i; } } else { if (v > ilo[j]) { ilo[j] = v; if (sl) sl[j] = i; } }
            }
            if (hasl && nmax - iu == 0) {
                const double v = (w->rl[i] - (lmax - (iu ? 0 : a * bu))) / a;
                if (a > 0) { if (v > ilo[j]) { ilo[j] = v; if (sl) sl[j] = i; } } else { if (v < iup[j]) { iup[j] = v; if (su) su[j] = i; } }
            }
        }
    }
}
/* (26 October 2026) the implied bounds of the columns marked in mk only, each from all its rows (the others' are left as
 * they are): the aggregator's later sweeps, where only the columns of the rows a substitution changed can have new ones
 * (pds-20, LP: 24 sweeps of the whole matrix, 0.13 s of its 0.37 s presolve). The same values and, on a tie, the same
 * row (the first) as implied_col_bounds; each row's activity is computed once (ra*, stamped st) */
static void implied_col_bounds_mk(const W *w, const char *mk, double *ilo, double *iup, int *sl, int *su,
                                  double *ra, int *rn, int *rst, int st) {
    for (int j = 0; j < w->n; j++) {
        if (!mk[j] || !w->colon[j]) continue;
        ilo[j] = -PINF; iup[j] = PINF; sl[j] = su[j] = -1;
        for (int q = 0; q < w->clen[j]; q++) {
            const int i = w->ci[j][q];
            if (!w->rowon[i]) continue;
            const int hasl = w->rl[i] > -PHUGE, hasu = w->ru[i] < PHUGE;
            if (!hasl && !hasu) continue;
            if (rst[i] != st) { activity(w, i, -1, &ra[2 * i], &ra[2 * i + 1], &rn[2 * i], &rn[2 * i + 1]); rst[i] = st; }
            const double lmin = ra[2 * i], lmax = ra[2 * i + 1]; const int nmin = rn[2 * i], nmax = rn[2 * i + 1];
            if (nmin > 1 && nmax > 1) continue;
            const double a = w->cv[j][q];
            const double bl = a > 0 ? w->l[j] : w->u[j], bu = a > 0 ? w->u[j] : w->l[j];
            const int il = fabs(bl) >= PHUGE, iu = fabs(bu) >= PHUGE;
            if (hasu && nmin - il == 0) {
                const double v = (w->ru[i] - (lmin - (il ? 0 : a * bl))) / a;
                if (a > 0) { if (v < iup[j] || (v == iup[j] && i < su[j])) { iup[j] = v; su[j] = i; } }
                else { if (v > ilo[j] || (v == ilo[j] && i < sl[j])) { ilo[j] = v; sl[j] = i; } }
            }
            if (hasl && nmax - iu == 0) {
                const double v = (w->rl[i] - (lmax - (iu ? 0 : a * bu))) / a;
                if (a > 0) { if (v > ilo[j] || (v == ilo[j] && i < sl[j])) { ilo[j] = v; sl[j] = i; } }
                else { if (v < iup[j] || (v == iup[j] && i < su[j])) { iup[j] = v; su[j] = i; } }
            }
        }
    }
}
/* (24 October 2026) the implied bounds kept while the problem is as it was: four steps of a presolve pass compute them
 * (the dominated columns, the rows the cost makes tight, the substitution, the rows the other rows make redundant), each a
 * sweep of the whole matrix, and nch counts every change; on osa-30 and pds-20 (LPs) 16 and 36 sweeps, a tenth of their
 * dual simplex method's time. The rows that give the bounds are kept with them. BRISK_SXPRE_NOIBCACHE=1: every time */
typedef struct { long nch; double *ilo, *iup; int *sl, *su; } IbCache;
static void implied_col_bounds_c(const W *w, IbCache *C, long nch, double *ilo, double *iup, int *sl, int *su) {
    static int off = -1; if (off < 0) off = getenv("BRISK_SXPRE_NOIBCACHE") != NULL;
    const size_t n = (size_t)w->n;
    if (off) { implied_col_bounds(w, ilo, iup, sl, su); return; }
    if (C->nch != nch) {
        implied_col_bounds(w, C->ilo, C->iup, C->sl, C->su);
        C->nch = nch;
    }
    memcpy(ilo, C->ilo, sizeof(double) * n); memcpy(iup, C->iup, sizeof(double) * n);
    if (sl) { memcpy(sl, C->sl, sizeof(int) * n); memcpy(su, C->su, sizeof(int) * n); }
}

/* implied bounds on the row duals of an optimal solution (minimization): y_i >= 0 for a row with only a lower bound,
 * <= 0 for one with only an upper bound (the original bounds); then each column's optimality condition, z_j = c_j - a_j'y
 * >= 0 if x_j has no upper bound and <= 0 if it has no lower bound (a bound the rows imply strictly counts as absent),
 * gives a bound on each y_i from the original bounds of the others (as HiGHS does: no chaining of derived bounds, which
 * could be circular) */
/* (24 October 2026, the night) an inert bound: x_j's upper bound when c_j > 0 and, at that bound, x_j alone keeps every
 * row of its column within the bound that a decrease of x_j moves towards (whatever the other columns do within their
 * bounds); its lower bound likewise when c_j < 0. Dropping all such bounds at once keeps the optimal value and every
 * optimum: projecting a solution of the relaxed problem onto them (x_j -> min(x_j, u_j), max(x_j, l_j)) keeps every row
 * feasible (a row whose activity a projection lowered is above its lower bound by that column's condition, all columns
 * then being within their bounds) and lowers the cost. So for the dual reasoning below the bound counts as absent. On a
 * set covering (x in [0, 1], A x >= 1, c > 0) every upper bound is inert: lp_setcover_8k (LP) 60,001 -> 12,313 columns,
 * the interior-point method's presolve and Gurobi's alike. BRISK_SXINERT=0: off */
static int inert_on(void) { static int c = -1; if (c < 0) { const char *e = getenv("BRISK_SXINERT"); c = !(e && atoi(e) == 0); } return c; }
static void inert_bounds(const W *w, char *inu, char *inl) {
    const int m = w->m, n = w->n;
    memset(inu, 0, n); memset(inl, 0, n);
    if (!inert_on()) return;
    double *amin = xmalloc(sizeof(double) * (m + 1)), *amax = xmalloc(sizeof(double) * (m + 1));
    int *nmin = xmalloc(sizeof(int) * (m + 1)), *nmax = xmalloc(sizeof(int) * (m + 1));
    for (int i = 0; i < m; i++) {
        amin[i] = amax[i] = 0; nmin[i] = nmax[i] = 0;
        if (!w->rowon[i]) continue;
        for (int p = 0; p < w->rlen[i]; p++) {
            const int j = w->ri[i][p];
            if (!w->colon[j]) continue;
            const double a = w->rv[i][p], lo = a > 0 ? w->l[j] : w->u[j], hi = a > 0 ? w->u[j] : w->l[j];
            if (fabs(lo) >= PHUGE) nmin[i]++; else amin[i] += a * lo;
            if (fabs(hi) >= PHUGE) nmax[i]++; else amax[i] += a * hi;
        }
    }
    for (int j = 0; j < n; j++) {
        if (!w->colon[j] || w->c[j] == 0) continue;
        const int up = w->c[j] > 0;                 /* the bound the cost pushes x_j to leave from above (up) or below */
        const double b = up ? w->u[j] : w->l[j];
        if (fabs(b) >= PHUGE) continue;
        int ok = 1;
        for (int p = 0; p < w->clen[j] && ok; p++) {
            const int i = w->ci[j][p];
            if (!w->rowon[i]) continue;
            const double a = w->cv[j][p];
            /* moving x_j towards its other side lowers the row's activity when a > 0 and up (or a < 0 and down): then the
             * row's lower bound must hold with x_j at b and the others at their least; otherwise its upper bound with the
             * others at their most */
            if ((a > 0) == up) {
                if (w->rl[i] <= -PHUGE) continue;
                const double lo = a > 0 ? w->l[j] : w->u[j];
                const int own = fabs(lo) >= PHUGE;
                if (nmin[i] - own > 0 || a * b + (amin[i] - (own ? 0 : a * lo)) < w->rl[i]) ok = 0;
            } else {
                if (w->ru[i] >= PHUGE) continue;
                const double hi = a > 0 ? w->u[j] : w->l[j];
                const int own = fabs(hi) >= PHUGE;
                if (nmax[i] - own > 0 || a * b + (amax[i] - (own ? 0 : a * hi)) > w->ru[i]) ok = 0;
            }
        }
        if (ok) { if (up) inu[j] = 1; else inl[j] = 1; }
    }
    free(amin); free(amax); free(nmin); free(nmax);
}
static void dual_bounds(const W *w, double *ylo, double *yup, const double *ilo, const double *iup, double *y0l, double *y0u, int *slo, int *sup, const char *inu, const char *inl) {
    for (int i = 0; i < w->m; i++) {
        if (!w->rowon[i]) { y0l[i] = y0u[i] = 0; continue; }
        const int linf = w->rl[i] <= -PHUGE, uinf = w->ru[i] >= PHUGE;
        y0l[i] = uinf ? 0 : -PINF; y0u[i] = linf ? 0 : PINF;
        ylo[i] = y0l[i]; yup[i] = y0u[i]; slo[i] = sup[i] = -1;
    }
    for (int j = 0; j < w->n; j++) {
        if (!w->colon[j]) continue;
        const double mu = (w->ccnt[j] != 1 ? 1 : -1) * FTOL * (1 + fabs(w->u[j])), ml = (w->ccnt[j] != 1 ? 1 : -1) * FTOL * (1 + fabs(w->l[j]));
        const int noup = w->u[j] >= PHUGE || iup[j] < w->u[j] - mu || inu[j], nolo = w->l[j] <= -PHUGE || ilo[j] > w->l[j] + ml || inl[j];
        if (!noup && !nolo) continue;
        double smin = 0, smax = 0; int nmin = 0, nmax = 0;     /* a_j'y over the original box */
        for (int p = 0; p < w->clen[j]; p++) {
            const int i = w->ci[j][p];
            if (!w->rowon[i]) continue;
            const double a = w->cv[j][p];
            const double lo = a > 0 ? y0l[i] : y0u[i], hi = a > 0 ? y0u[i] : y0l[i];
            if (fabs(lo) >= PHUGE) nmin++; else smin += a * lo;
            if (fabs(hi) >= PHUGE) nmax++; else smax += a * hi;
        }
        const double cj = w->c[j];
        for (int p = 0; p < w->clen[j]; p++) {
            const int i = w->ci[j][p];
            if (!w->rowon[i]) continue;
            const double a = w->cv[j][p];
            if (noup) {      /* a y_i <= c_j - min of the others */
                const double lo = a > 0 ? y0l[i] : y0u[i];
                const int own = fabs(lo) >= PHUGE;
                if (nmin - own == 0) {
                    const double b = (cj - (smin - (own ? 0 : a * lo))) / a;
                    if (a > 0) { if (b < yup[i]) { yup[i] = b; sup[i] = j; } } else { if (b > ylo[i]) { ylo[i] = b; slo[i] = j; } }
                }
            }
            if (nolo) {      /* a y_i >= c_j - max of the others */
                const double hi = a > 0 ? y0u[i] : y0l[i];
                const int own = fabs(hi) >= PHUGE;
                if (nmax - own == 0) {
                    const double b = (cj - (smax - (own ? 0 : a * hi))) / a;
                    if (a > 0) { if (b > ylo[i]) { ylo[i] = b; slo[i] = j; } } else { if (b < yup[i]) { yup[i] = b; sup[i] = j; } }
                }
            }
        }
    }
}

/* parallel rows: rows whose active entries are proportional (a_k = lambda a_i) become one row, with the intersection of
 * their bounds (row k's divided by lambda); the postsolve gives the dual and the nonbasic status to the row whose bound
 * is the active one. Rows are compared through a hash of their entries scaled by the entry of the smallest column, then
 * exactly. Returns the rows dropped; *infeas when two bounds contradict. */
/* (27 October 2026) pairs of words sorted inline (quicksort, median of three, insertion sort below 16), for the sorts
 * that went through qsort and a comparator: the symmetry search's refinement (class, key, line packed into two words:
 * over half the search's instructions on small problems, sc205, LP) and the parallel rows and columns (hash, index); the
 * same orders */
typedef struct { unsigned long long a, b; } RefKey;
static inline int rk_lt(const RefKey *x, const RefKey *y) { return x->a < y->a || (x->a == y->a && x->b < y->b); }
static void rk_sort(RefKey *v, int n) {
    while (n > 16) {
        RefKey *m = v + n / 2, *l = v + n - 1, t;
        if (rk_lt(m, v)) { t = *m; *m = *v; *v = t; }
        if (rk_lt(l, m)) { t = *l; *l = *m; *m = t; if (rk_lt(m, v)) { t = *m; *m = *v; *v = t; } }
        const RefKey p = *m;
        int i = 0, j = n - 1;
        for (;;) {
            while (rk_lt(&v[i], &p)) i++;
            while (rk_lt(&p, &v[j])) j--;
            if (i >= j) break;
            t = v[i]; v[i] = v[j]; v[j] = t; i++; j--;
        }
        /* the smaller part by recursion, the larger by the loop (depth at most log2 n) */
        if (j + 1 < n - j - 1) { rk_sort(v, j + 1); v += j + 1; n -= j + 1; } else { rk_sort(v + j + 1, n - j - 1); n = j + 1; }
    }
    for (int i = 1; i < n; i++) { const RefKey x = v[i]; int k = i - 1; while (k >= 0 && rk_lt(&x, &v[k])) { v[k + 1] = v[k]; k--; } v[k + 1] = x; }
}
static unsigned long long pmix(unsigned long long z) {
    z += 0x9e3779b97f4a7c15ULL; z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL; z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL; return z ^ (z >> 31);
}
static unsigned long long pbits(double v) { unsigned long long b; if (v == 0) v = 0; memcpy(&b, &v, sizeof b); return b; }
/* (4 November 2026) the parallel rows' and columns' hash of an entry over the line's reference entry: the bits rounded to 32
 * significant ones, and the candidates then compared to a relative 1e-12 (PAR_EQ), not exactly. Two rows that are multiples of
 * each other in the problem's own units stay multiples after each row is multiplied by its own factor (a copy that must
 * cost the same), but their quotients a_ij / a_i,ref are then rounded apart in the last bit, hashed apart and never
 * compared: neos-5018451-chiese (LP) with its rows multiplied by 10^u, u in [-2, 2], kept 22,127 of 170,175 rows against
 * 5,013 as written (14,458 parallel rows found against 24,201, 106,833 parallel columns against 122,628), 1.8 -> 6.7 s */
static unsigned long long pbitsq(double v) { unsigned long long b = pbits(v); b += 1ULL << 19; return b & ~((1ULL << 20) - 1); }
#define PAR_EQ(x, y) (fabs((x) - (y)) <= 1e-12 * fabs(x))
static int parallel_rows(W *w, SxPre *P, int *infeas) {
    const int m = w->m;
    unsigned long long *h = xmalloc(sizeof(unsigned long long) * (m + 1)); double *sc = xmalloc(sizeof(double) * (m + 1));
    int *ord = xmalloc(sizeof(int) * (m + 1)), no = 0;
    for (int i = 0; i < m; i++) {
        if (!w->rowon[i] || w->rcnt[i] < 1) continue;
        int jmin = -1; double amin = 0;
        for (int p = 0; p < w->rlen[i]; p++) { const int j = w->ri[i][p]; if (w->colon[j] && w->rv[i][p] != 0 && (jmin < 0 || j < jmin)) { jmin = j; amin = w->rv[i][p]; } }
        if (jmin < 0) continue;
        sc[i] = 1.0 / amin;
        unsigned long long hh = 0; int c = 0;
        for (int p = 0; p < w->rlen[i]; p++) { const int j = w->ri[i][p]; if (w->colon[j] && w->rv[i][p] != 0) { hh += pmix(((unsigned long long)j << 32) ^ pmix(pbitsq(w->rv[i][p] * sc[i]))); c++; } }
        h[i] = hh + pmix((unsigned long long)c); ord[no++] = i;
    }
    {   RefKey *v = xmalloc(sizeof(RefKey) * ((size_t)no + 1));
        for (int t = 0; t < no; t++) { v[t].a = h[ord[t]]; v[t].b = (unsigned long long)ord[t]; }
        rk_sort(v, no);
        for (int t = 0; t < no; t++) ord[t] = (int)v[t].b;
        free(v); }
    double *val = xcalloc(w->n + 1, sizeof(double)); char *mk = xcalloc(w->n + 1, 1);
    int ndrop = 0;
    for (int t = 0; t < no; ) {
        int e = t; while (e < no && h[ord[e]] == h[ord[t]]) e++;
        if (e - t > 1) {
            const int i = ord[t];
            int ci = 0;
            for (int p = 0; p < w->rlen[i]; p++) { const int j = w->ri[i][p]; if (w->colon[j] && w->rv[i][p] != 0) { val[j] = w->rv[i][p] * sc[i]; mk[j] = 1; ci++; } }
            for (int u = t + 1; u < e; u++) {
                const int k = ord[u];
                int ck = 0, same = 1;
                for (int p = 0; p < w->rlen[k] && same; p++) { const int j = w->ri[k][p]; if (!w->colon[j] || w->rv[k][p] == 0) continue; ck++; if (!mk[j] || !PAR_EQ(val[j], w->rv[k][p] * sc[k])) same = 0; }
                if (!same || ck != ci) continue;
                const double lam = sc[i] / sc[k];          /* a_k = lam a_i */
                const double lk = lam > 0 ? (w->rl[k] > -PHUGE ? w->rl[k] / lam : -PINF) : (w->ru[k] < PHUGE ? w->ru[k] / lam : -PINF);
                const double uk = lam > 0 ? (w->ru[k] < PHUGE ? w->ru[k] / lam : PINF) : (w->rl[k] > -PHUGE ? w->rl[k] / lam : PINF);
                rec_begin(P, R_PARROW); rec_i(P, i); rec_i(P, k); rec_d(P, lam); rec_d(P, w->rl[i]); rec_d(P, w->ru[i]); rec_d(P, lk); rec_d(P, uk);
                double nl = w->rl[i] > lk ? w->rl[i] : lk, nu = w->ru[i] < uk ? w->ru[i] : uk;
                /* (4 November 2026) the sides' rounding (rmag) counted, row k's carried by 1 / lam (the empty-row tests) */
                const double rmk = w->rmag[k] / fabs(lam);
                if (nl > nu) { if (nl - nu > FTOL * (1 + fabs(nl) + w->rmag[i] + rmk)) { *infeas = 1; break; } nu = nl; }
                w->rl[i] = nl; w->ru[i] = nu; if (rmk > w->rmag[i]) w->rmag[i] = rmk;
                drop_row(w, k); ndrop++;
            }
            for (int p = 0; p < w->rlen[i]; p++) { const int j = w->ri[i][p]; mk[j] = 0; val[j] = 0; }
            if (*infeas) break;
        }
        t = e;
    }
    free(h); free(sc); free(ord); free(val); free(mk);
    return ndrop;
}
/* (31 October 2026, ISSUES 638) an equality row replaced by its difference with another, s <- s - t r with t = a_sj / a_rj,
 * when entries cancel exactly (a_sk - t a_rk == 0 in floating point), more of them than the fill r adds to s, and s keeps
 * one or two entries (restricted, see below): Gondzio's
 * "making A sparser" (1997), Gurobi's sparsify. No tolerance decides anything: only exact cancellations are counted.
 * near_dependent (LPs of an independent benchmark): rows A x = b and copies (A + eps I) x = b'; each copy becomes the
 * singleton row eps x_i = b'_i - b_i, which fixes x_i, and the LP's optimum is found (-262.03 on a fresh instance, where
 * Gurobi, HiGHS and BRISK without it return -380.34, the optimum of the problem relaxed by the feasibility tolerance).
 * The candidates of row r are the equality rows of r's shortest column with at least as many entries; the work is capped
 * at 20 times the matrix's entries. Postsolve: x as it is; y_r <- y_r - t y_s (the reduced problem's rows r and s' = s -
 * t r give A'y = a_r (y_r - t y_s') + a_s y_s'); the statuses as they are. BRISK_SXPRE_SPARSIFY=0: off */
static int sparsify_rows(W *w, SxPre *P) {
    const int m = w->m, n = w->n;
    long nz = 0; for (int i = 0; i < m; i++) if (w->rowon[i]) nz += w->rcnt[i];
    const long budget = 20 * nz + 1000; long work = 0;
    double *val = xcalloc((size_t)n + 1, sizeof(double)); char *mk = xcalloc((size_t)n + 1, 1);
    int *cand = xmalloc(sizeof(int) * ((size_t)m + 1));
    int nred = 0;
    for (int r = 0; r < m && work < budget; r++) {
        if (SXPRE_STOPPED()) break;
        if (!w->rowon[r] || w->rcnt[r] < 2 || w->rl[r] != w->ru[r] || !(w->rl[r] > -PHUGE)) continue;
        int j0 = -1, na = 0;
        for (int p = 0; p < w->rlen[r]; p++) {
            const int k = w->ri[r][p]; const double v = w->rv[r][p];
            if (!w->colon[k] || v == 0) continue;
            mk[k] = 1; val[k] = v; na++;
            if (j0 < 0 || w->ccnt[k] < w->ccnt[j0]) j0 = k;
        }
        work += w->rlen[r];
        if (j0 >= 0 && na >= 2) {
            int nc = 0;
            for (int q = 0; q < w->clen[j0]; q++) { const int sr = w->ci[j0][q]; if (sr != r && w->rowon[sr] && w->cv[j0][q] != 0) cand[nc++] = sr; }
            work += w->clen[j0];
            for (int c = 0; c < nc && work < budget; c++) {
                const int sr = cand[c];
                if (!w->rowon[sr] || w->rl[sr] != w->ru[sr] || !(w->rl[sr] > -PHUGE) || w->rcnt[sr] < na) continue;
                const int pj = wrow_find(w, sr, j0);
                if (pj < 0 || w->rv[sr][pj] == 0) continue;
                const double t = w->rv[sr][pj] / val[j0];
                int cancel = 0, ins = 0, act = 0;
                for (int p = 0; p < w->rlen[sr]; p++) {
                    const int k = w->ri[sr][p]; if (!w->colon[k] || w->rv[sr][p] == 0) continue;
                    act++;
                    if (!mk[k]) continue;
                    ins++;
                    if (k == j0 || w->rv[sr][p] - t * val[k] == 0.0) cancel++;
                }
                work += w->rlen[sr];
                const int fill = na - ins;
                /* (only when s - t r keeps one or two entries: a singleton equation, which fixes its column, or a doubleton,
                 * which the presolve substitutes; either removes the row. Combinations made only to save entries changed the
                 * dual simplex method's path on held-out LPs: momentum2 8,145 -> over 270,000 iterations, hgms62 63 -> 122 s,
                 * ns1904248 38 s -> past 150 s) */
                if (cancel - fill <= 0 || cancel >= act || act - cancel + fill > 2) continue;
                for (int p = 0; p < w->rlen[r]; p++) {
                    const int k = w->ri[r][p]; if (!w->colon[k] || w->rv[r][p] == 0) continue;
                    if (k == j0) {          /* (the pivot's entry is zero by construction, whatever the rounding of t) */
                        const int a = wrow_find(w, sr, j0), b = wfind(w->ci, w->clen, j0, sr);
                        if (a >= 0 && b >= 0) { wrow_del(w, sr, a); wdel(w->ci, w->cv, w->clen, j0, b); w->rcnt[sr]--; w->ccnt[j0]--; w->cmod[j0] = 1; w->cver[j0]++; }
                        continue;
                    }
                    const int a = wrow_find(w, sr, k);
                    const double ask = a >= 0 ? w->rv[sr][a] : 0.0, d = -t * val[k];
                    if (a >= 0 && ask + d == 0.0) {     /* (an exact cancellation) */
                        const int b = wfind(w->ci, w->clen, k, sr);
                        wrow_del(w, sr, a); if (b >= 0) wdel(w->ci, w->cv, w->clen, k, b);
                        w->rcnt[sr]--; w->ccnt[k]--; w->cmod[k] = 1; w->cver[k]++;
                    } else add_entry(w, sr, k, d, fmax(fabs(ask), fabs(d)));
                }
                const double nb = w->rl[sr] - t * w->rl[r];
                w->rl[sr] = w->ru[sr] = nb;
                rec_begin(P, R_ROWDIFF); rec_i(P, r); rec_i(P, sr); rec_d(P, t);
                nred++;
            }
        }
        for (int p = 0; p < w->rlen[r]; p++) { mk[w->ri[r][p]] = 0; val[w->ri[r][p]] = 0; }
    }
    free(val); free(mk); free(cand);
    return nred;
}
/* (28 October 2026, the night; ISSUES 626) linearly dependent equality rows dropped (lp_dependent_rows, lpio.c), once at the end
 * of the passes; the record of a redundant row (its dual 0, its logical basic). BRISK_SXPRE_DEPROWS=0: off */
static int dep_key3_cmp(const void *a_, const void *b_) { const long *a = a_, *b = b_; for (int t = 0; t < 3; t++) if (a[t] != b[t]) return a[t] < b[t] ? -1 : 1; return 0; }
static int dependent_rows(W *w, SxPre *P) {
    const int m = w->m, n = w->n;
    int me = 0; long nze = 0;
    for (int i = 0; i < m; i++) if (w->rowon[i] && w->rcnt[i] > 0 && w->rl[i] == w->ru[i] && w->rl[i] > -PHUGE) { me++; nze += w->rlen[i]; }
    if (me < 2) return 0;
    int *rows = xmalloc(sizeof(int) * (size_t)me); long *ep = xmalloc(sizeof(long) * ((size_t)me + 1));
    int *ej = xmalloc(sizeof(int) * ((size_t)nze + 1)); double *ev = xmalloc(sizeof(double) * ((size_t)nze + 1)), *eb = xmalloc(sizeof(double) * (size_t)me);
    char *dep = xmalloc((size_t)me);
    int t = 0; long q = 0;
    /* (experiment BRISK_SXPRE_DEPCOST=1) the rows holding a column with a cost first: within a length the search takes the
     * rows in the order given, so those become pivots and the rows found dependent are rows without costs */
    static int depcost = -1; if (depcost < 0) { const char *e_ = getenv("BRISK_SXPRE_DEPCOST"); depcost = e_ ? atoi(e_) : 1; }
    int *ord = xmalloc(sizeof(int) * (size_t)me);
    { int a = 0; for (int i = 0; i < m; i++) if (w->rowon[i] && w->rcnt[i] > 0 && w->rl[i] == w->ru[i] && w->rl[i] > -PHUGE) ord[a++] = i;
      if (depcost) { int b = 0; int *o2 = xmalloc(sizeof(int) * (size_t)me);
          int *nc_ = xcalloc((size_t)m + 1, sizeof(int)); int mx_ = 0;
          for (int k = 0; k < me; k++) { const int i = ord[k]; for (int p = 0; p < w->rlen[i]; p++) { const int j = w->ri[i][p]; nc_[i] += w->colon[j] && w->rv[i][p] != 0 && w->c[j] != 0; } if (nc_[i] > mx_) mx_ = nc_[i]; }
          if (depcost == 1) { for (int pass_ = 0; pass_ < 2; pass_++) for (int k = 0; k < me; k++) { const int i = ord[k]; if ((nc_[i] > 0) == (pass_ == 0)) o2[b++] = i; } }
          else for (int v = mx_; v >= 0; v--) for (int k = 0; k < me; k++) { const int i = ord[k]; if (nc_[i] == v) o2[b++] = i; }
          free(nc_);
          memcpy(ord, o2, sizeof(int) * (size_t)me); free(o2); } }
    {   /* (15 November 2026; BRISK_SXPRE_DEPFREE=0: off) within the order above, the rows ordered by the count of their columns
         * with two active entries, fewest first (a stable sort): the search takes the rows in this order within a length, the rows it
         * finds dependent are those that come last, and a row dropped leaves each such column alone in its other row, where the passes
         * after (BRISK_SXPRE_DEPREPASS) remove it as a slack column or a singleton. Satellites4-25 (LP): the slack columns removed
         * depended on the rows' order (20,690 as written, 30,188 with the rows in reverse Cuthill-McKee order, 17,864 on a shuffled copy);
         * the columns kept in one form only were columns of three entries without cost */
        /* (on by default since 15 November 2026: the 618 development LPs 548 of 560 identical, the satellites 17-42 % fewer iterations, the
         * held-out set 1.007 with the same iterations on 68 of 72 and no loss; BRISK_SXPRE_DEPFREE=0: the order as before) */
        static int depfree = -1; if (depfree < 0) { const char *e_ = getenv("BRISK_SXPRE_DEPFREE"); depfree = !(e_ && atoi(e_) == 0); }
        if (depfree) {
            /* (each row's key once: its part (a row with a cost first, as the order above has it) and its count; then one stable sort by
             * (part, count, place in the order above). A first version recomputed the part in a loop over every count value: rail03, LP,
             * whose rows hold thousands of such columns, 1.9 -> 21 s of presolve) */
            long *key3 = xmalloc(sizeof(long) * (size_t)me * 3);
            for (int k = 0; k < me; k++) { const int i = ord[k]; long cnt = 0; int hasc = !depcost;
                for (int p = 0; p < w->rlen[i]; p++) { const int j = w->ri[i][p]; if (!w->colon[j] || w->rv[i][p] == 0) continue; if (w->ccnt[j] == 2) cnt++; if (w->c[j] != 0) hasc = 1; }
                key3[3 * k] = hasc ? 0 : 1; key3[3 * k + 1] = cnt; key3[3 * k + 2] = k; }
            qsort(key3, (size_t)me, 3 * sizeof(long), dep_key3_cmp);
            int *o3 = xmalloc(sizeof(int) * (size_t)me);
            for (int k = 0; k < me; k++) o3[k] = ord[key3[3 * k + 2]];
            memcpy(ord, o3, sizeof(int) * (size_t)me);
            free(key3); free(o3);
        }
    }
    for (int k_ = 0; k_ < me; k_++) { const int i = ord[k_];
        rows[t] = i; ep[t] = q; eb[t] = w->rl[i];
        for (int p = 0; p < w->rlen[i]; p++) if (w->colon[w->ri[i][p]] && w->rv[i][p] != 0) { ej[q] = w->ri[i][p]; ev[q++] = w->rv[i][p]; }
        t++;
    }
    free(ord);
    ep[me] = q;
    int nd = lp_dependent_rows(me, n, ep, ej, ev, eb, dep, g_sxpre_stop);
    /* (15 November 2026, the night; BRISK_SXPRE_DEPPAIR=k: an experiment, k rounds) the choice refined by the columns it frees: a
     * column with two active entries is freed when one of its rows is dropped, a column of three entries without cost when two are
     * (it then becomes a slack column or a singleton in the passes after). Given the rows found dependent, each equation's gain is
     * the count of its columns that dropping it would free with the others dropped (two entries; three entries without cost whose
     * other two rows hold a dropped one); the rows ordered by (part, gain, place), the search run again, and its result kept when it
     * frees more columns. Satellites4-25 (LP): of 1,260 rows dropped, 40 differed between the file and a shuffled copy, all ties of
     * the order above, and the three-entry columns freed were 25,036 against 24,815, the difference of the slack columns removed */
    static int deppair = -1; if (deppair < 0) { const char *e_ = getenv("BRISK_SXPRE_DEPPAIR"); deppair = e_ ? atoi(e_) : 0; }
    if (deppair > 0 && nd > 0) {
        char *dr = xcalloc((size_t)m + 1, 1);
        long *key3 = xmalloc(sizeof(long) * (size_t)me * 3);
        int *rows2 = xmalloc(sizeof(int) * (size_t)me); long *ep2 = xmalloc(sizeof(long) * ((size_t)me + 1));
        int *ej2 = xmalloc(sizeof(int) * ((size_t)nze + 1)); double *ev2 = xmalloc(sizeof(double) * ((size_t)nze + 1)), *eb2 = xmalloc(sizeof(double) * (size_t)me);
        char *dep2 = xmalloc((size_t)me);
        for (int r_ = 0; r_ < deppair && !brisk_flag_get(g_sxpre_stop); r_++) {
            long T[2];
            for (int s_ = 0; s_ < 2; s_++) {   /* (s_ 0: the current choice; 1: after the search in the new order) */
                const int *rw = s_ ? rows2 : rows; const char *dp = s_ ? dep2 : dep;
                for (int k = 0; k < me; k++) dr[rw[k]] = dp[k];
                long tot = 0;
                for (int j = 0; j < n; j++) {
                    if (!w->colon[j] || (w->ccnt[j] != 2 && !(w->ccnt[j] == 3 && w->c[j] == 0))) continue;
                    int kd = 0; for (int p = 0; p < w->clen[j]; p++) if (w->rowon[w->ci[j][p]] && w->cv[j][p] != 0) kd += dr[w->ci[j][p]];
                    tot += kd >= w->ccnt[j] - 1;
                }
                T[s_] = tot;
                if (s_ == 1) break;
                /* the gains under the current choice, the new order and the search */
                for (int k = 0; k < me; k++) { const int i = rows[k]; long g = 0; int hasc = !depcost;
                    for (int p = 0; p < w->rlen[i]; p++) { const int j = w->ri[i][p]; if (!w->colon[j] || w->rv[i][p] == 0) continue;
                        if (w->c[j] != 0) hasc = 1;
                        if (w->ccnt[j] == 2) g++;
                        else if (w->ccnt[j] == 3 && w->c[j] == 0) { int kd = 0; for (int pp = 0; pp < w->clen[j]; pp++) { const int r2 = w->ci[j][pp]; if (r2 != i && w->rowon[r2] && w->cv[j][pp] != 0) kd += dr[r2]; } g += kd >= 1; } }
                    key3[3 * k] = hasc ? 0 : 1; key3[3 * k + 1] = g; key3[3 * k + 2] = k; }
                qsort(key3, (size_t)me, 3 * sizeof(long), dep_key3_cmp);
                long q2 = 0;
                for (int k = 0; k < me; k++) { const int i = rows[key3[3 * k + 2]]; rows2[k] = i; ep2[k] = q2; eb2[k] = w->rl[i];
                    for (int p = 0; p < w->rlen[i]; p++) if (w->colon[w->ri[i][p]] && w->rv[i][p] != 0) { ej2[q2] = w->ri[i][p]; ev2[q2++] = w->rv[i][p]; } }
                ep2[me] = q2;
                const int nd2 = lp_dependent_rows(me, n, ep2, ej2, ev2, eb2, dep2, g_sxpre_stop);
                if (nd2 != nd) { T[1] = -1; break; }   /* (the rank is the rank: a different count is a numerical difference, not a choice) */
            }
            for (int k = 0; k < me; k++) dr[rows[k]] = 0, dr[rows2[k]] = 0;
            if (getenv("BRISK_LPDEPDBG")) fprintf(stderr, "  [dependent rows, round %d: columns freed %ld -> %ld]\n", r_ + 1, T[0], T[1]);
            if (T[1] <= T[0]) break;
            { int *t1 = rows; rows = rows2; rows2 = t1; } { long *t2 = ep; ep = ep2; ep2 = t2; } { int *t3 = ej; ej = ej2; ej2 = t3; }
            { double *t4 = ev; ev = ev2; ev2 = t4; } { double *t5 = eb; eb = eb2; eb2 = t5; } { char *t6 = dep; dep = dep2; dep2 = t6; }
        }
        free(dr); free(key3); free(rows2); free(ep2); free(ej2); free(ev2); free(eb2); free(dep2);
    }
    if (getenv("BRISK_LPDEPDBG")) { long h[8] = {0}, g[8] = {0}; for (int k = 0; k < me; k++) { const long l = ep[k + 1] - ep[k]; const int b = l <= 2 ? 0 : l <= 4 ? 1 : l <= 8 ? 2 : l <= 16 ? 3 : l <= 32 ? 4 : l <= 64 ? 5 : 6; g[b]++; if (dep[k]) h[b]++; }
        fprintf(stderr, "  [dependent rows by length (<=2,4,8,16,32,64,more): dropped %ld %ld %ld %ld %ld %ld %ld of %ld %ld %ld %ld %ld %ld %ld]\n", h[0], h[1], h[2], h[3], h[4], h[5], h[6], g[0], g[1], g[2], g[3], g[4], g[5], g[6]); }
    if (getenv("BRISK_LPDEPLIST")) { FILE *f_ = fopen(getenv("BRISK_LPDEPLIST"), "a"); if (f_) { for (int k = 0; k < me; k++) if (dep[k]) fprintf(f_, "%d\n", rows[k]); fclose(f_); } }
    for (int k = 0; k < me; k++) if (dep[k]) { rec_begin(P, R_REDROW); rec_i(P, rows[k]); drop_row(w, rows[k]); }
    free(rows); free(ep); free(ej); free(ev); free(eb); free(dep);
    return nd;
}
/* (15 November 2026, the night; the SDP developer's finding on two SDPs, MESSAGES.md 15 November) the free columns that are linear
 * combinations of other free columns with the same combination of costs: A_f d = 0 and c_f'd = 0 for some d over free columns. Along d
 * every solution stays a solution of the same value, so a column k with d_k != 0 is fixed at 0 (a solution keeps a representation with
 * x_k = 0: shift it along d), and its dual equation a_k'y = c_k is implied by the others' (its reduced cost 0 at every dual feasible
 * point). The search is the dependent equations' (lp_dependent_rows) on the free columns as rows over the active rows, the costs as the
 * right-hand sides: it marks a column only when the cost follows the combination (otherwise the LP is unbounded or infeasible, and
 * nothing is fixed). In the interior-point method such a direction is held only by the free columns' weight; the normal equations
 * lose rank and their pivots are replaced: stockholm (LP) has 45 among its 18,720 free columns. BRISK_SXPRE_DEPFREECOL=0: kept */
static int dependent_free_cols(W *w, SxPre *P) {
    const int m = w->m, n = w->n;
    int nf = 0; long nzf = 0;
    for (int j = 0; j < n; j++) if (w->colon[j] && w->l[j] <= -PHUGE && w->u[j] >= PHUGE && w->ccnt[j] > 0) {
        nf++; for (int p = 0; p < w->clen[j]; p++) nzf += w->rowon[w->ci[j][p]] && w->cv[j][p] != 0; }
    if (nf < 2) return 0;
    int *fcol = xmalloc(sizeof(int) * (size_t)nf); long *ep = xmalloc(sizeof(long) * ((size_t)nf + 1));
    int *ej = xmalloc(sizeof(int) * ((size_t)nzf + 1)); double *ev = xmalloc(sizeof(double) * ((size_t)nzf + 1)), *eb = xmalloc(sizeof(double) * (size_t)nf);
    char *dep = xmalloc((size_t)nf); int t = 0; long q = 0;
    for (int j = 0; j < n; j++) {
        if (!(w->colon[j] && w->l[j] <= -PHUGE && w->u[j] >= PHUGE && w->ccnt[j] > 0)) continue;
        fcol[t] = j; ep[t] = q; eb[t] = w->c[j];
        for (int p = 0; p < w->clen[j]; p++) if (w->rowon[w->ci[j][p]] && w->cv[j][p] != 0) { ej[q] = w->ci[j][p]; ev[q++] = w->cv[j][p]; }
        t++;
    }
    ep[nf] = q;
    const int nd = lp_dependent_rows(nf, m, ep, ej, ev, eb, dep, g_sxpre_stop);
    for (int k = 0; k < nf; k++) if (dep[k]) { const int j = fcol[k]; rec_begin(P, R_FIXCOL); rec_i(P, j); rec_i(P, SX_ZERO); rec_d(P, 0.0); drop_col(w, j, 0.0); }
    free(fcol); free(ep); free(ej); free(ev); free(eb); free(dep);
    return nd;
}
/* parallel columns: columns whose active entries and costs are proportional (a_k = lambda a_j, c_k = lambda c_j) act
 * as one variable x_j + lambda x_k: column k is dropped and column j takes the bounds of that sum. The postsolve splits
 * the value: at a bound of the sum both are at the bounds that give it; inside, one of them is basic.
 * (27 October 2026) Proportional entries with costs that are not: the reduced costs satisfy z_k = delta + lambda z_j with
 * delta = c_k - lambda c_j, since a_k^T y = lambda a_j^T y. When x_j has no upper bound, every dual feasible point has
 * z_j >= 0 (z_j <= 0 when it has no lower bound); if that makes lambda z_j >= 0 and delta > 0, then z_k > 0 at every dual
 * feasible point and x_k sits at its lower bound in every optimal solution (with the signs turned, at its upper bound).
 * The column is fixed there (Andersen and Andersen's dominated duplicate columns; Gurobi's presolve removes 8,841 columns
 * of osa-60, LP, where the merging above removes 764). The test runs both ways (x_j fixed through x_k's missing bound);
 * a fixed x_j hands the bucket to x_k, whose scaled entries are the same. BRISK_SXPRE_DOMPAR=0: not done. */
/* (27 October 2026, BRISK_SXPRE_PAIRDOM=1: an experiment) columns dominated by another. Column j dominates column k when
 * c_j <= c_k and moving a unit from x_k to x_j keeps every row feasible: in a row with only an upper side a_ij <= a_ik,
 * with only a lower side a_ij >= a_ik, in an equation or a range a_ij = a_ik (an entry not there is zero). In an optimal
 * solution with x_k > l_k the move of x_k - l_k to x_j is then feasible and no dearer when x_j has no upper bound, or when
 * the rows imply its upper bound from the other columns' bounds (the moved point satisfies those rows and bounds, so x_j
 * stays within it): some optimal solution has x_k = l_k, and x_k is fixed there. One column at a time (two equal columns
 * dominate each other; the first fixed leaves the other). The implied bounds are those at the call: a column fixed at its
 * bound only tightens them. The candidates j of a column k are the columns of k's shortest row. Gurobi's dual reductions
 * remove such columns (piperout-08, LP: zero-cost binary columns in rows of two entries, Gurobi 6,711 x 7,626 against our
 * 11,489 x 9,429) */
static int pair_dominated(W *w, SxPre *P) {
    static int on = -1; if (on < 0) on = getenv("BRISK_SXPRE_PAIRDOM") && atoi(getenv("BRISK_SXPRE_PAIRDOM")) == 1;
    if (!on) return 0;
    const int m = w->m, n = w->n;
    double *ilo = xmalloc(sizeof(double) * (n + 1)), *iup = xmalloc(sizeof(double) * (n + 1));
    implied_col_bounds(w, ilo, iup, NULL, NULL);
    int *stk = xcalloc(m + 1, sizeof(int)), *stj = xcalloc(m + 1, sizeof(int)); double *vk = xmalloc(sizeof(double) * (m + 1));
    long work = 0;
    int nfix = 0, sk = 0, sj = 0;
    long nnz = 0; for (int j = 0; j < n; j++) nnz += w->clen[j];
    /* (the work, the pairs tried and the entries compared, at most about four times the matrix's entries: osa-60, LP, tried
     * 2.4 million pairs and compared 1.5 million in 0.15 s for 20,243 columns, the solve no faster; neos-2987310-joes, LP,
     * 9,368 columns in 0.95 million) */
    const long wlim = 4 * nnz + 1000000L;
    /* a filter by row patterns, one bit per row modulo 64: j can dominate k only if k has an entry in every row where j's
     * entry cannot be matched by a zero (hu: a_ij > 0 in a row with an upper side only, < 0 with a lower side only, any in a
     * two-sided row), and j in every row where k's cannot (he: the opposite signs, any in a two-sided row) */
    uint64_t *mk = xcalloc(n + 1, sizeof(uint64_t)), *hu = xcalloc(n + 1, sizeof(uint64_t)), *he = xcalloc(n + 1, sizeof(uint64_t));
    for (int j = 0; j < n; j++) {
        if (!w->colon[j]) continue;
        for (int p = 0; p < w->clen[j]; p++) {
            const int i = w->ci[j][p]; if (!w->rowon[i]) continue;
            const uint64_t b = (uint64_t)1 << (i & 63); const double a = w->cv[j][p];
            const int hasl = w->rl[i] > -PHUGE, hasu = w->ru[i] < PHUGE;
            mk[j] |= b;
            if (hasl && hasu) { hu[j] |= b; he[j] |= b; }
            else if (hasu) { if (a > 0) hu[j] |= b; else he[j] |= b; }
            else if (hasl) { if (a < 0) hu[j] |= b; else he[j] |= b; }
        }
    }
    work += nnz;
    for (int k = 0; k < n && work < wlim; k++) {
        if (!w->colon[k] || w->ccnt[k] < 1 || w->ccnt[k] > 32 || w->l[k] <= -PHUGE) continue;
        /* k's rows marked with its entries; its shortest row */
        sk++; int i0 = -1;
        for (int p = 0; p < w->clen[k]; p++) {
            const int i = w->ci[k][p]; if (!w->rowon[i]) continue;
            stk[i] = sk; vk[i] = w->cv[k][p];
            if (i0 < 0 || w->rcnt[i] < w->rcnt[i0]) i0 = i;
        }
        if (i0 < 0 || w->rcnt[i0] > 64) continue;
        const double ck = w->c[k], ctol = 1e-12 * (1 + fabs(ck));
        for (int q = 0; q < w->rlen[i0]; q++) {
            const int j = w->ri[i0][q];
            if (j == k || !w->colon[j] || w->ccnt[j] > 64 || w->c[j] > ck + ctol) continue;
            work++;
            if ((hu[j] & ~mk[k]) || (he[k] & ~mk[j])) continue;
            if (w->u[j] < PHUGE && !(iup[j] <= w->u[j] + FTOL * (1 + fabs(w->u[j])))) continue;
            sj++; int ok = 1, cov = 0;
            for (int p = 0; p < w->clen[j] && ok; p++) {
                const int i = w->ci[j][p]; if (!w->rowon[i]) continue;
                stj[i] = sj;
                const double aj = w->cv[j][p], ak = stk[i] == sk ? vk[i] : 0.0;
                if (stk[i] == sk) cov++;
                const int hasl = w->rl[i] > -PHUGE, hasu = w->ru[i] < PHUGE;
                const double tol = 1e-12 * (1 + fmax(fabs(aj), fabs(ak)));
                if (hasl && hasu) ok = fabs(aj - ak) <= tol;
                else if (hasu) ok = aj <= ak + tol;
                else if (hasl) ok = aj >= ak - tol;
            }
            work += w->clen[j] + w->clen[k];
            if (ok && cov < w->ccnt[k]) {
                for (int p = 0; p < w->clen[k] && ok; p++) {
                    const int i = w->ci[k][p]; if (!w->rowon[i] || stj[i] == sj) continue;
                    const double ak = w->cv[k][p];
                    const int hasl = w->rl[i] > -PHUGE, hasu = w->ru[i] < PHUGE;
                    const double tol = 1e-12 * (1 + fabs(ak));
                    if (hasl && hasu) ok = fabs(ak) <= tol;
                    else if (hasu) ok = 0.0 <= ak + tol;
                    else if (hasl) ok = 0.0 >= ak - tol;
                }
            }
            if (!ok) continue;
            rec_begin(P, R_FIXCOL); rec_i(P, k); rec_i(P, fixstat(w->l[k], w->l[k], w->u[k], P->l[k], P->u[k])); rec_d(P, w->l[k]);
            drop_col(w, k, w->l[k]); nfix++;
            break;
        }
    }
    free(ilo); free(iup); free(stk); free(stj); free(vk); free(mk); free(hu); free(he);
    return nfix;
}
/* the largest value over the box of the active columns' bounds of t1 * (row i1 without column j) + t2 * (row i2 without
 * column j) (i2 < 0: none); PINF when a column with a coefficient is unbounded on that side. cf, tch: scratch (zero) */
static double box_sup(const W *w, int j, int i1, double t1, int i2, double t2, double *cf, int *tch) {
    int nt = 0;
    for (int h = 0; h < 2; h++) {
        const int i = h ? i2 : i1; const double t = h ? t2 : t1;
        if (i < 0) continue;
        for (int p = 0; p < w->rlen[i]; p++) {
            const int k = w->ri[i][p]; if (k == j || !w->colon[k]) continue;
            if (cf[k] == 0) tch[nt++] = k;
            cf[k] += t * w->rv[i][p];
            if (cf[k] == 0) cf[k] = 1e-300;      /* (kept in the list; read as zero below) */
        }
    }
    double sup = 0;
    for (int q = 0; q < nt; q++) {
        const int k = tch[q]; const double a = cf[k]; cf[k] = 0;
        if (fabs(a) <= 1e-14 * (fabs(t1) + fabs(t2))) continue;
        const double b = a > 0 ? w->u[k] : w->l[k];
        if (fabs(b) >= PHUGE) sup = PINF;
        if (sup < PHUGE) sup += a * b;
    }
    return sup;
}
/* (7 November 2026) BRISK_SXPRE_TIGHTZERO=0: the rule that makes a row an equation by a cost (below) leaves out the columns of cost zero */
static int sx_tightzero_off(void) { static int off = -1; if (off < 0) { const char *e = getenv("BRISK_SXPRE_TIGHTZERO"); off = e && atoi(e) == 0; } return off; }
/* (27 October 2026; BRISK_SXPRE_ZFREE=0 turns it off) a column of cost zero whose rows its value can always satisfy:
 * for every point of the other columns within their bounds, some x_j within its bounds satisfies every row of x_j. Then
 * x_j and its rows are removed, and the postsolve gives x_j such a value. Each side of a row gives x_j a lower or an upper
 * bound linear in the other columns, L(x) = (b - s(x)) / a with s the row's other entries; the condition: on the box of
 * the other columns' bounds, every lower bound at most x_j's upper bound, every upper bound at least x_j's lower bound,
 * and for two rows the largest value of L - U at most zero (the rows of the elimination of x_j (Fourier-Motzkin) all
 * redundant by the bounds). Gurobi's dual reductions remove such columns (piperout-08, LP: c168 in the rows
 * c168 <= c166 + c167, c166 <= c168, c167 <= c168, an OR of two binaries, always satisfied by c168 = max(c166, c167));
 * removing one leaves its neighbours dominated or in singleton rows, and the reductions cascade. Not with a row that is
 * the source of a dropped bound (the bound would go with the row). The bounds of the other columns are those of the
 * problem now (a bound the presolve tightened came with its row removed: a constraint of the problem now). On the 19 LPs
 * of the four sets whose presolved size it changes by 1 % or more (two interleaved passes, four threads, the medians):
 * 0.90 times the time without it, 1.39 -> 1.25 times Gurobi's (piperout-08 0.213 -> 0.090 s, neos-4722843-widden 0.46
 * -> 0.27, bnl2 0.068 -> 0.042, lectsched-5-obj 0.082 -> 0.053; slower: cycle 0.042 -> 0.053) */
static int zero_cost_free(W *w, SxPre *P, const char *rowdropsrc) {
    static int on = -1; if (on < 0) { const char *e = getenv("BRISK_SXPRE_ZFREE"); on = e ? atoi(e) != 0 : 1; }
    if (!on) return 0;
    const int n = w->n;
    double *cf = xcalloc(n + 1, sizeof(double)); int *tch = xmalloc(sizeof(int) * (n + 1));
    int nrem = 0;
    /* (the work, the entries of the rows read, at most about 20 times the matrix's per call) */
    long nnz = 0, work = 0; for (int j = 0; j < n; j++) nnz += w->clen[j];
    const long wlim = 20 * nnz + 1000000L;
    for (int j = 0; j < n && work < wlim; j++) {
        if (!w->colon[j] || w->c[j] != 0 || w->ccnt[j] < 1 || w->ccnt[j] > 8) continue;
        int rr[8]; double aa[8]; int k = 0, ok = 1;
        for (int p = 0; p < w->clen[j] && ok; p++) {
            const int i = w->ci[j][p]; if (!w->rowon[i]) continue;
            if (rowdropsrc[i] || w->rcnt[i] > 32 || k >= 8) ok = 0; else { rr[k] = i; aa[k] = w->cv[j][p]; k++; }
        }
        if (!ok || k == 0) continue;
        for (int t = 0; t < k; t++) work += (long)(2 * k + 2) * w->rlen[rr[t]];
        /* each row's sides as x_j's lower (bl) and upper (bu) bounds, L = bl/a - s/a and U = bu/a - s/a, and the largest
         * values of -s/a (sm) and s/a (sp) on the box, each row alone */
        double bl[8], bu[8], sm[8], sp[8];
        for (int t = 0; t < k; t++) {
            const int i = rr[t]; const double a = aa[t];
            bl[t] = a > 0 ? w->rl[i] : w->ru[i]; bu[t] = a > 0 ? w->ru[i] : w->rl[i];
            double vm = 0, vp = 0;
            for (int p = 0; p < w->rlen[i]; p++) {
                const int c = w->ri[i][p]; if (c == j || !w->colon[c]) continue;
                const double e = w->rv[i][p] / a, lo = w->l[c], up = w->u[c];
                const double bm = e > 0 ? lo : up, bp = e > 0 ? up : lo;     /* -e x largest at bm, e x at bp */
                if (fabs(bm) >= PHUGE) vm = PINF; else if (vm < PHUGE) vm -= e * bm;
                if (fabs(bp) >= PHUGE) vp = PINF; else if (vp < PHUGE) vp += e * bp;
            }
            sm[t] = vm; sp[t] = vp;
        }
        const double lj = w->l[j], uj = w->u[j];
        for (int t = 0; t < k && ok; t++) {
            /* L <= u_j: bl/a + sup(-s/a) <= u_j; U >= l_j: sup(s/a) - bu/a <= -l_j */
            if (fabs(bl[t]) < PHUGE && uj < PHUGE) { const double r = uj - bl[t] / aa[t]; if (sm[t] > r + FTOL * (1 + fabs(r))) ok = 0; }
            if (ok && fabs(bu[t]) < PHUGE && lj > -PHUGE) { const double r = -lj + bu[t] / aa[t]; if (sp[t] > r + FTOL * (1 + fabs(r))) ok = 0; }
            for (int u = 0; u < k && ok; u++) {
                if (u == t || fabs(bl[t]) >= PHUGE || fabs(bu[u]) >= PHUGE) continue;
                /* L_t - U_u = bl_t/a_t - bu_u/a_u - s_t/a_t + s_u/a_u <= 0: the two rows' largest values added bound it; where
                 * that fails, the combined row's own (its shared columns may cancel) */
                const double r = bu[u] / aa[u] - bl[t] / aa[t], tol = FTOL * (1 + fabs(r));
                if (sm[t] < PHUGE && sp[u] < PHUGE && sm[t] + sp[u] <= r + tol) continue;
                if (box_sup(w, j, rr[t], -1 / aa[t], rr[u], 1 / aa[u], cf, tch) > r + tol) ok = 0;
            }
        }
        if (!ok) continue;
        rec_begin(P, R_ZFREE); rec_i(P, j); rec_i(P, k); rec_d(P, lj); rec_d(P, uj);
        for (int t = 0; t < k; t++) {
            const int i = rr[t]; int cnt = 0;
            for (int p = 0; p < w->rlen[i]; p++) if (w->colon[w->ri[i][p]] && w->ri[i][p] != j) cnt++;
            rec_i(P, i); rec_i(P, cnt); rec_d(P, aa[t]); rec_d(P, w->rl[i]); rec_d(P, w->ru[i]);
            for (int p = 0; p < w->rlen[i]; p++) { const int c = w->ri[i][p]; if (w->colon[c] && c != j) { rec_i(P, c); rec_d(P, w->rv[i][p]); } }
        }
        for (int t = 0; t < k; t++) drop_row(w, rr[t]);
        drop_col(w, j, 0);
        nrem++;
    }
    free(cf); free(tch);
    return nrem;
}
static int parallel_cols(W *w, SxPre *P) {
    const int n = w->n;
    static int dompar = -1; if (dompar < 0) { const char *e = getenv("BRISK_SXPRE_DOMPAR"); dompar = !(e && atoi(e) == 0); }
    unsigned long long *h = xmalloc(sizeof(unsigned long long) * (n + 1)); double *sc = xmalloc(sizeof(double) * (n + 1));
    int *ord = xmalloc(sizeof(int) * (n + 1)), no = 0;
    for (int j = 0; j < n; j++) {
        if (!w->colon[j] || w->ccnt[j] < 1) continue;
        int imin = -1; double amin = 0;
        for (int p = 0; p < w->clen[j]; p++) { const int i = w->ci[j][p]; if (w->rowon[i] && w->cv[j][p] != 0 && (imin < 0 || i < imin)) { imin = i; amin = w->cv[j][p]; } }
        if (imin < 0) continue;
        sc[j] = 1.0 / amin;
        unsigned long long hh = 0; int c = 0;
        for (int p = 0; p < w->clen[j]; p++) { const int i = w->ci[j][p]; if (w->rowon[i] && w->cv[j][p] != 0) { hh += pmix(((unsigned long long)i << 32) ^ pmix(pbitsq(w->cv[j][p] * sc[j]))); c++; } }
        h[j] = hh + pmix((unsigned long long)c); ord[no++] = j;
    }
    {   RefKey *v = xmalloc(sizeof(RefKey) * ((size_t)no + 1));
        for (int t = 0; t < no; t++) { v[t].a = h[ord[t]]; v[t].b = (unsigned long long)ord[t]; }
        rk_sort(v, no);
        for (int t = 0; t < no; t++) ord[t] = (int)v[t].b;
        free(v); }
    double *val = xcalloc(w->m + 1, sizeof(double)); char *mk = xcalloc(w->m + 1, 1);
    int ndrop = 0;
    for (int t = 0; t < no; ) {
        int e = t; while (e < no && h[ord[e]] == h[ord[t]]) e++;
        if (e - t > 1) {
            int j = ord[t];
            int cj = 0;
            for (int p = 0; p < w->clen[j]; p++) { const int i = w->ci[j][p]; if (w->rowon[i] && w->cv[j][p] != 0) { val[i] = w->cv[j][p] * sc[j]; mk[i] = 1; cj++; } }
            for (int u = t + 1; u < e; u++) {
                const int k = ord[u];
                int ck = 0, same = 1;
                for (int p = 0; p < w->clen[k] && same; p++) { const int i = w->ci[k][p]; if (!w->rowon[i] || w->cv[k][p] == 0) continue; ck++; if (!mk[i] || !PAR_EQ(val[i], w->cv[k][p] * sc[k])) same = 0; }
                if (!same || ck != cj) continue;
                const double lam = sc[j] / sc[k];        /* a_k = lam a_j */
                const double delta = w->c[k] - lam * w->c[j];
                if (fabs(delta) > 1e-12 * (1 + fabs(w->c[k]))) {
                    if (!dompar || fabs(delta) <= 1e-9 * (1 + fabs(w->c[k]) + fabs(lam * w->c[j]))) continue;
                    /* the sign of z_j at every dual feasible point that x_j's bounds give: 1 (z_j >= 0, no upper bound), -1
                     * (z_j <= 0, no lower bound), 0 (z_j = 0, free), 2 (none, both bounds finite); the same for z_k */
                    const int sj = w->u[j] >= PHUGE ? (w->l[j] <= -PHUGE ? 0 : 1) : (w->l[j] <= -PHUGE ? -1 : 2);
                    const int sk = w->u[k] >= PHUGE ? (w->l[k] <= -PHUGE ? 0 : 1) : (w->l[k] <= -PHUGE ? -1 : 2);
                    /* z_k = delta + lam z_j, and z_j = -delta/lam + z_k/lam (1/lam has the sign of lam) */
                    const int jpos = sj == 0 || (sj == 1 && lam > 0) || (sj == -1 && lam < 0);   /* lam z_j >= 0 */
                    const int jneg = sj == 0 || (sj == 1 && lam < 0) || (sj == -1 && lam > 0);   /* lam z_j <= 0 */
                    const int kpos = sk == 0 || (sk == 1 && lam > 0) || (sk == -1 && lam < 0);   /* z_k / lam >= 0 */
                    const int kneg = sk == 0 || (sk == 1 && lam < 0) || (sk == -1 && lam > 0);   /* z_k / lam <= 0 */
                    const double dj = -delta / lam;
                    int f = -1, side = 0;  /* the column fixed, and where: -1 at its lower bound, 1 at its upper bound */
                    if (jpos && delta > 0) f = k, side = -1;            /* z_k > 0 */
                    else if (jneg && delta < 0) f = k, side = 1;        /* z_k < 0 */
                    else if (kpos && dj > 0) f = j, side = -1;          /* z_j > 0 */
                    else if (kneg && dj < 0) f = j, side = 1;           /* z_j < 0 */
                    if (f < 0) continue;
                    const double v = side < 0 ? w->l[f] : w->u[f];
                    if (fabs(v) >= PHUGE) continue;     /* (no bound on that side: the problem is unbounded or infeasible; left to the solver) */
                    rec_begin(P, R_FIXCOL); rec_i(P, f); rec_i(P, fixstat(v, w->l[f], w->u[f], P->l[f], P->u[f])); rec_d(P, v);
                    drop_col(w, f, v);
                    ndrop++;
                    if (f == j) j = k;      /* (the same scaled entries: the marks stay) */
                    continue;
                }
                const double lj = w->l[j], uj = w->u[j], lk = w->l[k], uk = w->u[k];
                rec_begin(P, R_PARCOL); rec_i(P, j); rec_i(P, k); rec_d(P, lam); rec_d(P, lj); rec_d(P, uj); rec_d(P, lk); rec_d(P, uk);
                const double klo = lam > 0 ? lk : uk, kup = lam > 0 ? uk : lk;     /* the k giving the least / largest sum */
                w->l[j] = (lj <= -PHUGE || fabs(klo) >= PHUGE) ? -PINF : lj + lam * klo;
                w->u[j] = (uj >= PHUGE || fabs(kup) >= PHUGE) ? PINF : uj + lam * kup;
                w->colon[k] = 0;
                for (int p = 0; p < w->clen[k]; p++) if (w->rowon[w->ci[k][p]]) w->rcnt[w->ci[k][p]]--;
                ndrop++;
            }
            for (int p = 0; p < w->clen[j]; p++) { const int i = w->ci[j][p]; mk[i] = 0; val[i] = 0; }
        }
        t = e;
    }
    free(h); free(sc); free(ord); free(val); free(mk);
    return ndrop;
}
/* a column with cost zero and a single entry is a slack of its row: dropped, the row's bounds widen by its range
 * (rl - max a s <= rest <= ru - min a s over l <= s <= u). Record: i, j, the other active columns of the row (count,
 * columns, entries); a, l, u and the row's bounds before. */
static void slack_singleton(W *w, SxPre *P, int i, int j, double a) {
    /* (24 October 2026) slack columns of one row removed one after another form a run: each record holds the column's own
     * data, the row's other columns are recorded once when the run ends (R_SLACKREST), and the postsolve adds each restored
     * slack's activity to the row's rest. Before, each record listed the whole row: a row with k slack columns cost k times
     * its length (neos-787933, LP: 172,668 slack columns in 1,897 rows, 31 million values). BRISK_SXPRE_SLACKRUN=0: the
     * whole row in every record */
    static int srun = -1; if (srun < 0) { const char *e = getenv("BRISK_SXPRE_SLACKRUN"); srun = !(e && atoi(e) == 0); }
    if (!srun) {
        rec_begin(P, R_SLACK); rec_i(P, i); rec_i(P, j);
        int cnt = 0;
        for (int p = 0; p < w->rlen[i]; p++) if (w->ri[i][p] != j && w->colon[w->ri[i][p]]) cnt++;
        rec_i(P, cnt);
        rec_d(P, a); rec_d(P, w->l[j]); rec_d(P, w->u[j]); rec_d(P, w->rl[i]); rec_d(P, w->ru[i]);
        for (int p = 0; p < w->rlen[i]; p++) { const int k = w->ri[i][p]; if (k != j && w->colon[k]) { rec_i(P, k); rec_d(P, w->rv[i][p]); } }
    } else {
        if (P->run_i != i) {
            run_close(P);
            if (!P->run_pos) { P->run_pos = xmalloc(sizeof(int) * (P->n + 1)); for (int k = 0; k < P->n; k++) P->run_pos[k] = -1; }
            if (w->rlen[i] > P->run_cap) { P->run_cap = w->rlen[i] + 64; P->run_col = xrealloc(P->run_col, sizeof(int) * P->run_cap); P->run_val = xrealloc(P->run_val, sizeof(double) * P->run_cap); }
            P->run_n = 0;
            for (int p = 0; p < w->rlen[i]; p++) { const int k = w->ri[i][p]; if (w->colon[k]) { P->run_pos[k] = P->run_n; P->run_col[P->run_n] = k; P->run_val[P->run_n++] = w->rv[i][p]; } }
            P->run_i = i;
        }
        const int t = P->run_pos[j];
        if (t >= 0) {       /* (j leaves the row's rest: the last entry takes its place) */
            const int last = --P->run_n;
            P->run_col[t] = P->run_col[last]; P->run_val[t] = P->run_val[last]; P->run_pos[P->run_col[t]] = t; P->run_pos[j] = -1;
        }
        rec_begin(P, R_SLACK2); rec_i(P, i); rec_i(P, j);
        rec_d(P, a); rec_d(P, w->l[j]); rec_d(P, w->u[j]); rec_d(P, w->rl[i]); rec_d(P, w->ru[i]);
    }
    const double l = w->l[j], u = w->u[j];
    const double smax = a > 0 ? (u >= PHUGE ? PINF : a * u) : (l <= -PHUGE ? PINF : a * l);     /* max a s */
    const double smin = a > 0 ? (l <= -PHUGE ? -PINF : a * l) : (u >= PHUGE ? -PINF : a * u);   /* min a s */
    if (w->rl[i] > -PHUGE) w->rl[i] = smax >= PHUGE ? -PINF : w->rl[i] - smax;
    if (w->ru[i] < PHUGE) w->ru[i] = smin <= -PHUGE ? PINF : w->ru[i] - smin;
    w->colon[j] = 0; w->rcnt[i]--;
}
/* a column with a cost and a single entry, in an equation or in an inequality its cost makes tight (taken as the
 * equation at the bound rhs): x_j = (rhs - rest) / a, so its cost moves to the row's other columns, and it is then a
 * slack of its row (slack_singleton, after this: its bounds become the row's range over the rest). Record: i, j; a, c_j,
 * the row's bounds before, rhs. */
static void cost_singleton(W *w, SxPre *P, int i, int j, double a, double rhs) {
    const double cj = w->c[j];
    rec_begin(P, R_COSTSING); rec_i(P, i); rec_i(P, j); rec_d(P, a); rec_d(P, cj); rec_d(P, w->rl[i]); rec_d(P, w->ru[i]); rec_d(P, rhs);
    for (int p = 0; p < w->rlen[i]; p++) { const int k = w->ri[i][p]; if (k != j && w->colon[k]) cost_sub(w, k, cj * w->rv[i][p] / a); }
    w->off += cj * rhs / a;
    w->c[j] = 0;
    w->rl[i] = w->ru[i] = rhs;
}
static int sxpre_presolve_m(const LpProb *L, LpProb *R, SxPre **out, int verbose, int aggbmode, char *why, size_t nwhy);
/* (24 October 2026, a diagnostic: BRISK_SXPRE_STATS=1) the structure of the presolved problem, to compare with another
 * solver's: rows by kind and length, columns by bounds and length, column singletons by the kind of their row */
static void presolved_stats(const LpProb *R) {
    const int m = R->m, n = R->n;
    int *rlen = calloc(m + 1, sizeof(int)); long rk[4] = {0}, rh[6] = {0}, ch[6] = {0}, cb[5] = {0}, c0 = 0, cs[4] = {0};
    for (int p = 0; p < R->Ap[n]; p++) rlen[R->Ai[p]]++;
    #define KIND(i) (R->rl[i] == R->ru[i] ? 0 : (R->rl[i] <= -1e30 || R->ru[i] >= 1e30) ? ((R->rl[i] <= -1e30 && R->ru[i] >= 1e30) ? 3 : 1) : 2)
    #define BIN(k) ((k) <= 1 ? (k) : (k) == 2 ? 2 : (k) == 3 ? 3 : (k) <= 10 ? 4 : 5)
    for (int i = 0; i < m; i++) { rk[KIND(i)]++; rh[BIN(rlen[i])]++; }
    for (int j = 0; j < n; j++) {
        const int len = R->Ap[j + 1] - R->Ap[j]; ch[BIN(len)]++;
        const int hl = R->l[j] > -1e30, hu = R->u[j] < 1e30;
        cb[R->l[j] == R->u[j] ? 4 : hl && hu ? 3 : hl ? 1 : hu ? 2 : 0]++;
        if (R->c[j] == 0) c0++;
        if (len == 1) cs[KIND(R->Ai[R->Ap[j]])]++;
    }
    printf("presolved: %d rows (%ld equations, %ld one-sided, %ld ranged, %ld free); row lengths 0:%ld 1:%ld 2:%ld 3:%ld 4-10:%ld >10:%ld\n", m, rk[0], rk[1], rk[2], rk[3], rh[0], rh[1], rh[2], rh[3], rh[4], rh[5]);
    printf("presolved: %d columns (%ld free, %ld lower only, %ld upper only, %ld boxed, %ld fixed; %ld of cost zero); lengths 0:%ld 1:%ld 2:%ld 3:%ld 4-10:%ld >10:%ld; singletons in equations %ld, one-sided rows %ld, ranged %ld\n",
           n, cb[0], cb[1], cb[2], cb[3], cb[4], c0, ch[0], ch[1], ch[2], ch[3], ch[4], ch[5], cs[0], cs[1], cs[2]);
    /* three ranged rows, each column with its bounds, cost, length and the other rows it meets */
    int shown = 0;
    int *rp = calloc(m + 2, sizeof(int)), *rc = malloc(sizeof(int) * (R->Ap[n] + 1)), *fill = calloc(m + 1, sizeof(int));
    for (int i = 0; i < m; i++) rp[i + 1] = rp[i] + rlen[i];
    for (int j = 0; j < n; j++) for (int p = R->Ap[j]; p < R->Ap[j + 1]; p++) { const int i = R->Ai[p]; rc[rp[i] + fill[i]++] = j; }
    for (int i = 0; i < m && shown < 3; i++) {
        static int want = -1; if (want < 0) { const char *e = getenv("BRISK_SXPRE_STATS"); want = e ? atoi(e) : 1; }
        static int wrow = -2; if (wrow == -2) { const char *e = getenv("BRISK_SXPRE_ROW"); wrow = e ? atoi(e) : -1; }
        if (wrow >= 0 ? i != wrow : want == 1 ? (KIND(i) != 2 || rlen[i] > 8) : rlen[i] != want) continue;
        shown++; printf("presolved: row %d, %d entries, [%g, %g]:\n", i, rlen[i], R->rl[i], R->ru[i]);
        for (int q = rp[i]; q < rp[i + 1]; q++) {
            const int j = rc[q]; double a = 0;
            for (int p = R->Ap[j]; p < R->Ap[j + 1]; p++) if (R->Ai[p] == i) a = R->Ax[p];
            printf("   column %d: a %g, [%g, %g], cost %g, %d entries;", j, a, R->l[j], R->u[j], R->c[j], R->Ap[j + 1] - R->Ap[j]);
            for (int p = R->Ap[j]; p < R->Ap[j + 1]; p++) if (R->Ai[p] != i) { const int k = R->Ai[p]; printf(" row %d (%s, %d entries, [%g, %g], a %g)", k, KIND(k) == 0 ? "equation" : KIND(k) == 1 ? "one-sided" : "ranged", rlen[k], R->rl[k], R->ru[k], R->Ax[p]); }
            printf("\n");
        }
    }
    free(rp); free(rc); free(fill);
    fflush(stdout);
    #undef KIND
    #undef BIN
    free(rlen);
}
static __thread int g_sxpre_simplex = 0;
int sxpre_set_for_simplex(int v) { const int o = g_sxpre_simplex; g_sxpre_simplex = v; return o; }
int sxpre_presolve(const LpProb *L, LpProb *R, SxPre **out, int verbose, char *why, size_t nwhy) {
    return sxpre_presolve_m(L, R, out, verbose, -1, why, nwhy);
}
/* (23 October 2026) aggbmode 1: the bounded substitution of columns without cost, at most 4 new entries (the simplex
 * method's own presolve of the race, simplex.c aggb_route); 0: none; -1: as the environment says (experiments) */
int sxpre_presolve_ex(const LpProb *L, LpProb *R, SxPre **out, int verbose, int aggbmode, char *why, size_t nwhy) {
    return sxpre_presolve_m(L, R, out, verbose, aggbmode, why, nwhy);
}
static int sxpre_presolve_m(const LpProb *L, LpProb *R, SxPre **out, int verbose, int aggbmode, char *why, size_t nwhy) {
    const double tq0 = omp_get_wtime();
    const int m = L->m, n = L->n;
    SxPre *P = xcalloc(1, sizeof(SxPre));
    P->m = m; P->n = n; P->run_i = -1;
    const double sg = L->maxim ? -1.0 : 1.0;
    W w; memset(&w, 0, sizeof w);
    w.m = m; w.n = n;
    w.ci = xcalloc(n + 1, sizeof(int *)); w.cv = xcalloc(n + 1, sizeof(double *)); w.clen = xcalloc(n + 1, sizeof(int)); w.ccap = xcalloc(n + 1, sizeof(int));
    w.ri = xcalloc(m + 1, sizeof(int *)); w.rv = xcalloc(m + 1, sizeof(double *)); w.rlen = xcalloc(m + 1, sizeof(int)); w.rcap = xcalloc(m + 1, sizeof(int));
    {   size_t nzn = 0; for (int p = 0; p < L->Ap[n]; p++) nzn += L->Ax[p] != 0;
        w.npool = 2 * nzn + 2 * ((size_t)m + n) + 1;
        w.ipool = xmalloc(sizeof(int) * w.npool); w.vpool = xmalloc(sizeof(double) * w.npool); }
    size_t poff = 0;
    for (int j = 0; j < n; j++) {
        int c = 0; for (int p = L->Ap[j]; p < L->Ap[j + 1]; p++) if (L->Ax[p] != 0) c++;
        w.ccap[j] = c + 2; w.ci[j] = w.ipool + poff; w.cv[j] = w.vpool + poff; poff += w.ccap[j];
        for (int p = L->Ap[j]; p < L->Ap[j + 1]; p++) if (L->Ax[p] != 0) { w.ci[j][w.clen[j]] = L->Ai[p]; w.cv[j][w.clen[j]] = L->Ax[p]; w.clen[j]++; w.rcap[L->Ai[p]]++; }
    }
    for (int i = 0; i < m; i++) { w.rcap[i] += 2; w.ri[i] = w.ipool + poff; w.rv[i] = w.vpool + poff; poff += w.rcap[i]; }
    for (int j = 0; j < n; j++) for (int p = 0; p < w.clen[j]; p++) { const int i = w.ci[j][p]; w.ri[i][w.rlen[i]] = j; w.rv[i][w.rlen[i]] = w.cv[j][p]; w.rlen[i]++; }
    w.ridx = xcalloc((size_t)m + 1, 1);
    /* (the table of a long row is built at its first search or change, not here: square47, LP, 27 million entries in rows of
     * thousands, none searched, spent 4.5 s of its presolve inserting them, 0.85 -> 5.3 s of set-up; 2 November 2026) */
    w.c = xmalloc(sizeof(double) * (n + 1)); w.l = xmalloc(sizeof(double) * (n + 1)); w.u = xmalloc(sizeof(double) * (n + 1));
    w.rl = xmalloc(sizeof(double) * (m + 1)); w.ru = xmalloc(sizeof(double) * (m + 1)); w.rmag = xmalloc(sizeof(double) * (m + 1));
    P->l = xmalloc(sizeof(double) * (n + 1)); P->u = xmalloc(sizeof(double) * (n + 1)); P->rl = xmalloc(sizeof(double) * (m + 1)); P->ru = xmalloc(sizeof(double) * (m + 1));
    {   const long nz = L->Ap[n];
        P->Ap = xmalloc(sizeof(int) * (n + 1)); P->Ai = xmalloc(sizeof(int) * (nz + 1)); P->Ax = xmalloc(sizeof(double) * (nz + 1)); P->c = xmalloc(sizeof(double) * (n + 1));
        memcpy(P->Ap, L->Ap, sizeof(int) * (n + 1)); memcpy(P->Ai, L->Ai, sizeof(int) * nz); memcpy(P->Ax, L->Ax, sizeof(double) * nz);
        for (int j = 0; j < n; j++) P->c[j] = sg * L->c[j]; }
    for (int j = 0; j < n; j++) { w.c[j] = sg * L->c[j]; w.l[j] = L->l[j] <= -PHUGE ? -PINF : L->l[j]; w.u[j] = L->u[j] >= PHUGE ? PINF : L->u[j]; }
    for (int i = 0; i < m; i++) { w.rl[i] = L->rl[i] <= -PHUGE ? -PINF : L->rl[i]; w.ru[i] = L->ru[i] >= PHUGE ? PINF : L->ru[i];
        w.rmag[i] = fmax(w.rl[i] > -PHUGE ? fabs(w.rl[i]) : 0.0, w.ru[i] < PHUGE ? fabs(w.ru[i]) : 0.0); }
    memcpy(P->l, w.l, sizeof(double) * n); memcpy(P->u, w.u, sizeof(double) * n); memcpy(P->rl, w.rl, sizeof(double) * m); memcpy(P->ru, w.ru, sizeof(double) * m);
    w.rowon = xmalloc(m + 1); w.colon = xmalloc(n + 1); memset(w.rowon, 1, m); memset(w.colon, 1, n);
    w.cmod = xcalloc(n + 1, 1);
    w.crec_i = xmalloc(sizeof(long) * (n + 1)); w.crec_d = xmalloc(sizeof(long) * (n + 1));
    for (int j = 0; j <= n; j++) w.crec_i[j] = w.crec_d[j] = -1;
    w.cver = xcalloc(n + 1, sizeof(int)); w.crec_ver = xcalloc(n + 1, sizeof(int)); w.crec_skip = xmalloc(sizeof(int) * (n + 1));
    w.rcnt = xmalloc(sizeof(int) * (m + 1)); w.ccnt = xmalloc(sizeof(int) * (n + 1));
    for (int i = 0; i < m; i++) w.rcnt[i] = w.rlen[i];
    for (int j = 0; j < n; j++) w.ccnt[j] = w.clen[j];
    int status = 0;
    int nsing = 0, nfix = 0, nfree = 0, nforce = 0, nred = 0, ndual = 0, nempty = 0, ndbl = 0;
    const int nodbl = getenv("BRISK_SXPRE_NODBL") != NULL, noimpl = getenv("BRISK_SXPRE_NOIMPLFREE") != NULL, nodom = getenv("BRISK_SXPRE_NODOM") != NULL;
    double *ylo = xmalloc(sizeof(double) * (m + 1)), *yup = xmalloc(sizeof(double) * (m + 1));
    double *ilo = xmalloc(sizeof(double) * (n + 1)), *iup = xmalloc(sizeof(double) * (n + 1));
    IbCache ibc = { -1, xmalloc(sizeof(double) * (n + 1)), xmalloc(sizeof(double) * (n + 1)), xmalloc(sizeof(int) * (n + 1)), xmalloc(sizeof(int) * (n + 1)) };
    double *y0l = xmalloc(sizeof(double) * (m + 1)), *y0u = xmalloc(sizeof(double) * (m + 1));
    int *ysl = xmalloc(sizeof(int) * (m + 1)), *ysu = xmalloc(sizeof(int) * (m + 1));
    char *inu = xmalloc(n + 1), *inl = xmalloc(n + 1);
    const int noweak = getenv("BRISK_SXPRE_NOWEAKDOM") != NULL, noagg = getenv("BRISK_SXPRE_NOAGG") != NULL;
    const int noslack = getenv("BRISK_SXPRE_NOSLACK") != NULL, notight = getenv("BRISK_SXPRE_NOTIGHT") != NULL;
    const int noside = getenv("BRISK_SXPRE_NOSIDE") != NULL; int nside = 0;
    const int notightrow = getenv("BRISK_SXPRE_NOTIGHTROW") != NULL; int ntightrow = 0;
    const int noimplred = getenv("BRISK_SXPRE_NOIMPLRED") != NULL; int nimplred = 0;
    const int nonegl = getenv("BRISK_SXPRE_NONEGL") != NULL; int nnegl = 0;
    const int nocompact = getenv("BRISK_SXPRE_NOCOMPACT") != NULL;
    int *dropsrc = xmalloc(sizeof(int) * (n + 1)); for (int j = 0; j < n; j++) dropsrc[j] = -1;
    char *trole = xcalloc(m + 1, 1);    /* the rows the cost made tight (2) and the rows such an argument rested on (1) */
    char *rowdropsrc = xcalloc(m + 1, 1), *rowanydrop = xcalloc(m + 1, 1);   /* (the guard of the dropped bounds) */   /* the row whose implication dropped x_j's bound */
    const int nocostsing = getenv("BRISK_SXPRE_NOCOSTSING") != NULL;
    const int noparallel = getenv("BRISK_SXPRE_NOPARALLEL") != NULL, nocascade = getenv("BRISK_SXPRE_NOCASCADE") != NULL;
    int nslack = 0, ntight = 0, npar = 0, nparc = 0, ncostsing = 0;
    int nagg = 0, naggb = 0, naggi = 0, naggi_bz[3] = { 0, 0, 0 };
    /* (5 November 2026, an experiment: BRISK_SXAGGBSX=1, simplex.c) the presolve of the simplex methods alone (g_sxpre_simplex
     * 2: the dual or the primal simplex method, whose presolved problem no interior-point method sees) makes the bounded
     * substitution of columns without cost, at most 4 new entries, as the second presolve of the bounded-substitution route
     * does (aggbmode 1) */
    const int sxonly = aggbmode < 0 && g_sxpre_simplex == 2 && !getenv("BRISK_SXAGGB");
    const int aggb = aggbmode >= 0 ? aggbmode == 1 : sxonly ? 1 : getenv("BRISK_SXAGGB") && atoi(getenv("BRISK_SXAGGB")) == 1;   /* (experiment) */
    const long aggbfill = aggbmode >= 0 || sxonly ? 4 : getenv("BRISK_SXAGGBFILL") ? atol(getenv("BRISK_SXAGGBFILL")) : 16;   /* (experiment: the fill allowed a bounded substitution) */
    const int aggbzc = aggbmode >= 0 || sxonly ? 1 : getenv("BRISK_SXAGGBZC") && atoi(getenv("BRISK_SXAGGBZC")) == 1;   /* (experiment: only columns without cost) */
    /* (5 November 2026, with BRISK_SXAGGBSX=1 and BRISK_SXAGGBLATE=1) the bounded substitution waits until the other steps (the dependent equations
     * included) find nothing more: a row it keeps as a range is no longer an equation for the slack columns, the doubleton
     * equations and the dependent equations (with the other steps: pds-06, LP, 2,358 -> 3,943 rows and 13,420 -> 19,252
     * columns, seqsolve1 8,829 -> 37,545 rows, its iterations 10,994 -> 27,758). Waiting keeps those LPs as they were but
     * loses most of the satellites' gain (satellites4-25, LP: 4,040-5,167 -> 6,105-8,936 iterations) and costs mzzv11 (LP)
     * 16,929 -> 70,855. Off: BRISK_SXAGGBLATE=1 waits */
    static int aglate = -1; if (aglate < 0) { const char *e_ = getenv("BRISK_SXAGGBLATE"); aglate = e_ && atoi(e_) == 1; }
    const int aggb_late = sxonly && aglate; int aggb_phase = 0;
    /* (5 November 2026) two limits on the bounded substitution, from ofi (LP), whose substituted problem the dual simplex
     * method took over 150 s on against 21 s without it:
     * (1) one of the column's bounds implied by the rows, so that the row becomes a one-sided inequality, not a range (as in
     *     Gurobi's aggregation, whose presolved problems hold no ranged rows: satellites2-25, LP, x1 + x2 + x3 - z = 0 with x1
     *     in [0, 1], x1 <= 1 implied, becomes z - x2 - x3 >= 0). With both bounds kept by the row (21,549 ranged rows on ofi)
     *     the pivot rows were 1.5 times longer and the run 1.35 times slower even without (2). BRISK_SXAGGBHALF=0: either;
     * (2) not into a long row: the substitution moves the row's other columns into each other row of the column, and in a
     *     row of tens of thousands of entries (ofi: 10,806 of its 35,000 substitutions, columns of two entries one of which
     *     is in a row of 97,000 to 112,000) they tie the basis to it, so that the rows of the basis inverse reach it far
     *     more often (average |rho| 2,194 -> 8,322, pivot rows 19,600 -> 89,000 entries). A row is long when it is dense by
     *     the rule of the AMD ordering: more than max(16, 10 sqrt(n)) entries, n the columns (ofi 4,698; satellites2-25
     *     1,638, whose substitutions reach rows of at most 99). A limit relative to the mean row length (10 times) cost
     *     satellites2-60-fs (LP, mean 6 entries a row) most of its gain. BRISK_SXAGGBLONG: the factor 10 (0: no limit);
     *     BRISK_SXAGGBMAXROW: an absolute limit instead (experiments) */
    const int aggbmaxrow0 = getenv("BRISK_SXAGGBMAXROW") ? atoi(getenv("BRISK_SXAGGBMAXROW")) : 0;
    const double aggblong = getenv("BRISK_SXAGGBLONG") ? atof(getenv("BRISK_SXAGGBLONG")) : 10.0;
    const int aggbmaxrow = aggbmaxrow0 > 0 ? aggbmaxrow0 : aggblong > 0 ? (int)fmin(2e9, fmax(16.0, ceil(aggblong * sqrt((double)n)))) : 0;
    const int aggbhalf = !(getenv("BRISK_SXAGGBHALF") && atoi(getenv("BRISK_SXAGGBHALF")) == 0);
    const int aggnet = getenv("BRISK_SXAGGNET") && atoi(getenv("BRISK_SXAGGNET")) == 1;   /* (experiment: substitutions by their net fill) */
    const int aggnet2 = getenv("BRISK_SXAGGNET2") ? atoi(getenv("BRISK_SXAGGNET2")) : 0;   /* (experiment: 1, only columns of two entries) */
    static int aggexact = -1; if (aggexact < 0) { const char *e_ = getenv("BRISK_SXAGGEXACT"); aggexact = !(e_ && atoi(e_) == 0); }
    long n_aggexact = 0;
    static int aggbex = -1; if (aggbex < 0) { const char *e_ = getenv("BRISK_SXAGGBEXACT"); aggbex = e_ && atoi(e_) != 0; }   /* (12 November 2026, an experiment: the exact count for the bounded substitution too) */
    static long aggexlim = -1000; if (aggexlim == -1000) { const char *e_ = getenv("BRISK_SXAGGEXACTLIM"); aggexlim = e_ ? atol(e_) : 0; }
    const double aggnettol = getenv("BRISK_SXAGGNETTOL") ? atof(getenv("BRISK_SXAGGNETTOL")) : 0.01;   /* (experiment: the substituted entry against its row's largest, for the net-fill substitutions) */
    /* (27 October 2026, an experiment) the substitution through a one-sided inequality row, the column kept as the row's
     * slack (the aggregator below); BRISK_SXPRE_AGGI=1: on */
    /* 1: every column with one infinite bound; 2: free columns only; 3: free columns only, in the simplex methods' presolve;
     * 9: as 3 with a rule on the row (4 November 2026, an experiment: aggibz below) */
    const int aggim = getenv("BRISK_SXPRE_AGGI") ? atoi(getenv("BRISK_SXPRE_AGGI")) : 9;   /* (9 by default since 7 November 2026, with the ratio test's last group in the order of its ratios: the default route on the 21 LPs whose presolve it changes 0.936 of the time, stockholm 1.56 -> 0.58 s, ns1644855 60.0 -> 63.7 s; BRISK_SXPRE_AGGI=0: off) */
    const int aggi = aggim == 1 || aggim == 2 || ((aggim == 3 || aggim == 9) && g_sxpre_simplex > 0);
    /* (4 November 2026, an experiment: BRISK_SXPRE_AGGI=9) as 3, a row not used when more than BRISK_SXPRE_AGGIBZ (default 0)
     * of its other columns are boxed without cost. The substitution replaces the free column, which the dual simplex method
     * keeps basic at no cost, by the row's other columns in the column's other rows; a column in [l, u] without cost takes
     * that place (the rows x <= 20 (z_1 + ... + z_8), z in [0, 1], of ns1644855, LP): basic far outside its bounds, the
     * steps that bring it back of length zero (no cost), the objective at -2.0e5 for 17,000 iterations and then near
     * -2,035 for 60,000 more (37,111 -> over 96,000 iterations with 3). With at most one such column 114 substitutions of
     * the 814 are left and take 79,658 iterations; with none 14, and 37,711 */
    const int aggibz = aggim == 9 ? (getenv("BRISK_SXPRE_AGGIBZ") ? atoi(getenv("BRISK_SXPRE_AGGIBZ")) : 0) : -1;
    const int aggifree = aggim >= 2;
    /* (12 November 2026) the substitution through an inequality row of a column with one infinite bound, in the simplex methods'
     * presolve, when it adds no entries on balance (the aggregator below) */
    /* Off by default (12 November 2026, four A/B runs on the 618 development LPs: 1.002; neos-4264598-oueme 0.61-0.70,
     * ger50_17_trans 0.54-0.73 and ger50-17-trans-dfn-3t (Gurobi's presolved size; 1,717 -> 1,108 iterations), but gmut-76-40
     * 1.21-1.26 from two substitutions and neos-5188808-nattai up to 1.24, var-smallemery-m6j6 1.33: the rule of no file
     * worse). BRISK_SXPRE_AGGINET=1: on */
    /* (13 November 2026) on by default for columns with a cost only (BRISK_SXPRE_AGGINETC, below): the substitution moves the
     * column's cost onto the row's other columns, which the dominated-column test can then fix (ger50-17-trans-dfn-3t, LP: about
     * 9,000 columns, Gurobi's presolved size); a column of cost zero moves no cost and only reshapes rows (gmut-76-40, LP: 7-11 %
     * more iterations on every permuted copy). Judged on the 618 development LPs (68 changed, no status changed; the 30 finished
     * ones by the default route 0.982, ger50-17-trans-dfn-3t 0.181 -> 0.091 s and ger50_17_trans 0.143 -> 0.086 s), the losers on
     * three permuted copies each (the originals' paths: level or gains on the copies), the held-out sets of 345 LPs untouched;
     * the cost a few milliseconds of later passes on small LPs (a1c1s1, cre-a). BRISK_SXPRE_AGGINET=0: off; BRISK_SXPRE_AGGINETC=0:
     * columns of cost zero as well (the form of 12 November) */
    const int agginet = g_sxpre_simplex > 0 && !(getenv("BRISK_SXPRE_AGGINET") && atoi(getenv("BRISK_SXPRE_AGGINET")) == 0);
    int agginet_phase = 0;
    long network = 0; const long netbudget = 10L * (long)L->Ap[L->n] + 1000L;
    int ndom = 0, ndomp = 0, nzfree = 0, pd_idle = 0, zf_idle = 0, nsparsify = 0, sp_idle = 0;
    int parstall = 0, depdone = 0, ndeprow = 0;
    /* (22 October 2026) a step that ran on the problem as it is now and found nothing finds nothing again: nch counts
     * the reductions, and the column pass, the dominated columns, the aggregator and the activity bounds are skipped
     * while nch is the count at which they last found nothing. A pass after a few reductions elsewhere repeated them in
     * full: lp_cheb_10000x300 (6.04 million nonzeros), one fixed column removed, a second pass of whole-matrix sweeps */
    long nch = 0, col_clean = -1, dom_clean = -1, agg_clean = -1, act_clean = -1;
    ActCache acache = { xmalloc(sizeof(double) * (m + 1)), xmalloc(sizeof(double) * (m + 1)), xmalloc(sizeof(double) * (m + 1)),
                        xmalloc(sizeof(int) * (m + 1)), xmalloc(sizeof(int) * (m + 1)), xcalloc(m + 1, 1),
                        xmalloc(sizeof(double) * (m + 1)), xmalloc(sizeof(double) * (m + 1)) };
    /* (25 October 2026) three costly steps that found nothing in their last run wait while the other steps still change
     * the problem (the previous pass changed it), and run again in a pass of every step before the presolve ends: the rows
     * the cost makes tight, the aggregator and the rows redundant by implied bounds sweep every row with the implied
     * bounds. A chain of
     * dominated columns and redundant rows took 14 passes on neos-3402294-bobin (LP: 591,076 rows, 586,984 -> 32,932 over
     * the passes), each sweeping all rows with both steps that found nothing. BRISK_SXPRE_NODEFER=1: every step each pass */
    static int nodefer = -1; if (nodefer < 0) nodefer = getenv("BRISK_SXPRE_NODEFER") != NULL;
    int tr_idle = 0, ir_idle = 0, ag_idle = 0, dm_idle = 0, prev_changed = 0;
    long dom_act = 0;   /* (the rows and columns left at the dominated-column test's last run) */
    long par_clean = -1;   /* (the count of changes at which the parallel rows and columns last found nothing) */
    /* (26 October 2026, a diagnostic: BRISK_SXPRE_TIMES=1) the time of each step over the passes, on stderr: the set-up,
     * rows, columns, dominated columns, rows the cost makes tight, doubleton equations, the aggregator, activity bounds,
     * rows redundant by implied bounds, parallel rows and columns, the reduced problem */
    static int ptimes = -1; if (ptimes < 0) ptimes = getenv("BRISK_SXPRE_TIMES") != NULL;
    double tt[12] = {0}, tq = ptimes ? omp_get_wtime() : 0;
    double tt_ib = 0;
#define PT(k) do { if (ptimes) { const double t_ = omp_get_wtime(); tt[k] += t_ - tq; tq = t_; } if (SXPRE_STOPPED() && status == 0) { status = 6; snprintf(why, nwhy, "stopped"); } } while (0)
    tt[0] = ptimes ? tq - tq0 : 0;
    for (int pass = 0; pass < 100 && status == 0; pass++) {
        /* (2 November 2026) a removed column's entries stay in its rows' lists (the scans skip them): a row more than half
         * such entries is compacted at the start of a pass, each entry removed once. neos-4984115-canche (LP: rows of
         * 50,000-100,000 entries, 268,859 columns to 13,866) computed its rows' activity bounds 476,898 times over the full
         * lists, 1.82 s of its 2.2 s presolve. BRISK_SXPRE_NOCOMPACT=1: off */
        if (!nocompact) for (int i = 0; i < m; i++) {
            if (!w.rowon[i] || w.rlen[i] <= 2 * w.rcnt[i] + 8) continue;
            for (int p = w.rlen[i] - 1; p >= 0; p--) if (!w.colon[w.ri[i][p]]) wrow_del(&w, i, p);
        }
        int changed = 0, deferred = 0;
        const int defer = !nodefer && prev_changed;
        int act0 = 0; for (int i = 0; i < m; i++) act0 += w.rowon[i]; for (int j = 0; j < n; j++) act0 += w.colon[j];
        PT(0); if (status) break;
        /* rows: empty, singleton */
        for (int i = 0; i < m && status == 0; i++) {
            if (!w.rowon[i]) continue;
            if (w.rl[i] > w.ru[i] + FTOL * (1 + fabs(w.ru[i]))) { status = 1; snprintf(why, nwhy, "row %d: lower bound above upper bound", i); break; }
            if (w.rcnt[i] == 0) {
                if (w.rl[i] > FTOL * (1 + fabs(w.rl[i]) + w.rmag[i]) || w.ru[i] < -FTOL * (1 + fabs(w.ru[i]) + w.rmag[i])) { status = 1; snprintf(why, nwhy, "row %d is empty and excludes 0", i); break; }   /* (rmag: as above) */
                rec_begin(P, R_EMPTYROW); rec_i(P, i);
                w.rowon[i] = 0; nempty++; changed = 1, nch++;
                continue;
            }
            if (w.rcnt[i] == 1) {
                int j;
                if (sing_row(&w, P, i, &j)) { status = 1; snprintf(why, nwhy, "column %d: the bounds from row %d are empty", j, i); break; }
                nsing++; changed = 1, nch++;
            }
        }
        if (status) break;
        PT(1); if (status) break;
        /* columns: fixed, empty, dual fixing, free and implied free singletons in equations */
        const long nch_c = nch; const int col_run = nch != col_clean;
        if (col_run) memset(acache.val, 0, m);
        for (int j = 0; j < n && status == 0 && col_run; j++) {
            if (!w.colon[j]) continue;
            if (ISFIXED(&w, j)) { act_leave(&w, &acache, j); fix_col(&w, P, j); nfix++; changed = 1, nch++; continue; }
            /* the directions in which the column can move without hurting a row */
            int down_ok = 1, up_ok = 1;
            for (int p = 0; p < w.clen[j] && (down_ok || up_ok); p++) {
                const int i = w.ci[j][p];
                if (!w.rowon[i]) continue;
                const double a = w.cv[j][p];
                if (a > 0) { if (w.rl[i] > -PHUGE) down_ok = 0; if (w.ru[i] < PHUGE) up_ok = 0; }
                else { if (w.ru[i] < PHUGE) down_ok = 0; if (w.rl[i] > -PHUGE) up_ok = 0; }
            }
            const double cj = w.c[j];
            if ((down_ok && cj > 0) || (down_ok && cj == 0 && w.ccnt[j] == 0 && w.l[j] > -PHUGE)) {
                if (w.l[j] <= -PHUGE) { status = 2; snprintf(why, nwhy, "column %d: unbounded below with a positive cost", j); break; }
                rec_begin(P, R_FIXCOL); rec_i(P, j); rec_i(P, P->l[j] == P->u[j] ? SX_FIXED : SX_LOWER); rec_d(P, w.l[j]);
                act_leave(&w, &acache, j); drop_col(&w, j, w.l[j]); ndual++; changed = 1, nch++; continue;
            }
            if ((up_ok && cj < 0) || (up_ok && cj == 0 && w.ccnt[j] == 0 && w.u[j] < PHUGE)) {
                if (w.u[j] >= PHUGE) { status = 2; snprintf(why, nwhy, "column %d: unbounded above with a negative cost", j); break; }
                rec_begin(P, R_FIXCOL); rec_i(P, j); rec_i(P, P->l[j] == P->u[j] ? SX_FIXED : SX_UPPER); rec_d(P, w.u[j]);
                act_leave(&w, &acache, j); drop_col(&w, j, w.u[j]); ndual++; changed = 1, nch++; continue;
            }
            if (w.ccnt[j] == 0) {      /* empty, zero cost, no finite bound: at zero */
                if (cj == 0) {
                    rec_begin(P, R_FIXCOL); rec_i(P, j); rec_i(P, SX_ZERO); rec_d(P, 0.0);
                    drop_col(&w, j, 0.0); nempty++; changed = 1, nch++;
                }
                continue;
            }
            if (w.ccnt[j] == 1) {
                int i = -1; double a = 0;
                for (int p = 0; p < w.clen[j]; p++) if (w.rowon[w.ci[j][p]]) { i = w.ci[j][p]; a = w.cv[j][p]; break; }
                if (i < 0) continue;
                const int eqrow = w.rl[i] == w.ru[i] && w.rl[i] > -PHUGE;
                double rhs = w.rl[i]; int tight = 0;
                if (!eqrow && cj != 0 && !notight) {
                    /* an inequality the cost makes tight: the column wants to go down (cost > 0) or up (< 0); when the
                     * row always gives the binding bound in that direction (its own bound there is implied) and keeps
                     * the column within its other bound, every optimum has the row at that bound (its dual is c_j/a):
                     * the row is an equation there, and the column is defined by it as below */
                    double lmin, lmax; int nmin, nmax;
                    act_get(&w, &acache, i, j, a, &lmin, &lmax, &nmin, &nmax);
                    const int down = cj > 0;
                    const double bs = (down == (a > 0)) ? w.rl[i] : w.ru[i];     /* the row bound that binds x_j */
                    if (fabs(bs) < PHUGE) {
                        /* x_j = (bs - rest) / a over rest in [lmin, lmax]: its least and largest value */
                        double xlo, xup; int flo, fup;
                        if (a > 0) { xlo = (bs - lmax) / a; flo = !nmax; xup = (bs - lmin) / a; fup = !nmin; }
                        else { xlo = (bs - lmin) / a; flo = !nmin; xup = (bs - lmax) / a; fup = !nmax; }
                        const double lj = w.l[j], uj = w.u[j];
                        const int okl = lj <= -PHUGE || (flo && xlo >= lj - FTOL * (1 + fabs(lj)));
                        const int oku = uj >= PHUGE || (fup && xup <= uj + FTOL * (1 + fabs(uj)));
                        /* (its bound in the cost's direction is never the binding one; the other one, when it is not
                         * implied, becomes the row's range below) */
                        if (down ? okl : oku) { rhs = bs; tight = bs == w.rl[i] ? 1 : 2; }
                    }
                }
                if (!eqrow && !tight) { if (cj == 0 && w.l[j] < w.u[j] && !noslack) { act_leave(&w, &acache, j); slack_singleton(&w, P, i, j, a); nslack++; changed = 1, nch++; } continue; }
                int freeok = w.l[j] <= -PHUGE && w.u[j] >= PHUGE;
                if (!freeok && !noimpl) {
                    /* implied free: the row and the other columns' bounds keep x_j within its own bounds */
                    double lmin, lmax; int nmin, nmax;
                    act_get(&w, &acache, i, j, a, &lmin, &lmax, &nmin, &nmax);
                    double xlo = -PINF, xup = PINF;          /* x_j = (rhs - rest) / a, rest in [lmin, lmax] */
                    if (a > 0) { if (!nmax) xlo = (rhs - lmax) / a; if (!nmin) xup = (rhs - lmin) / a; }
                    else { if (!nmin) xlo = (rhs - lmin) / a; if (!nmax) xup = (rhs - lmax) / a; }
                    const int lok = w.l[j] <= -PHUGE || (xlo > -PHUGE && xlo >= w.l[j] - FTOL * (1 + fabs(w.l[j])));
                    const int uok = w.u[j] >= PHUGE || (xup < PHUGE && xup <= w.u[j] + FTOL * (1 + fabs(w.u[j])));
                    freeok = lok && uok;
                }
                if (!freeok) {
                    if (cj == 0 && w.l[j] < w.u[j] && !noslack) { act_leave(&w, &acache, j); slack_singleton(&w, P, i, j, a); nslack++; changed = 1, nch++; }
                    /* (only when the row has one other column, which the row then bounds: the row goes too. In
                     * longer rows the problem is no smaller, only its costs move, and the method's path changed for
                     * the worse: 55 MIPLIB relaxations 0.543 -> 0.584 s in the shifted geometric mean, nw04 1.1 ->
                     * 3.2 s; in a doubleton row thor50dday 49.5 -> 0.15 s) */
                    else if (cj != 0 && w.l[j] < w.u[j] && !nocostsing && w.rcnt[i] == 2) { act_leave(&w, &acache, j); cost_singleton(&w, P, i, j, a, rhs); slack_singleton(&w, P, i, j, a); ncostsing++; changed = 1, nch++; }
                    continue;
                }
                /* record: i, j, a, rhs, c_j, then the row's other active columns and entries */
                rec_begin(P, R_FREESING); rec_i(P, i); rec_i(P, j); rec_d(P, a); rec_d(P, rhs); rec_d(P, cj); rec_d(P, tight);
                int cnt = 0;
                for (int p = 0; p < w.rlen[i]; p++) if (w.ri[i][p] != j && w.colon[w.ri[i][p]]) cnt++;
                rec_i(P, cnt);
                for (int p = 0; p < w.rlen[i]; p++) {
                    const int k = w.ri[i][p];
                    if (k == j || !w.colon[k]) continue;
                    rec_i(P, k); rec_d(P, w.rv[i][p]);
                    cost_sub(&w, k, cj * w.rv[i][p] / a);
                }
                w.off += cj * rhs / a;
                w.colon[j] = 0;
                drop_row(&w, i); acache.val[i] = 0;
                if (tight) ntight++; else nfree++;
                changed = 1, nch++;
            }
        }
        if (status) break;
        if (col_run && nch == nch_c) col_clean = nch;
        PT(2); if (status) break;
        /* dominated columns: with the implied bounds on the duals, a reduced cost that keeps its sign puts the column at
         * the bound it prefers (Andersen and Andersen 1995) */
        /* (26 October 2026) a run that found nothing waits, as the three costly steps below do, while the other steps still
         * change the problem: pds-20 (LP) swept the whole matrix for the implied bounds on the duals after each pass of
         * substitutions, finding nothing. BRISK_SXPRE_DOMNODEFER=1: each pass as before */
        static int domnodefer = -1; if (domnodefer < 0) domnodefer = getenv("BRISK_SXPRE_DOMNODEFER") != NULL;
        /* (13 November 2026, an experiment: BRISK_SXPRE_DOMSHRINK=f) a deferred run made anyway once the problem has shrunk by a
         * fraction f since the test's last run: its dual bounds rest on the columns and rows that are left, and a large removal
         * (bmocbd, LP: dual fixing of 126,223 columns in pass 0, the test finding nothing then and 101,148 dominated columns at
         * pass 5, the passes between on a problem twice the size) can make many columns dominated; a pass of a few substitutions
         * (pds-20, LP, the reason for the deferral) does not reach it */
        static double dsh_ = -1; if (dsh_ < 0) { const char *e_ = getenv("BRISK_SXPRE_DOMSHRINK"); dsh_ = e_ ? atof(e_) : 0; }
        const long nch_d = nch; const int dom_want = !nodom && nch != dom_clean;
        const int dom_shrunk = dsh_ > 0 && dom_act > 0 && (double)act0 < (1.0 - dsh_) * (double)dom_act;
        const int dom_run = dom_want && !(defer && dm_idle && !domnodefer && !dom_shrunk);
        if (dom_run) dom_act = act0;
        if (dom_want && !dom_run) deferred = 1;
        if (dom_run) {
            implied_col_bounds_c(&w, &ibc, nch, ilo, iup, NULL, NULL);
            inert_bounds(&w, inu, inl);
            dual_bounds(&w, ylo, yup, ilo, iup, y0l, y0u, ysl, ysu, inu, inl);
            for (int j = 0; j < n && status == 0; j++) {
                if (!w.colon[j] || w.ccnt[j] == 0) continue;
                double smin = 0, smax = 0; int nmin = 0, nmax = 0;
                for (int p = 0; p < w.clen[j]; p++) {
                    const int i = w.ci[j][p];
                    if (!w.rowon[i]) continue;
                    const double a = w.cv[j][p];
                    /* a bound derived from column j's own condition is not used for column j (it would be circular) */
                    const double yl = ysl[i] == j ? y0l[i] : ylo[i], yu = ysu[i] == j ? y0u[i] : yup[i];
                    const double lo = a > 0 ? yl : yu, hi = a > 0 ? yu : yl;
                    if (fabs(lo) >= PHUGE) nmin++; else smin += a * lo;
                    if (fabs(hi) >= PHUGE) nmax++; else smax += a * hi;
                }
                const double cj = w.c[j], tol = 1e-9 * (1 + fabs(cj));
                /* weakly dominated (z_j >= 0 only; HiGHS fixes those too): fixed one at a time. The fixing keeps the optimal
                 * value: an optimal dual solution of the problem without x_j satisfies the bounds that do not come from
                 * x_j's own condition, so z_j >= 0 there and x_j at its lower bound completes it to an optimal pair. Hence
                 * the bounds that came from x_j's condition are dropped (to the row's sign) before the next column is
                 * judged: fixing two at once can lose feasibility (x1 + x2 = 1, costs zero: each is weakly dominated
                 * through the other's condition). map10: 82,563 x 43,594 -> 50,040 x 26,698 (HiGHS 49,894 x 26,617). */
                const double wt = noweak ? tol : -tol;
                int fx = 0;
                if (nmax == 0 && cj - smax > wt && w.l[j] > -PHUGE) {   /* z_j >= 0: at the lower bound */
                    rec_begin(P, R_FIXCOL); rec_i(P, j); rec_i(P, fixstat(w.l[j], w.l[j], w.u[j], P->l[j], P->u[j])); rec_d(P, w.l[j]);
                    drop_col(&w, j, w.l[j]); ndom++; changed = 1, nch++; fx = 1;
                } else if (nmin == 0 && cj - smin < -wt && w.u[j] < PHUGE) {   /* z_j <= 0: at the upper bound */
                    rec_begin(P, R_FIXCOL); rec_i(P, j); rec_i(P, fixstat(w.u[j], w.l[j], w.u[j], P->l[j], P->u[j])); rec_d(P, w.u[j]);
                    drop_col(&w, j, w.u[j]); ndom++; changed = 1, nch++; fx = 1;
                }
                if (fx) for (int p = 0; p < w.clen[j]; p++) {
                    const int i = w.ci[j][p];
                    if (ysl[i] == j) { ylo[i] = y0l[i]; ysl[i] = -1; }
                    if (ysu[i] == j) { yup[i] = y0u[i]; ysu[i] = -1; }
                }
            }
        }
        if (dom_run && nch == nch_d) dom_clean = nch;
        if (dom_run) dm_idle = nch == nch_d;
        PT(3); if (status) break;
        /* (24 October 2026, the night) a row the cost makes tight: a column j with c_j > 0 whose decrease no other row of
         * it can stop (a <= row with a > 0 or a >= row with a < 0, without the other side), and whose lower bound the row r
         * implies (decreasing x_j until r is tight keeps x_j within its bounds): some optimum has r tight, and r becomes an
         * equation at that side (c_j < 0 likewise). A two-entry row then goes as a doubleton equation (neos-4387871-tavua,
         * LP: 2,000 rows y >= x / 13 with y of cost 1, Gurobi's presolved size). No record: a dual of r of the wrong sign for
         * the row as it was is the final check's (the problem as read from the postsolved basis). BRISK_SXPRE_NOTIGHTROW=1: off */
        const long nch_t = nch;
        const int tr_run = !notightrow && !(defer && tr_idle);
        if (!notightrow && !tr_run) deferred = 1;
        if (tr_run && status == 0) {
            int *sl = xmalloc(sizeof(int) * (n + 1)), *su = xmalloc(sizeof(int) * (n + 1));
            char *rchg = xcalloc(m + 1, 1);
            implied_col_bounds_c(&w, &ibc, nch, ilo, iup, sl, su);
            /* (and a column of cost zero, in either direction: moving it costs nothing; neos-1171448, LP: x0 <= 8 y with y of
             * cost zero, its other rows relaxed as it falls) */
            for (int jj = 0; jj < 2 * n; jj++) {
                const int j = jj % n;
                if (!w.colon[j] || (jj >= n && w.c[j] != 0)) continue;
                /* (7 November 2026; BRISK_SXPRE_TIGHTZERO=0) not for a column of cost zero: its row made an equation fixes the
                 * column, then substituted from it, at the value that makes the row tight, a fraction set by the row's other
                 * columns, where the inequality also has optima with the column at a bound (lectsched-1, MILP, cost zero
                 * throughout: 12,091 rows made equations, then 2,585 doubleton equations and 9,509 substitutions, 12,206 of
                 * the postsolved point's 28,718 columns fractional; without it 1,906) */
                if (w.c[j] == 0 && sx_tightzero_off()) continue;
                const int up = jj < n ? w.c[j] < 0 : 1;    /* the cost pushes x_j up (else down); cost zero: down, then up */
                const int r = up ? su[j] : sl[j];
                if (r < 0 || !w.rowon[r] || rchg[r] || w.rl[r] == w.ru[r]) continue;
                const double b = up ? w.u[j] : w.l[j], ib = up ? iup[j] : ilo[j];
                if (fabs(b) < PHUGE && (up ? ib > b + FTOL * (1 + fabs(b)) : ib < b - FTOL * (1 + fabs(b)))) continue;
                /* where r implies x_j's bound only as tight as it is (not strictly tighter), the bound is dropped, r keeping it:
                 * otherwise x_j could rest at it with r tight and r's dual of the wrong sign for the row as read (tavua: 1,104
                 * iterations of the final check); with no bound to rest at, x_j is basic or at its other bound, and r's dual
                 * has the sign of the row. Not when a column of r, or of another row of x_j, had a bound dropped from that row
                 * (the implied bounds would rest on each other) */
                const int eqb = fabs(b) < PHUGE && !(up ? ib < b - FTOL * (1 + fabs(b)) : ib > b + FTOL * (1 + fabs(b)));
                /* (BRISK_SXPRE_TIGHTEQ: 0 such a row is not made an equation, 1 it is and the bound kept, 2 the default, the
                 * bound dropped as above. 238 LPs (Netlib, Kennington, 124 MIPLIB relaxations, dual simplex method at one
                 * thread): the final check's iterations 1,099 with 1 (neos-873061, LP: 352 reduced costs of the wrong sign,
                 * 967 iterations), 34 with 2; the times the same, every objective the same) */
                static int teq = -1; if (teq < 0) { const char *e = getenv("BRISK_SXPRE_TIGHTEQ"); teq = e ? atoi(e) : 2; }
                if (eqb && teq == 0) continue;
                int ok = 1; double ar = 0;
                for (int p = 0; p < w.clen[j] && ok; p++) {
                    const int i = w.ci[j][p];
                    if (!w.rowon[i]) continue;
                    const double a = w.cv[j][p];
                    if (i == r) { ar = a; continue; }
                    /* the move of x_j lowers row i's activity when (a > 0) != up: then no lower side, else no upper side */
                    if ((a > 0) != up) { if (w.rl[i] > -PHUGE) ok = 0; } else { if (w.ru[i] < PHUGE) ok = 0; }
                }
                if (!ok || ar == 0) continue;
                /* the duals' signs: the argument needs x_j's other rows one-sided at the solution, so a row made an equation
                 * here never serves as another column's other row, and one that served is never made an equation (else the
                 * final check repairs the signs: physiciansched6-2, LP, 2,648 iterations) */
                if (trole[r] == 1) continue;
                for (int p = 0; p < w.clen[j] && ok; p++) { const int i = w.ci[j][p]; if (w.rowon[i] && i != r && trole[i] == 2) ok = 0; }
                if (!ok) continue;
                /* (the guard of the dropped bounds, by flags per row: a row that was the source of a dropped bound, a row with
                 * a column whose bound was dropped; tested only now, a scan of the rows for every column took 40 s on
                 * s250r10, LP, 269,588 columns) */
                if (eqb && teq == 2) {
                    if (dropsrc[j] >= 0 || rowanydrop[r]) continue;
                    for (int p = 0; p < w.clen[j] && ok; p++) { const int i = w.ci[j][p]; if (w.rowon[i] && rowdropsrc[i]) ok = 0; }
                    if (!ok) continue;
                }
                /* the side of r that bounds the move: the move lowers r's activity when (ar > 0) != up, so its lower side */
                if ((ar > 0) != up) { if (w.rl[r] <= -PHUGE) continue; w.ru[r] = w.rl[r]; }
                else { if (w.ru[r] >= PHUGE) continue; w.rl[r] = w.ru[r]; }
                if (eqb && teq == 2) {
                    if (up) w.u[j] = PINF; else w.l[j] = -PINF;
                    dropsrc[j] = r; rowdropsrc[r] = 1;
                    for (int p = 0; p < w.clen[j]; p++) rowanydrop[w.ci[j][p]] = 1;
                }
                rchg[r] = 1; trole[r] = 2; ntightrow++; changed = 1, nch++;
                for (int p = 0; p < w.clen[j]; p++) { const int i = w.ci[j][p]; if (w.rowon[i] && i != r) trole[i] = 1; }
            }
            free(sl); free(su); free(rchg);
        }
        if (tr_run) tr_idle = nch == nch_t;
        PT(4); if (status) break;
        /* doubleton equations: a x_j + b x_k = rhs, x_j substituted (the column with fewer entries, if its coefficient
         * is not small against the other's) */
        if (!nodbl) for (int i = 0; i < m && status == 0; i++) {
            if (!w.rowon[i] || w.rcnt[i] != 2 || w.rl[i] != w.ru[i] || w.rl[i] <= -PHUGE) continue;
            int jj[2] = {-1, -1}; double aa[2] = {0, 0}; int t = 0;
            for (int p = 0; p < w.rlen[i] && t < 2; p++) if (w.colon[w.ri[i][p]]) { jj[t] = w.ri[i][p]; aa[t] = w.rv[i][p]; t++; }
            if (t < 2) continue;
            int s = w.ccnt[jj[0]] <= w.ccnt[jj[1]] ? 0 : 1;
            { static int dblfree = -1; if (dblfree < 0) { const char *e_ = getenv("BRISK_SXPRE_DBLFREE"); dblfree = e_ ? atoi(e_) : 0; }
              if (dblfree) {
                  /* a free column is substituted rather than the bounded one: no bound moves to the column kept */
                  const int f0 = w.l[jj[0]] <= -PHUGE && w.u[jj[0]] >= PHUGE, f1 = w.l[jj[1]] <= -PHUGE && w.u[jj[1]] >= PHUGE;
                  if (f0 != f1) s = f0 ? 0 : 1;
              } }
            if (fabs(aa[s]) < 1e-3 * fabs(aa[1 - s])) s = 1 - s;
            if (fabs(aa[s]) < 1e-3 * fabs(aa[1 - s])) continue;
            const int j = jj[s], k = jj[1 - s];
            const double a = aa[s], b = aa[1 - s], rhs = w.rl[i];
            /* x_j's bounds as bounds on x_k: x_k = (rhs - a x_j) / b, decreasing in x_j when a / b > 0 */
            const double r = a / b;
            double klo, kup;
            if (r > 0) { klo = w.u[j] < PHUGE ? (rhs - a * w.u[j]) / b : -PINF; kup = w.l[j] > -PHUGE ? (rhs - a * w.l[j]) / b : PINF; }
            else { klo = w.l[j] > -PHUGE ? (rhs - a * w.l[j]) / b : -PINF; kup = w.u[j] < PHUGE ? (rhs - a * w.u[j]) / b : PINF; }
            int from = 0;
            if (klo > w.l[k] + 1e-12 * (1 + fabs(w.l[k]))) from |= 1;
            if (kup < w.u[k] - 1e-12 * (1 + fabs(w.u[k]))) from |= 2;
            rec_begin(P, R_DOUBLETON); rec_i(P, i); rec_i(P, j); rec_i(P, k); rec_i(P, from);
            rec_d(P, a); rec_d(P, b); rec_d(P, rhs);
            rec_d(P, from & 1 ? klo : w.l[k]); rec_d(P, from & 2 ? kup : w.u[k]);
            rec_col(P, &w, j, i);     /* column j now (for y_i when x_j is basic) */
            rec_col(P, &w, k, i);     /* column k before the substitution (for y_i when x_k is) */
            if (from & 1) w.l[k] = klo;
            if (from & 2) w.u[k] = kup;
            if (w.l[k] > w.u[k]) {
                if (w.l[k] > w.u[k] + FTOL * (1 + fabs(w.u[k]) + w.rmag[i] / fabs(b))) { status = 1; snprintf(why, nwhy, "row %d: the doubleton's bounds are empty", i); break; }   /* (rmag: the side's rounding, 4 November 2026) */
                const double mid = 0.5 * (w.l[k] + w.u[k]); w.l[k] = w.u[k] = mid;
            }
            /* costs and the other rows */
            cost_sub(&w, k, w.c[j] * b / a);
            w.off += w.c[j] * rhs / a;
            w.rowon[i] = 0; w.colon[j] = 0;
            w.ccnt[k]--;                      /* (row i's entry) */
            for (int p = 0; p < w.clen[j]; p++) {
                const int rr = w.ci[j][p];
                if (!w.rowon[rr]) continue;
                const double arj = w.cv[j][p];
                w.rcnt[rr]--;                 /* column j's entry leaves the row */
                if (w.rl[rr] > -PHUGE) w.rl[rr] -= arj * rhs / a;
                if (w.ru[rr] < PHUGE) w.ru[rr] -= arj * rhs / a;
                w.rmag[rr] += fabs(arj * rhs / a);
                double scale = fabs(arj * b / a); if (scale < 1) scale = 1;
                add_entry(&w, rr, k, -arj * b / a, scale);
            }
            ndbl++; changed = 1, nch++;
        }
        if (status) break;
        PT(5); if (status) break;
        /* the aggregator: a column whose bounds the rows imply (implied free) is substituted by one of its equation
         * rows in its other rows: the row of least fill (at most 16 new entries), its coefficient not small against
         * the row's largest. The implied bounds are those of the pass's start; a substitution drops the column's
         * bounds as implied, and an implied bound that came from a row with that column may have rested on them
         * (x + y = 1, 0 <= x, y <= 1: each is implied free by the other's bounds, not both): the columns whose needed
         * bound came from such a row wait for the next pass, whose implied bounds are computed again */
        const long nch_a = nch; const int agg_want = !noagg && nch != agg_clean, agg_run = agg_want && !(defer && ag_idle);
        if (agg_want && !agg_run) deferred = 1;
        if (agg_run) {
            int *srl = xmalloc(sizeof(int) * (n + 1)), *sru = xmalloc(sizeof(int) * (n + 1));
            memset(acache.val, 0, (size_t)m);     /* (the kept row sums, for the substitution through an inequality: rebuilt here, dropped for every row a substitution changes) */
            int *nmark = xmalloc(sizeof(int) * ((size_t)n + 1)); for (int k = 0; k <= n; k++) nmark[k] = 0; int stamp_ = 0;
            double *rmaxc = xmalloc(sizeof(double) * ((size_t)m + 1)); char *rmaxv = xcalloc((size_t)m + 1, 1); int *rbz = xmalloc(sizeof(int) * ((size_t)m + 1));
            char *rbzv = xcalloc((size_t)m + 1, 1);   /* (14 November 2026) rbz[i] exact (1) or not kept (0): computed where it is tested */
            int *rmaxk = xmalloc(sizeof(int) * ((size_t)m + 1));   /* (14 November 2026) the column of row i's largest entry, while rmaxv[i] = 1 */
            char *rold = xcalloc((size_t)m + 1, 1);   /* (14 November 2026) whether the row was read since a substitution last changed it (the net rule's budget charges a row's length then, as before the summaries were kept) */
            char *stale = xcalloc(n + 1, 1);
            /* (26 October 2026) sweeps until one substitutes nothing, the implied bounds computed again before each: the
             * stale columns waited for the next pass, and a chain of substitutions then took one full pass a link, every
             * other step sweeping the matrix again each time (pds-02, LP: 35 passes of 10-60 substitutions, three
             * quarters of its solve; Gurobi's whole solve took less than our presolve). The bounds of each sweep are
             * those of the problem as it then is, as at a new pass. BRISK_SXPRE_AGGONCE=1: one sweep a pass as before */
            static int aggonce = -1; if (aggonce < 0) aggonce = getenv("BRISK_SXPRE_AGGONCE") != NULL;
            /* (26 October 2026) a sweep after the first looks only at the columns of the rows a substitution changed since
             * the column was last looked at (cnow: those of the previous sweep, and of this one before the column's turn,
             * as a sweep of every column would see them), with the implied bounds of those columns computed again from
             * their rows: the others' rows, and so their bounds and the decision, are as at their last look (the
             * aggregator changes no column's bounds). pds-20 (LP): 24 sweeps, the implied bounds 0.13 s of the presolve's
             * 0.37 s. BRISK_SXPRE_AGGFULL=1: every column and the whole matrix each sweep, as before */
            static int aggfull = -1; if (aggfull < 0) aggfull = getenv("BRISK_SXPRE_AGGFULL") != NULL;
            char *cnow = xcalloc(n + 1, 1), *cnext = xcalloc(n + 1, 1);
            /* (2 November 2026) a substitution marked every column of each row it changed, scanning the row each time: rows of
             * 100,000 entries were scanned once per substitution in them (neos-4984115-canche, LP: 2.4 s). Now the row is
             * flagged (rdn: this sweep, rdx: the next), a column's check reads the flags of its own rows, the next sweep's
             * columns are taken from the flagged rows once at the sweep's end, and the columns whose implied bound came from
             * a row are listed by row (iptr, ilst), so the same columns wait as before: the same decisions in the same order */
            char *rdn = xcalloc((size_t)m + 1, 1), *rdx = xcalloc((size_t)m + 1, 1);
            int *iptr = xmalloc(sizeof(int) * ((size_t)m + 2)), *ilst = xmalloc(sizeof(int) * (2 * (size_t)n + 1));
            double *ra = xmalloc(sizeof(double) * (2 * (size_t)m + 2)); int *rn = xmalloc(sizeof(int) * (2 * (size_t)m + 2)), *rst = xmalloc(sizeof(int) * ((size_t)m + 1));
            for (int i = 0; i < m; i++) rst[i] = -1;
            for (int sweep = 0; sweep < (aggonce ? 1 : 50) && status == 0; sweep++) {
            const long nch_s = nch;
            const double tib0 = ptimes ? omp_get_wtime() : 0;
            const int part = sweep > 0 && !aggfull;
            if (!part) implied_col_bounds_c(&w, &ibc, nch, ilo, iup, srl, sru);
            else { char *t = cnow; cnow = cnext; cnext = t; memset(cnext, 0, (size_t)n + 1); implied_col_bounds_mk(&w, cnow, ilo, iup, srl, sru, ra, rn, rst, sweep); }
            if (ptimes) tt_ib += omp_get_wtime() - tib0;
            if (sweep) memset(stale, 0, (size_t)n + 1);
            memset(rdn, 0, (size_t)m + 1);
            {   /* the columns by the row their implied lower or upper bound came from */
                memset(iptr, 0, sizeof(int) * ((size_t)m + 2));
                for (int k = 0; k < n; k++) { if (srl[k] >= 0) iptr[srl[k] + 2]++; if (sru[k] >= 0 && sru[k] != srl[k]) iptr[sru[k] + 2]++; }
                for (int i = 0; i < m; i++) iptr[i + 2] += iptr[i + 1];
                for (int k = 0; k < n; k++) { if (srl[k] >= 0) ilst[iptr[srl[k] + 1]++] = k; if (sru[k] >= 0 && sru[k] != srl[k]) ilst[iptr[sru[k] + 1]++] = k; }
            }
            for (int j = 0; j < n && status == 0; j++) {
                if (!w.colon[j] || w.ccnt[j] < 2 || stale[j]) continue;
                if (part && !cnow[j]) { int d = 0; for (int q = 0; q < w.clen[j] && !d; q++) d = rdn[w.ci[j][q]]; if (!d) continue; }
                const int lok = w.l[j] <= -PHUGE || ilo[j] >= w.l[j] - FTOL * (1 + fabs(w.l[j]));
                const int uok = w.u[j] >= PHUGE || iup[j] <= w.u[j] + FTOL * (1 + fabs(w.u[j]));
                /* (an experiment, BRISK_SXAGGB=1, 23 October 2026) a column with bounds the rows do not imply is
                 * substituted too, its bounds kept by the row, which stays as a range (R_AGGB) */
                const int bnd = !(lok && uok);
                int eqok = !(bnd && (!aggb || (aggb_late && !aggb_phase) || (aggbzc && w.c[j] != 0.0)));
                if (eqok && bnd && aggbhalf && !lok && !uok) eqok = 0;     /* (experiment: one of the bounds implied) */
                if (eqok && bnd && aggbmaxrow > 0)     /* (the bounded substitution not into a long row: see aggbmaxrow) */
                    for (int p = 0; p < w.clen[j] && eqok; p++) { const int r = w.ci[j][p]; if (w.rowon[r] && w.rcnt[r] > aggbmaxrow) eqok = 0; }
                /* (12 November 2026) oknet: a column with one infinite bound substituted through a one-sided inequality row
                 * that implies its finite bound, in the simplex methods' presolve, when the substitution does not add
                 * entries on balance (below; BRISK_SXPRE_AGGINET=0: off) */
                /* (the entries the net rule's counts read capped at ten times the matrix's entries over the presolve: on rows
                 * of 100,000 entries every column of the row can pass the cached checks and each count reads the row, the
                 * square of its length; neos-4984115-canche, LP: its aggregator 0.64 -> 1.55 s for no substitution) */
                const int oneinf_j = w.l[j] <= -PHUGE || w.u[j] >= PHUGE;
                const int okold = aggi && (aggifree ? (w.l[j] <= -PHUGE && w.u[j] >= PHUGE) : oneinf_j);
                /* (at most 8 entries in the column: the check below costs the column's length for each of its rows, and a
                 * substitution that adds nothing needs each of the column's other rows to hold nearly all of the row's
                 * columns, which a long column rarely meets; neos-5149806-wieprz, LP, spent 53-83 s checking columns in
                 * thousands of rows for no substitution. ger50-17-trans-dfn-3t's columns have 2) */
                /* (13 November 2026, by default; BRISK_SXPRE_AGGINETC=0: every column) only a column with a cost: its substitution
                 * moves the cost onto the row's other columns, which can then be fixed as dominated (ger50-17-trans-dfn-3t, LP:
                 * about 9,000 columns); a column of cost zero moves no cost and only reshapes rows (gmut-76-40, LP: two such
                 * substitutions, 7-11 % more iterations on every permuted copy) */
                static int aggic = -1; if (aggic < 0) { const char *e_ = getenv("BRISK_SXPRE_AGGINETC"); aggic = !(e_ && atoi(e_) == 0); }
                const int oknet = agginet && agginet_phase && oneinf_j && !(w.l[j] <= -PHUGE && w.u[j] >= PHUGE) && w.ccnt[j] <= 8 && !(aggic && w.c[j] == 0);
                if (!eqok && !okold && !oknet) continue;
                int ib = -1; double ab = 0; long fb = 0; int ineq = 0;
                if (eqok) for (int p = 0; p < w.clen[j]; p++) {
                    const int i = w.ci[j][p];
                    if (!w.rowon[i] || w.rl[i] != w.ru[i] || w.rl[i] <= -PHUGE) continue;
                    long fill = (long)(w.ccnt[j] - 1) * (w.rcnt[i] - 2);
                    /* (BRISK_SXAGGNET=1, an experiment of 26 October 2026) a substitution that does not add entries
                     * on balance (fill minus the row's own entries, the column's other entries already in fill's
                     * count) is taken whatever the row's length: a column of two entries always removes two
                     * (neos-1445765, LP: sum x = k w, w in a knapsack row; Gurobi 1,001 rows left, ours 1,478) */
                    int oldok = !(w.rcnt[i] > 24 || fill > (bnd ? aggbfill : 16));
                    const int netok = aggnet && !bnd && fill - w.rcnt[i] <= 0 && (aggnet2 == 0 || w.ccnt[j] == 2);
                    /* (12 November 2026) the fill counted exactly: (entries of the column - 1)(entries of the row - 2) assumes that
                     * the column's other rows share none of the row's columns. Where they share them, the substitution adds fewer
                     * entries, none at all when the other row holds the same columns (neos-1367061, LP: 336 columns y of two
                     * entries, sum x - 8000 y = 0 with 25 flows x and -sum x + 4000 y <= 0 over the same flows; the substitution
                     * adds nothing and the second row becomes sum x >= 0, redundant; Gurobi's presolve does it, and the problem's
                     * symmetry, hidden by these rows, then halves it). A substitution that the estimate refuses and that adds no
                     * entry on balance by the exact count is taken whatever the row's length; the count for columns of at most 8
                     * entries, within the presolve's budget for such counts (network). Not those that add entries up to the
                     * estimate's limit of 16: on pilot (Netlib, LP) they took short rows with pivots 0.03-0.2 of their row's
                     * largest, and the interior-point method then stalled near the optimum (0.25 -> 0.48 s).
                     * BRISK_SXAGGEXACT=0: the estimate alone, as before; BRISK_SXAGGEXACTLIM=k: up to k new entries (0) */
                    if (!oldok && !netok && (!bnd || aggbex) && aggexact && w.ccnt[j] <= 8 && network <= netbudget) {
                        long lb = 0;
                        for (int pp = 0; pp < w.clen[j] && lb <= aggexlim; pp++) {
                            const int r = w.ci[j][pp]; if (!w.rowon[r] || r == i) continue;
                            const long sh = (w.rcnt[r] < w.rcnt[i] ? w.rcnt[r] : w.rcnt[i]) - 1;
                            lb += (w.rcnt[i] - 2) - (sh > 0 ? sh : 0);
                        }
                        if (lb <= aggexlim) {
                            network += w.rlen[i];
                            for (int pp = 0; pp < w.clen[j]; pp++) { const int r = w.ci[j][pp]; if (w.rowon[r] && r != i) network += w.rlen[r] > 4 * w.rlen[i] && w.rlen[r] >= RIDX_MIN ? w.rlen[i] : w.rlen[r]; }
                            stamp_++;
                            for (int q = 0; q < w.rlen[i]; q++) { const int k = w.ri[i][q]; if (k != j && w.colon[k]) nmark[k] = stamp_; }
                            long add = 0;
                            for (int pp = 0; pp < w.clen[j] && add <= aggexlim; pp++) {
                                const int r = w.ci[j][pp]; if (!w.rowon[r] || r == i) continue;
                                long have = 0;
                                if (w.rlen[r] > 4 * w.rlen[i] && w.rlen[r] >= RIDX_MIN) {
                                    if (!w.ridx[r]) ridx_build(&w, r);
                                    for (int q = 0; q < w.rlen[i]; q++) { const int k = w.ri[i][q]; if (k != j && w.colon[k] && wrow_find(&w, r, k) >= 0) have++; }
                                } else for (int q = 0; q < w.rlen[r]; q++) { const int k = w.ri[r][q]; if (k != j && w.colon[k] && nmark[k] == stamp_) have++; }
                                add += (w.rcnt[i] - 2) - have;
                            }
                            if (add <= aggexlim) { oldok = 1; fill = add; n_aggexact++;
                                if (getenv("BRISK_SXAGGEXACTDBG")) { double am_ = 0, cm_ = 0; for (int q = 0; q < w.rlen[i]; q++) if (w.colon[w.ri[i][q]]) am_ = fmax(am_, fabs(w.rv[i][q])); for (int q = 0; q < w.clen[j]; q++) if (w.rowon[w.ci[j][q]]) cm_ = fmax(cm_, fabs(w.cv[j][q]));
                                    fprintf(stderr, "[aggexact] column %d (%d entries) row %d (%d entries): new %ld, pivot/row max %.3g, pivot/column max %.3g\n", j, w.ccnt[j], i, w.rcnt[i], add, fabs(w.cv[j][p]) / am_, fabs(w.cv[j][p]) / cm_); } }
                        }
                    }
                    if (!oldok && !netok) continue;
                    if (ib >= 0 && fill >= fb) continue;
                    double amax = 0;
                    for (int q = 0; q < w.rlen[i]; q++) if (w.colon[w.ri[i][q]] && fabs(w.rv[i][q]) > amax) amax = fabs(w.rv[i][q]);
                    /* (3 November 2026) the pivot's size: the substitution x_j = (b_i - sum a_ik x_k) / a_ij puts a_rk - (a_rj / a_ij)
                     * a_ik into each other row r of column j, whose growth is bounded either by the row (|a_ik / a_ij| <= 100) or
                     * by the column (|a_rj / a_ij| <= 100), as a threshold pivot of an LU factorization is: either test passing is
                     * enough. Only the row's was used: neos-4984115-canche (LP) kept 9,521 of its rows (Gurobi 2,384), its pivots
                     * small in their rows and not in their columns; with either test 2,384 rows and 2,087 dual simplex iterations
                     * instead of 12,221, the run 3.0 -> 2.2 s. The 618 development LPs otherwise unchanged (1,478.5 -> 1,478.0 s,
                     * no objective changed). BRISK_SXAGGPCOL=0: the row's test alone, as before */
                    { static int pcol = -1; if (pcol < 0) { const char *e_ = getenv("BRISK_SXAGGPCOL"); pcol = !(e_ && atoi(e_) == 0); }
                      if (pcol && oldok) {
                          double cmax = 0; for (int q = 0; q < w.clen[j]; q++) if (w.rowon[w.ci[j][q]] && fabs(w.cv[j][q]) > cmax) cmax = fabs(w.cv[j][q]);
                          if (fabs(w.cv[j][p]) < 0.01 * cmax && fabs(w.cv[j][p]) < 0.01 * amax) continue;
                      } else if (fabs(w.cv[j][p]) < (oldok ? 0.01 : aggnettol) * amax) continue; }
                    ib = i; ab = w.cv[j][p]; fb = fill;
                }
                /* (27 October 2026) with no equation to use, a one-sided inequality row a x_j + s <= rhs (or >= rhs) that
                 * bounds x_j on one side, x_j's bound on the other side being infinite, and that alone implies x_j's
                 * finite bound: x_j = (rhs - s) / a + x_j', the row then the bound x_j' >= 0 (or <= 0), an exact change
                 * of variables. The row is dropped; the column stays as x_j' (its entries in the other rows and its cost
                 * as before, bounds [0, inf) or (-inf, 0]); the row's other columns take x_j's entries and cost as in the
                 * substitution by an equation. On cre-a (LP), -20 X + D1 + D2 <= 0 with X >= 0: X = X' + (D1 + D2) / 20,
                 * X' >= 0; Gurobi's presolve does this (1,260 rows left of 3,516; ours 2,674). Off by default (TRICKS 8): cre-a
                 * 2,482 -> 1,357 dual iterations and 25 % faster, cre-c 25-39 %, but on cre-d the pivot rows 35 % longer
                 * for 12 % fewer rows, and the fill it puts into the normal equations stops the interior-point method
                 * (irish-electricity, LP: 8.3 s without, unfinished in 40 s with it) */
                if (ib < 0 && (okold || oknet)) for (int p = 0; p < w.clen[j]; p++) {
                    const int i = w.ci[j][p];
                    if (!w.rowon[i]) continue;
                    const int le = w.ru[i] < PHUGE;
                    if (le == (w.rl[i] > -PHUGE)) continue;             /* (one-sided rows only) */
                    const double a = w.cv[j][p];
                    const int below = le == (a < 0);                     /* the row bounds x_j from below: x_j' >= 0 */
                    if (below ? w.u[j] < PHUGE : w.l[j] > -PHUGE) continue;
                    /* (12 November 2026) the row's largest entry and its activity sums kept per row (rmaxc; the presolve's
                     * cache of long rows' sums, act_get) and dropped for each row a substitution changes: with long rows
                     * every column of a row that bounds it on the right side is a candidate, and each asked the whole row
                     * twice (ger50-17-trans-dfn-3t, LP, with the row limits lifted: 4.35 s of the aggregator, 76 % of the
                     * instructions in the activity's scans) */
                    /* (a candidate of the net rule alone pays from its budget for the rows it reads: the cached sums of a row
                     * are computed again after a substitution changed it) */
                    const int netonly = oknet && !okold;
                    if (netonly) {
                        if (network > netbudget) continue;
                        if (!rold[i]) network += w.rlen[i];
                        if (w.rlen[i] < 64 || !acache.val[i]) network += w.rlen[i];
                    }
                    /* (14 November 2026) rmaxv[i] = 1: the row's largest entry (rmaxc, in column rmaxk) is exact; the substitutions keep it so
                     * (below, where they change the rows) and drop it only when the column holding it leaves the row or that entry shrinks.
                     * Before, every row a substitution touched lost it, and the next candidate of the row scanned it whole: with the
                     * candidates of a long row one after another the scans grew with the square of its length (neos-4984115-canche, LP, its
                     * rows in reverse Cuthill-McKee order: 4.5e9 entries scanned, 5.7 s; as written 5.3e8, 2.6 s; now 2.6e6 and 2.8e6 entries,
                     * 1.8 s both; the same presolved problem). The count of boxed columns without cost (rbz) is computed where it is tested */
                    { static int rd_ = -1; if (rd_ < 0) { const char *e_ = getenv("BRISK_SXPRE_RMAXDBG"); rd_ = e_ && atoi(e_) != 0; }   /* (a diagnostic: the rows' scans) */
                      if (rd_) { static long c0_, c3_, cl_; if (!rmaxv[i]) c0_++, cl_ += w.rlen[i]; else c3_++;
                        if ((c0_ + c3_) % 200000 == 0) fprintf(stderr, "  [the rows' largest entries: scanned %ld times (%ld entries), kept %ld times]\n", c0_, cl_, c3_); } }
                    if (!rmaxv[i]) {   /* (with the count of the row's columns boxed without cost: x_j itself is never one, it has an infinite bound) */
                        double am = 0; int bzc_ = 0, ak_ = -1;
                        for (int q = 0; q < w.rlen[i]; q++) { const int k = w.ri[i][q]; if (!w.colon[k]) continue; if (fabs(w.rv[i][q]) > am) { am = fabs(w.rv[i][q]); ak_ = k; }
                            if (w.l[k] > -PHUGE && w.u[k] < PHUGE && w.l[k] < w.u[k] && w.c[k] == 0) bzc_++; }
                        rmaxc[i] = am; rmaxk[i] = ak_; rbz[i] = bzc_; rmaxv[i] = 1; rbzv[i] = 1; }
                    rold[i] = 1;
                    const double amax = rmaxc[i];
                    if (fabs(a) < 0.01 * amax) continue;
                    /* x_j's finite bound implied by this row alone: x_j >= (rhs - s) / a for all s (below), <= (above) */
                    const double bj = below ? w.l[j] : w.u[j];
                    if (fabs(bj) < PHUGE) {
                        double lmin, lmax; int n0, n1; act_get(&w, &acache, i, j, a, &lmin, &lmax, &n0, &n1);
                        const double rhs = le ? w.ru[i] : w.rl[i];
                        const int sm = below == (a > 0);                 /* the extreme from s's maximum (else its minimum) */
                        if (sm ? n1 : n0) continue;
                        const double ext = (rhs - (sm ? lmax : lmin)) / a;
                        if (below ? ext < bj - FTOL * (1 + fabs(bj)) : ext > bj + FTOL * (1 + fabs(bj))) continue;
                    }
                    const int bzmax = aggibz >= 0 ? aggibz : oknet && !okold ? 0 : -1;
                    if (bzmax >= 0 && !rbzv[i]) {   /* (the count where it is tested: a row's scan only for the candidates that reach here) */
                        int bzc_ = 0;
                        for (int q = 0; q < w.rlen[i]; q++) { const int k = w.ri[i][q]; if (w.colon[k] && w.l[k] > -PHUGE && w.u[k] < PHUGE && w.l[k] < w.u[k] && w.c[k] == 0) bzc_++; }
                        rbz[i] = bzc_; rbzv[i] = 1;
                    }
                    if (bzmax >= 0 && rbz[i] > bzmax) continue;   /* (mode 9, and the net rule) at most bzmax of the row's other columns boxed without cost */
                    long fill = (long)(w.ccnt[j] - 1) * (w.rcnt[i] - 1);
                    const int small_ = okold && w.rcnt[i] <= 24 && fill <= 16;
                    int bynet = 0;
                    if (!small_) {
                        if (!oknet) continue;
                        /* the entries the substitution adds (row i's other columns not yet in each other row of column j)
                         * against those it removes (row i's): at most as many. Counted exactly with marks */
                        /* first a lower bound (row r shares at most min(|row i|, |row r|) - 1 of row i's other columns), then the
                         * count, each row compared from its shorter side (row i's columns looked up in a long row's index):
                         * a column in many long rows was counted by scanning each of them, in every sweep (neos-5149806-wieprz,
                         * LP: 52.6 s of the aggregator for no substitution) */
                        long lb = 0;
                        for (int pp = 0; pp < w.clen[j] && lb <= w.rcnt[i]; pp++) {
                            const int r = w.ci[j][pp]; if (!w.rowon[r] || r == i) continue;
                            const long sh = (w.rcnt[r] < w.rcnt[i] ? w.rcnt[r] : w.rcnt[i]) - 1;
                            lb += (w.rcnt[i] - 1) - (sh > 0 ? sh : 0);
                        }
                        if (lb > w.rcnt[i]) continue;
                        if (network > netbudget) continue;
                        network += w.rlen[i];
                        for (int pp = 0; pp < w.clen[j]; pp++) { const int r = w.ci[j][pp]; if (w.rowon[r] && r != i) network += w.rlen[r] > 4 * w.rlen[i] && w.rlen[r] >= RIDX_MIN ? w.rlen[i] : w.rlen[r]; }
                        stamp_++;
                        for (int q = 0; q < w.rlen[i]; q++) { const int k = w.ri[i][q]; if (k != j && w.colon[k]) nmark[k] = stamp_; }
                        long add = 0;
                        for (int pp = 0; pp < w.clen[j] && add <= w.rcnt[i]; pp++) {
                            const int r = w.ci[j][pp]; if (!w.rowon[r] || r == i) continue;
                            long have = 0;
                            if (w.rlen[r] > 4 * w.rlen[i] && w.rlen[r] >= RIDX_MIN) {
                                if (!w.ridx[r]) ridx_build(&w, r);
                                for (int q = 0; q < w.rlen[i]; q++) { const int k = w.ri[i][q]; if (k != j && w.colon[k] && wrow_find(&w, r, k) >= 0) have++; }
                            } else for (int q = 0; q < w.rlen[r]; q++) { const int k = w.ri[r][q]; if (k != j && w.colon[k] && nmark[k] == stamp_) have++; }
                            add += (w.rcnt[i] - 1) - have;
                        }
                        if (add > w.rcnt[i]) continue;
                        fill = add; bynet = 1;
                    }
                    if (ib >= 0 && fill >= fb) continue;
                    (void)bynet;
                    ib = i; ab = a; fb = fill; ineq = (below ? 1 : 2) | (le ? 4 : 0);
                }
                if (ib < 0) continue;
                const int i = ib; const double a = ab, cj = w.c[j], rhs = !ineq ? w.rl[i] : (ineq & 4) ? w.ru[i] : w.rl[i];
                const int bsub = bnd && !ineq;
                if (bsub || ineq) { w.cmod[j] = 1; w.cver[j]++; }      /* (the record lists the column without row i, which stays: in full) */
                if (ineq) {     /* (the row's other columns that are boxed with no cost: a diagnostic, and BRISK_SXPRE_AGGIBZ) */
                    int bz = 0;
                    for (int q = 0; q < w.rlen[i]; q++) { const int k = w.ri[i][q]; if (k != j && w.colon[k] && w.l[k] > -PHUGE && w.u[k] < PHUGE && w.l[k] < w.u[k] && w.c[k] == 0) bz++; }
                    naggi_bz[bz >= 2 ? 2 : bz]++;
                }
                if (ineq && getenv("SXDBG") && naggi < 60) {
                    printf("    AGGI: col %d (bounds %g %g, %d entries, cost %g) row %d (a %g, sides %g %g, %d entries) fill %ld; other cols:", j, w.l[j], w.u[j], w.ccnt[j], w.c[j], i, ab, w.rl[i], w.ru[i], w.rcnt[i], fb);
                    for (int q = 0; q < w.rlen[i]; q++) { const int k = w.ri[i][q]; if (k != j && w.colon[k]) printf(" %g*c%d[%g,%g;%d;c%g]", w.rv[i][q], k, w.l[k], w.u[k], w.ccnt[k], w.c[k]); }
                    printf("; col's rows:"); for (int p = 0; p < w.clen[j]; p++) { const int r = w.ci[j][p]; if (w.rowon[r]) printf(" r%d(%g;%g,%g;%d)", r, w.cv[j][p], w.rl[r], w.ru[r], w.rcnt[r]); }
                    printf("\n"); }
                if (bsub && getenv("SXDBG") && naggb < 40) printf("    bounded substitution: column %d (bounds %g, %g) by row %d (a %g, rhs %g, %d entries)\n", j, w.l[j], w.u[j], i, ab, w.rl[i], w.rcnt[i]);
                rec_begin(P, ineq ? R_AGGI : bsub ? R_AGGB : R_AGG); rec_i(P, i); rec_i(P, j); rec_d(P, a); rec_d(P, rhs);
                int cnt = 0;
                for (int q = 0; q < w.rlen[i]; q++) if (w.ri[i][q] != j && w.colon[w.ri[i][q]]) cnt++;
                rec_i(P, cnt);
                for (int q = 0; q < w.rlen[i]; q++) { const int k = w.ri[i][q]; if (k != j && w.colon[k]) { rec_i(P, k); rec_d(P, w.rv[i][q]); } }
                if (ineq) rec_i(P, ineq);
                rec_col(P, &w, j, i);
                /* the row's other columns and entries (a copy: the substitution changes the rows) */
                int *kk = xmalloc(sizeof(int) * (cnt + 1)); double *ak = xmalloc(sizeof(double) * (cnt + 1)); int t = 0;
                for (int q = 0; q < w.rlen[i]; q++) { const int k = w.ri[i][q]; if (k != j && w.colon[k]) { kk[t] = k; ak[t] = w.rv[i][q]; t++; } }
                /* (14 November 2026) the rows' largest entries kept exact through the substitution: a row's is lost only when the
                 * column holding it leaves the row (x_j, which leaves its rows unless it is kept as the row's slack) or that entry
                 * shrinks; an entry that grows above it takes its place (below, with the entries the substitution changes) */
                acache.val[i] = 0; rbzv[i] = 0; rold[i] = 0; if (rmaxv[i] == 1 && !ineq && rmaxk[i] == j) rmaxv[i] = 0;
                for (int p = 0; p < w.clen[j]; p++) { const int r_ = w.ci[j][p]; acache.val[r_] = 0; rbzv[r_] = 0; rold[r_] = 0; if (rmaxv[r_] == 1 && !ineq && rmaxk[r_] == j) rmaxv[r_] = 0; }
                for (int q = 0; q < t; q++) cost_sub(&w, kk[q], cj * ak[q] / a);
                for (int p = 0; p < w.clen[j]; p++) {
                    const int r = w.ci[j][p];
                    if (!w.rowon[r]) continue;
                    rdn[r] = rdx[r] = 1;
                    for (int q = iptr[r]; q < iptr[r + 1]; q++) {
                        const int k = ilst[q];
                        if ((srl[k] == r && w.l[k] > -PHUGE) || (sru[k] == r && w.u[k] < PHUGE)) stale[k] = 1;
                    }
                }
                w.off += cj * rhs / a;
                if (!ineq) w.colon[j] = 0;
                if (!bsub) drop_row(&w, i);
                else {
                    /* row i without x_j: x_j = (rhs - s) / a in [l_j, u_j] is s in [rhs - a u_j, rhs - a l_j] (a > 0) */
                    const double lj = w.l[j], uj = w.u[j];
                    const double s1 = uj >= PHUGE ? (a > 0 ? -PINF : PINF) : rhs - a * uj, s2 = lj <= -PHUGE ? (a > 0 ? PINF : -PINF) : rhs - a * lj;
                    w.rl[i] = fmin(s1, s2); w.ru[i] = fmax(s1, s2);
                    w.rmag[i] += fabs(rhs) + (uj < PHUGE ? fabs(a * uj) : 0.0) + (lj > -PHUGE ? fabs(a * lj) : 0.0);
                    if (w.rl[i] <= -PHUGE) w.rl[i] = -PINF;
                    if (w.ru[i] >= PHUGE) w.ru[i] = PINF;
                    w.rcnt[i]--;
                    naggb++;
                    if (getenv("BRISK_SXAGGBSTATS")) {   /* (a diagnostic, one problem a process: its counts are static) the substituted column's entries and its row's length, and the rows its entries go to */
                        static long hc[6], hr[6], nsub, sumlen, maxlen, hl[6]; const int cc = w.ccnt[j], rc = w.rcnt[i] + 1;
                        { int ml = 0; for (int p = 0; p < w.clen[j]; p++) { const int r = w.ci[j][p]; if (w.rowon[r] && r != i && w.rcnt[r] > ml) ml = w.rcnt[r]; }
                          hl[ml <= 30 ? 0 : ml <= 100 ? 1 : ml <= 300 ? 2 : ml <= 1000 ? 3 : ml <= 10000 ? 4 : 5]++;
                          if (nsub % 5000 == 0) fprintf(stderr, "  [the longest other row of the substituted columns: <=30 %ld, <=100 %ld, <=300 %ld, <=1000 %ld, <=10000 %ld, more %ld]\n", hl[0], hl[1], hl[2], hl[3], hl[4], hl[5]); }
                        hc[cc < 5 ? cc : 5]++; hr[rc < 5 ? rc : 5]++; nsub++;
                        for (int p = 0; p < w.clen[j]; p++) { const int r = w.ci[j][p]; if (!w.rowon[r] || r == i) continue; sumlen += w.rcnt[r]; if (w.rcnt[r] > maxlen) maxlen = w.rcnt[r]; }
                        if (nsub % 5000 == 0 || nsub < 2) fprintf(stderr, "  [bounded substitutions %ld: column entries 2:%ld 3:%ld 4:%ld 5+:%ld; row entries 2:%ld 3:%ld 4:%ld 5+:%ld; the other rows of the columns: mean %.1f entries, longest %ld]\n",
                                nsub, hc[2], hc[3], hc[4], hc[5], hr[2], hr[3], hr[4], hr[5], (double)sumlen / nsub, maxlen);
                    }
                }
                for (int p = 0; p < w.clen[j]; p++) {
                    const int r = w.ci[j][p];
                    if (!w.rowon[r] || r == i) continue;
                    const double f = w.cv[j][p] / a;
                    if (!ineq) w.rcnt[r]--;             /* (x_j' keeps its entry) */
                    if (w.rl[r] > -PHUGE) w.rl[r] -= f * rhs;
                    if (w.ru[r] < PHUGE) w.ru[r] -= f * rhs;
                    w.rmag[r] += fabs(f * rhs);
                    for (int q = 0; q < t; q++) { double sc = fabs(f * ak[q]); if (sc < 1) sc = 1; const double nv_ = add_entry(&w, r, kk[q], -f * ak[q], sc);
                        if (rmaxv[r] == 1) { if (nv_ > rmaxc[r]) { rmaxc[r] = nv_; rmaxk[r] = kk[q]; } else if (kk[q] == rmaxk[r] && nv_ < rmaxc[r]) rmaxv[r] = 0; } }
                }
                free(kk); free(ak);
                if (ineq) { w.l[j] = (ineq & 1) ? 0 : -PINF; w.u[j] = (ineq & 1) ? PINF : 0; stale[j] = 1; naggi++; }
                else nagg++;
                changed = 1, nch++;
            }
            if (ptimes) fprintf(stderr, "  [aggregator pass %d sweep %d: %ld substitutions; implied bounds so far %.4f s]\n", pass, sweep, nch - nch_s, tt_ib);
            /* the flagged rows' columns for the next sweep (whether the row is still on or not, as the marks were) */
            for (int i = 0; i < m; i++) if (rdx[i]) { rdx[i] = 0; for (int q = 0; q < w.rlen[i]; q++) cnext[w.ri[i][q]] = 1; }
            if (nch == nch_s) break;
            }
            free(stale); free(srl); free(sru); free(cnow); free(cnext); free(ra); free(rn); free(rst); free(rdn); free(rdx); free(iptr); free(ilst); free(rmaxc); free(rmaxv); free(rbzv); free(rmaxk); free(rold); free(nmark); free(rbz);
        }
        if (status) break;
        if (agg_run && nch == nch_a) agg_clean = nch;
        if (agg_run) ag_idle = nch == nch_a;
        PT(6); if (status) break;
        /* rows: activity bounds */
        const long nch_r = nch; const int act_run = nch != act_clean;
        for (int i = 0; i < m && status == 0 && act_run; i++) {
            if (!w.rowon[i] || w.rcnt[i] < 2) continue;
            double lmin, lmax; int ninfmin, ninfmax;
            activity(&w, i, -1, &lmin, &lmax, &ninfmin, &ninfmax);
            const double rl = w.rl[i], ru = w.ru[i];
            const double tl = FTOL * (1 + fabs(rl) + w.rmag[i]), tu = FTOL * (1 + fabs(ru) + w.rmag[i]);   /* (rmag: the sides' rounding, 4 November 2026) */
            if ((ninfmax == 0 && rl > -PHUGE && lmax < rl - tl) || (ninfmin == 0 && ru < PHUGE && lmin > ru + tu)) {
                status = 1; snprintf(why, nwhy, "row %d cannot reach its bounds", i); break;
            }
            const int lowok = rl <= -PHUGE || (ninfmin == 0 && lmin >= rl - tl);
            const int upok = ru >= PHUGE || (ninfmax == 0 && lmax <= ru + tu);
            if (lowok && upok) {
                rec_begin(P, R_REDROW); rec_i(P, i);
                drop_row(&w, i); nred++; changed = 1, nch++;
                continue;
            }
            /* (24 October 2026, the night) one side of a ranged row that the columns' bounds cannot reach is dropped: the
             * feasible set is the same, and the row becomes one-sided, so that dual fixing and the dual bounds can use its
             * sign (supportcase12, LP: 15,271 ranged rows x - y1 - y2 - y3 in [0, 7.2] with x <= 1.33, the y of cost zero;
             * Gurobi's presolved problem has none). No record: a solution, its duals and its statuses are the same for both
             * rows (the dropped side is never active). BRISK_SXPRE_NOSIDE=1: kept */
            if (!noside && rl > -PHUGE && ru < PHUGE && (lowok || upok)) {
                if (lowok) w.rl[i] = -PINF; else w.ru[i] = PINF;
                nside++; changed = 1, nch++;
            }
            int side = 0;    /* +1: forced at rl (the columns at the bounds of the largest activity), -1: at ru */
            if (ninfmax == 0 && rl > -PHUGE && lmax <= rl + tl) side = 1;
            else if (ninfmin == 0 && ru < PHUGE && lmin >= ru - tu) side = -1;
            if (side && !forcing_tight(&w, i, side, side > 0 ? lmax - rl : ru - lmin)) side = 0;
            if (!side) continue;
            rec_begin(P, R_FORCING); rec_i(P, i); rec_i(P, side);
            int cnt = 0;
            for (int p = 0; p < w.rlen[i]; p++) if (w.colon[w.ri[i][p]]) cnt++;
            rec_i(P, cnt);
            for (int p = 0; p < w.rlen[i]; p++) {
                const int j = w.ri[i][p];
                if (!w.colon[j]) continue;
                const double a = w.rv[i][p];
                const double v = (side > 0) == (a > 0) ? w.u[j] : w.l[j];
                rec_i(P, j); rec_i(P, fixstat(v, w.l[j], w.u[j], P->l[j], P->u[j])); rec_d(P, a); rec_d(P, v);
                rec_col(P, &w, j, i);
            }
            /* the columns fixed, then the row (whose bounds now hold for the fixed values) */
            w.rowon[i] = 0;
            for (int p = 0; p < w.rlen[i]; p++) {
                const int j = w.ri[i][p];
                if (!w.colon[j]) continue;
                const double a = w.rv[i][p];
                drop_col(&w, j, (side > 0) == (a > 0) ? w.u[j] : w.l[j]);
            }
            nforce++; changed = 1, nch++;
        }
        if (status) break;
        PT(7); if (status) break;
        /* (24 October 2026, the night) rows redundant by the bounds the other rows imply (not only the columns' own): a row
         * whose activity, with each column's bounds tightened by the rows that imply tighter ones, stays within its sides is
         * dropped; the bounds themselves are not changed. In one pass a row that gave such a bound is not dropped, and a
         * dropped row gives none, so no two rows justify each other's removal (neos-1171448, LP: x0 + x567 <= 8 with
         * x0 <= 3 and x567 <= 3 from rows of 22 entries; Gurobi drops them). BRISK_SXPRE_NOIMPLRED=1: off */
        /* (2 November 2026) entries too small to matter: in a row whose entries span more than twelve orders of magnitude, an
         * entry whose contribution over its column's range (its bounds, or bounds implied by other rows) is at most 1e-12 of
         * the row's largest entry, all such entries of the row together, is dropped (its contribution at the column's lower
         * bound, or upper, moved to the sides). bley_xs2 (LP, MIPLIB 2017): 349 rows 1e20 (x307 + x308 + x2103 + x2104) -
         * x323 >= 0, x323 <= 1e4 by other rows; no tolerance tells the two scales apart, and every method lost its accuracy
         * (5 s, numerical difficulties; Gurobi 0.02 s). The postsolve restores each such row exactly on the problem as read:
         * when the point violates it, the row's column of the largest entry moves by the deficit over that entry (about 1e-16
         * in bley_xs2), within its bounds. Once, in the first pass. BRISK_SXPRE_NONEGL=1: off */
        if (pass == 0 && status == 0 && !nonegl) {
            int *sl = NULL, *su = NULL, *dj = NULL; double *da = NULL; int dcap = 0;
            for (int i = 0; i < m; i++) {
                if (!w.rowon[i] || w.rcnt[i] < 2) continue;
                double amax = 0, amin = HUGE_VAL;
                for (int p = 0; p < w.rlen[i]; p++) { const int j = w.ri[i][p]; if (!w.colon[j]) continue; const double a = fabs(w.rv[i][p]); if (a > amax) amax = a; if (a < amin) amin = a; }
                if (!(amin * 1e12 < amax)) continue;
                if (!sl) { sl = xmalloc(sizeof(int) * (n + 1)); su = xmalloc(sizeof(int) * (n + 1)); implied_col_bounds_c(&w, &ibc, nch, ilo, iup, sl, su); }
                if (dcap < w.rlen[i]) { dcap = w.rlen[i]; dj = xrealloc(dj, sizeof(int) * dcap); da = xrealloc(da, sizeof(double) * dcap); }
                double used = 0; int nd = 0, na = 0;
                for (int p = 0; p < w.rlen[i]; p++) {
                    const int j = w.ri[i][p];
                    if (!w.colon[j]) continue;
                    na++;
                    const double a = w.rv[i][p];
                    double lj = w.l[j], uj = w.u[j];
                    if (sl[j] >= 0 && sl[j] != i && ilo[j] > lj) lj = ilo[j];
                    if (su[j] >= 0 && su[j] != i && iup[j] < uj) uj = iup[j];
                    /* (an entry 1e-12 of the row's largest whose term cannot help the row, a x_j <= 0 in a row bounded below
                     * only, >= 0 in a row bounded above only, at any range: dropping it only loosens the row, the reduced
                     * problem is a relaxation, and the postsolve's move of the largest entry's column makes the point
                     * feasible at a cost of that column's cost times the move: x323 in bley_xs2's rows has no finite bound) */
                    const int loosen = fabs(a) <= 1e-12 * amax && ((w.ru[i] >= PHUGE && w.rl[i] > -PHUGE && ((a < 0 && lj >= 0) || (a > 0 && uj <= 0)))
                                                               || (w.rl[i] <= -PHUGE && w.ru[i] < PHUGE && ((a > 0 && lj >= 0) || (a < 0 && uj <= 0))));
                    if (loosen) { dj[nd] = j; da[nd] = a; nd++; continue; }
                    if (fabs(lj) >= PHUGE || fabs(uj) >= PHUGE) continue;
                    const double wd = fabs(a) * (uj - lj);
                    if (used + wd <= 1e-12 * amax) { used += wd; dj[nd] = j; da[nd] = a; nd++; if (getenv("BRISK_SXPRE_NEGLDBG")) fprintf(stderr, "   [negligible: row %d col %d a %.3g range [%.3g, %.3g] amax %.3g rcnt %d]\n", i, j, a, lj, uj, amax, w.rcnt[i]); }
                }
                if (!nd || nd == na) continue;
                rec_begin(P, R_NEGL); rec_i(P, i); rec_i(P, na);
                for (int p = 0; p < w.rlen[i]; p++) { const int j = w.ri[i][p]; if (!w.colon[j]) continue; rec_i(P, j); rec_d(P, w.rv[i][p]); rec_d(P, w.l[j]); rec_d(P, w.u[j]); }
                rec_d(P, w.rl[i]); rec_d(P, w.ru[i]);
                for (int t = 0; t < nd; t++) {
                    const int j = dj[t]; const double a = da[t];
                    const double b0 = w.l[j] > -PHUGE ? w.l[j] : w.u[j] < PHUGE ? w.u[j] : 0.0, sh = a * b0;
                    if (w.rl[i] > -PHUGE) w.rl[i] -= sh;
                    if (w.ru[i] < PHUGE) w.ru[i] -= sh;
                    w.rmag[i] += fabs(sh);
                    add_entry(&w, i, j, -a, amax);
                }
                nnegl += nd; changed = 1, nch++;
            }
            free(sl); free(su); free(dj); free(da);
        }
        const long nch_i = nch;
        const int ir_run = act_run && !noimplred && !(defer && ir_idle);
        if (act_run && !noimplred && !ir_run) deferred = 1;
        if (ir_run && status == 0) {
            int *sl = xmalloc(sizeof(int) * (n + 1)), *su = xmalloc(sizeof(int) * (n + 1));
            char *src = xcalloc(m + 1, 1), *gone = xcalloc(m + 1, 1);
            implied_col_bounds_c(&w, &ibc, nch, ilo, iup, sl, su);
            for (int i = 0; i < m && status == 0; i++) {
                if (!w.rowon[i] || w.rcnt[i] < 2 || src[i]) continue;
                double a0 = 0, a1 = 0; int n0 = 0, n1 = 0, used = 0, ok = 1;
                for (int p = 0; p < w.rlen[i] && ok; p++) {
                    const int j = w.ri[i][p];
                    if (!w.colon[j]) continue;
                    const double a = w.rv[i][p];
                    double lj = w.l[j], uj = w.u[j];
                    if (sl[j] >= 0 && sl[j] != i && !gone[sl[j]] && ilo[j] > lj) { lj = ilo[j]; used = 1; }
                    if (su[j] >= 0 && su[j] != i && !gone[su[j]] && iup[j] < uj) { uj = iup[j]; used = 1; }
                    const double bl = a > 0 ? lj : uj, bu = a > 0 ? uj : lj;
                    if (fabs(bl) >= PHUGE) n0++; else a0 += a * bl;
                    if (fabs(bu) >= PHUGE) n1++; else a1 += a * bu;
                }
                if (!used) continue;          /* (the plain test above has judged it) */
                const double rl = w.rl[i], ru = w.ru[i];
                /* (a margin: an implied bound is a sum of rounded terms) */
                const double tl = 1e-9 * (1 + fabs(rl)), tu = 1e-9 * (1 + fabs(ru));
                const int lowok = rl <= -PHUGE || (n0 == 0 && a0 >= rl + tl), upok = ru >= PHUGE || (n1 == 0 && a1 <= ru - tu);
                if (!(lowok && upok)) continue;
                for (int p = 0; p < w.rlen[i]; p++) {
                    const int j = w.ri[i][p];
                    if (!w.colon[j]) continue;
                    if (sl[j] >= 0 && sl[j] != i && !gone[sl[j]] && ilo[j] > w.l[j]) src[sl[j]] = 1;
                    if (su[j] >= 0 && su[j] != i && !gone[su[j]] && iup[j] < w.u[j]) src[su[j]] = 1;
                }
                rec_begin(P, R_REDROW); rec_i(P, i);
                drop_row(&w, i); gone[i] = 1; nred++; nimplred++; changed = 1, nch++;
            }
            free(sl); free(su); free(src); free(gone);
        }
        if (ir_run) ir_idle = nch == nch_i;
        if (act_run && nch == nch_r) act_clean = nch;
        if (!nocascade && cascade(&w, P, &nsing, &nfix, &nempty, &status, why, nwhy)) changed = 1, nch++;
        if (status) break;
        PT(8); if (status) break;
        /* a pass that removed less than a thousandth of the rows and columns: a chain that moves one link a pass
         * (snp-02-004-104: a dominated column, then a column singleton, 9 rows a pass for over 100 passes, 2.9 s of
         * full passes). The parallel rows and columns once more, then the end. */
        if (changed && pass >= 2) {
            int act1 = 0; for (int i = 0; i < m; i++) act1 += w.rowon[i]; for (int j = 0; j < n; j++) act1 += w.colon[j];
            static double stf_ = -1; if (stf_ < 0) { const char *e_ = getenv("BRISK_SXPRE_STALLFRAC"); stf_ = e_ ? atof(e_) : 1e-3; }   /* (13 November 2026, an experiment: the fraction) */
            if ((double)(act0 - act1) < stf_ * act0) { if (parstall) break; parstall = 1; changed = 0; }
        }
        /* (26 October 2026) not again when nothing changed since a run found nothing (pds-20, LP: three runs at the end) */
        if (!changed && !noparallel && nch != par_clean) {
            const long nch_p = nch;
            int inf = 0;
            const int d = parallel_rows(&w, P, &inf);
            if (inf) { status = 1; snprintf(why, nwhy, "two parallel rows have contradicting bounds"); break; }
            if (d) { npar += d; changed = 1, nch++; }
            const int dc = parallel_cols(&w, P);
            if (dc) { nparc += dc; changed = 1, nch++; }
            {   static int spf = -1; if (spf < 0) { const char *e_ = getenv("BRISK_SXPRE_SPARSIFY"); spf = !(e_ && atoi(e_) == 0); }
                const int dsp = spf && !sp_idle ? sparsify_rows(&w, P) : 0;
                if (dsp) { nsparsify += dsp; P->nsparsify += dsp; changed = 1, nch++; } else sp_idle = 1; }
            /* (each while its last call found something: a call that found nothing is not repeated, a full scan each time) */
            const int dd = pd_idle ? 0 : pair_dominated(&w, P);
            pd_idle = 1;            /* (once: a second call found little and cost as much: osa-60, LP, 0.11 s for none) */
            if (dd) { ndomp += dd; changed = 1, nch++; }
            const int dz = zf_idle ? 0 : zero_cost_free(&w, P, rowdropsrc);
            if (dz) { nzfree += dz; changed = 1, nch++; } else zf_idle = 1;
            if (nch == nch_p) par_clean = nch;
        }
        PT(9); if (status) break;
        /* (26 October 2026, a diagnostic: BRISK_SXPRE_PASSES=1) each pass: the rows and columns left, the reductions so far */
        { static int pp = -1; if (pp < 0) pp = getenv("BRISK_SXPRE_PASSES") != NULL;
          static _Thread_local double pt_last = 0; { const double t_ = omp_get_wtime(); if (pp) { long cl_ = 0, ca_ = 0, rl_ = 0, maxcl_ = 0; for (int j_ = 0; j_ < n; j_++) if (w.colon[j_]) { cl_ += w.clen[j_]; if (w.clen[j_] > maxcl_) maxcl_ = w.clen[j_]; for (int p_ = 0; p_ < w.clen[j_]; p_++) ca_ += w.rowon[w.ci[j_][p_]] != 0; } for (int i_ = 0; i_ < m; i_++) if (w.rowon[i_]) rl_ += w.rlen[i_];
              fprintf(stderr, "  [presolve pass %d lists: columns %ld entries listed, %ld in active rows, the longest %ld; rows %ld listed]\n", pass, cl_, ca_, maxcl_, rl_); }
          if (pp) { fprintf(stderr, "  [presolve pass %d time %.5f s; steps so far:", pass, pt_last > 0 && pass > 0 ? t_ - pt_last : 0.0); if (ptimes) for (int k_ = 1; k_ < 12; k_++) fprintf(stderr, " %.4f", tt[k_]); fprintf(stderr, "]\n"); } pt_last = t_; }
          if (pp) { int a1 = 0; for (int i = 0; i < m; i++) a1 += w.rowon[i]; for (int j = 0; j < n; j++) a1 += w.colon[j];
                    fprintf(stderr, "  [presolve pass %d: %d rows and columns left (-%d), changed %d, deferred %d; singleton rows %d, fixed %d, dual %d, dominated %d, free %d, slack %d, redundant %d, parallel %d/%d, doubleton %d, substituted %d/%d, forcing %d, empty %d, tight rows %d, implied redundant %d, cost singletons %d, sides %d]\n",
                            pass, a1, act0 - a1, changed, deferred, nsing, nfix, ndual, ndom, nfree, nslack, nred, npar, nparc, ndbl, nagg, naggb, nforce, nempty, ntightrow, nimplred, ncostsing, nside); } }
        /* (a pass without a change that deferred a step: one more pass with every step) */
        if (!changed && !deferred) {
            /* (4 November 2026) the dependent equations once the passes end, then the passes again when some were dropped:
             * a column whose other rows were all dependent is then alone in its row, and the passes reduce it (a slack
             * column, a singleton). satellites2-25 (LP): 589 equations of a network are sums of the short rows each of its
             * 8,129 columns also sits in; dropped at the very end, those columns stayed, and 11,664 of 17,997 rows stayed
             * equations (Gurobi: 2,663 of 16,697). BRISK_SXPRE_DEPREPASS=0: the dependent equations once at the end, no
             * passes after them */
            static int deprep = -1; if (deprep < 0) { const char *e_ = getenv("BRISK_SXPRE_DEPREPASS"); deprep = !(e_ && atoi(e_) == 0); }
            if (deprep && !depdone && !SXPRE_STOPPED() && !(getenv("BRISK_SXPRE_DEPROWS") && atoi(getenv("BRISK_SXPRE_DEPROWS")) == 0)) {
                depdone = 1;
                const int nd_ = dependent_rows(&w, P);
                ndeprow += nd_;
                if (nd_ > 0) { nch++; prev_changed = 1; parstall = 0; continue; }
            }
            if (aggb_late && !aggb_phase && !SXPRE_STOPPED()) { aggb_phase = 1; nch++; prev_changed = 0; parstall = 0; continue; }   /* (the bounded substitution now, then the passes again) */
            /* (12 November 2026) the substitution through an inequality row with no net fill likewise waits until the other steps
             * find nothing more: taken early, it changed rows that the singleton rows and the doubleton equations would have
             * removed (neos4, LP: 3,939 x 3,825 left before, 5,182 x 4,555 with it early, 1,404 -> 1,528 iterations) */
            if (agginet && !agginet_phase && !SXPRE_STOPPED()) {
                agginet_phase = 1;
                /* (only when some column can qualify: one infinite bound, at most 8 entries, in a one-sided row that bounds it
                 * on that side; otherwise the passes are not run again for nothing) */
                int cand_ = 0;
                /* (with BRISK_SXPRE_AGGINETC=1 a column of cost zero is no candidate: it is never substituted) */
                const int aggic_ = !(getenv("BRISK_SXPRE_AGGINETC") && atoi(getenv("BRISK_SXPRE_AGGINETC")) == 0);
                for (int j = 0; j < n && !cand_; j++) {
                    if (!w.colon[j] || w.ccnt[j] < 2 || w.ccnt[j] > 8 || (aggic_ && w.c[j] == 0)) continue;
                    const int li = w.l[j] <= -PHUGE, ui = w.u[j] >= PHUGE;
                    if (li == ui) continue;
                    for (int p = 0; p < w.clen[j] && !cand_; p++) {
                        const int i = w.ci[j][p]; if (!w.rowon[i]) continue;
                        const int le = w.ru[i] < PHUGE; if (le == (w.rl[i] > -PHUGE)) continue;
                        const int below = le == (w.cv[j][p] < 0);
                        if (below ? ui : li) cand_ = 1;
                    }
                }
                /* (the aggregator alone runs again: the other steps keep their marks of the count of changes and run again only
                 * if it substitutes something; before, every step ran once more, 15-50 ms on LPs where nothing was substituted) */
                if (cand_) { agg_clean = -1; ag_idle = 0; prev_changed = 0; parstall = 0; continue; }
            }
            break;
        }
        prev_changed = changed;
    }
    /* (28 October 2026) the dependent equations, once at the end (dependent_rows above), when the passes did not */
    if (status == 0 && !depdone && !SXPRE_STOPPED() && !(getenv("BRISK_SXPRE_DEPROWS") && atoi(getenv("BRISK_SXPRE_DEPROWS")) == 0)) ndeprow = dependent_rows(&w, P);
    /* (15 November 2026) the free columns that are combinations of other free columns with the same costs, fixed at 0 (dependent_free_cols) */
    int ndepfree = 0;
    if (status == 0 && !SXPRE_STOPPED() && !(getenv("BRISK_SXPRE_DEPFREECOL") && atoi(getenv("BRISK_SXPRE_DEPFREECOL")) == 0)) ndepfree = dependent_free_cols(&w, P);
    PT(10);
    run_close(P);
    /* the reduced problem */
    memset(R, 0, sizeof *R);
    if (status == 0) {
        int *rnew = xmalloc(sizeof(int) * (m + 1)), *cnew = xmalloc(sizeof(int) * (n + 1));
        int mr = 0, nr = 0;
        for (int i = 0; i < m; i++) rnew[i] = w.rowon[i] ? mr++ : -1;
        for (int j = 0; j < n; j++) cnew[j] = w.colon[j] ? nr++ : -1;
        P->mr = mr; P->nr = nr;
        P->rmap = xmalloc(sizeof(int) * (mr + 1)); P->cmap = xmalloc(sizeof(int) * (nr + 1));
        for (int i = 0; i < m; i++) if (rnew[i] >= 0) P->rmap[rnew[i]] = i;
        for (int j = 0; j < n; j++) if (cnew[j] >= 0) P->cmap[cnew[j]] = j;
        R->m = mr; R->n = nr;
        long nzr = 0;
        for (int j = 0; j < n; j++) if (w.colon[j]) for (int p = 0; p < w.clen[j]; p++) if (w.rowon[w.ci[j][p]]) nzr++;
        R->Ap = xmalloc(sizeof(int) * (nr + 1)); R->Ai = xmalloc(sizeof(int) * (nzr + 1)); R->Ax = xmalloc(sizeof(double) * (nzr + 1));
        R->c = xmalloc(sizeof(double) * (nr + 1)); R->l = xmalloc(sizeof(double) * (nr + 1)); R->u = xmalloc(sizeof(double) * (nr + 1));
        R->rl = xmalloc(sizeof(double) * (mr + 1)); R->ru = xmalloc(sizeof(double) * (mr + 1));
        int q = 0;
        for (int k = 0; k < nr; k++) {
            const int j = P->cmap[k];
            R->Ap[k] = q;
            const int q0 = q;
            for (int p = 0; p < w.clen[j]; p++) if (w.rowon[w.ci[j][p]]) { R->Ai[q] = rnew[w.ci[j][p]]; R->Ax[q] = w.cv[j][p]; q++; }
            /* rows in increasing order within each column (insertion sort: the columns are short) */
            for (int a = q0 + 1; a < q; a++) { const int ti = R->Ai[a]; const double tv = R->Ax[a]; int b = a - 1; while (b >= q0 && R->Ai[b] > ti) { R->Ai[b + 1] = R->Ai[b]; R->Ax[b + 1] = R->Ax[b]; b--; } R->Ai[b + 1] = ti; R->Ax[b + 1] = tv; }
            R->c[k] = w.c[j]; R->l[k] = w.l[j]; R->u[k] = w.u[j];
        }
        R->Ap[nr] = q;
        for (int k = 0; k < mr; k++) { const int i = P->rmap[k]; R->rl[k] = w.rl[i]; R->ru[k] = w.ru[i]; }
        /* (31 October 2026) the presolved problem's constant is the whole constant, the problem's own (as minimization) and
         * the presolve's: its objective is that of the problem as read. It had been the presolve's alone, and the
         * interior-point method then measured its gap against c'x without the problem's constant (lp_std_objoff,
         * ipm_point): degenerate_duties (LP, constant -3761, c'x about 3755, objective -0.105) stopped with the
         * objectives 9.6e-5 apart, the objective 1.3e-5 off (an independent benchmark of 29 October) */
        R->c0 = w.off + sg * L->c0; R->maxim = 0;
        snprintf(R->name, sizeof R->name, "%.48s (presolved)", L->name);
        free(rnew); free(cnew);
        PT(11);
        if (ptimes) fprintf(stderr, "  [presolve times: set-up %.4f, rows %.4f, columns %.4f, dominated %.4f, cost-tight rows %.4f, doubletons %.4f, aggregator %.4f, activity %.4f, implied-redundant rows %.4f, parallel %.4f, other %.4f, reduced problem %.4f s]\n",
                            tt[0], tt[1], tt[2], tt[3], tt[4], tt[5], tt[6], tt[7], tt[8], tt[9], tt[10], tt[11]);
        if (verbose && n_aggexact) printf("presolve: %ld substitutions by the exact count of their fill\n", n_aggexact);
        if (verbose) printf("presolve: %d of %d rows and %d of %d columns left (%d singleton rows, %d fixed columns, %d by dual fixing, %d column singletons in equations, %d in inequalities their cost makes tight, %d slack columns, %d column singletons whose cost moved to their row and bounds to its range, %d doubleton equations, %d substituted columns (%d with their bounds kept by their row), %d inequality rows dropped by a change of variables (a column bounded by the row taken as its slack), %d dominated columns, %d columns dominated by another, %d columns of cost zero removed with their rows (always satisfiable), %d parallel rows, %d parallel columns, %d forcing and %d redundant rows, %d empty, %d sides of ranged rows dropped, %d rows made equations by a cost, %d rows redundant by implied bounds, %d dependent equations, %d equations replaced by their difference with another, %d negligible entries dropped, %d free columns fixed at 0 as combinations of other free columns with the same costs)\n",
                            mr, m, nr, n, nsing, nfix, ndual, nfree, ntight, nslack, ncostsing, ndbl, nagg, naggb, naggi, ndom, ndomp, nzfree, npar, nparc, nforce, nred, nempty, nside, ntightrow, nimplred, ndeprow, nsparsify, nnegl, ndepfree);
        if (getenv("BRISK_SXPRE_STATS")) presolved_stats(R);
    }
    if (naggi && getenv("SXDBG")) printf("presolve: substitutions through an inequality: %d; the row's other columns boxed without cost: none %d, one %d, two or more %d\n", naggi, naggi_bz[0], naggi_bz[1], naggi_bz[2]);
    for (int j = 0; j < n; j++) if (!win_pool(&w, w.ci[j])) { free(w.ci[j]); free(w.cv[j]); }
    for (int i = 0; i < m; i++) if (!win_pool(&w, w.ri[i])) { free(w.ri[i]); free(w.rv[i]); }
    free(w.ipool); free(w.vpool);
    free(w.ci); free(w.cv); free(w.clen); free(w.ccap); free(w.ri); free(w.rv); free(w.rlen); free(w.rcap); free(w.ridx); free(w.hkey); free(w.hpos);
    free(w.c); free(w.l); free(w.u); free(w.rl); free(w.ru); free(w.rmag); free(w.rowon); free(w.colon); free(w.cmod); free(w.crec_i); free(w.crec_d); free(w.cver); free(w.crec_ver); free(w.crec_skip); free(w.rcnt); free(w.ccnt);
    free(ylo); free(yup); free(ilo); free(iup); free(ibc.ilo); free(ibc.iup); free(ibc.sl); free(ibc.su); free(y0l); free(y0u); free(ysl); free(ysu); free(inu); free(inl); free(dropsrc); free(trole); free(rowdropsrc); free(rowanydrop);
    free(acache.a0); free(acache.a1); free(acache.big); free(acache.n0); free(acache.n1); free(acache.val); free(acache.a0l); free(acache.a1l);
    *out = P;
    return status;
}

/* a column's reduced cost at the stage of a record: its cost then minus its entries then times the duals of the rows
 * present now (restored, or in the reduced problem); the record's column at *I, *D (as rec_col wrote it), advanced */
static double zt_rec(const SxPre *P, int j, const int **I, const double **D, const double *y, const char *rowpres) {
    const int cnt = **I; (*I)++;
    if (cnt == -2) {        /* (the column as an earlier record has it: rec_col) */
        const long io = (long)(*I)[0] | ((long)(*I)[1] << 31), dof = (long)(*I)[2] | ((long)(*I)[3] << 31);
        *I += 4;
        const int *I2 = P->ibuf + io; const double *D2 = P->dbuf + dof;
        return zt_rec(P, j, &I2, &D2, y, rowpres);
    }
    if (cnt < 0) {          /* the column as read */
        double z = P->c[j];
        for (int p = P->Ap[j]; p < P->Ap[j + 1]; p++) { const int r = P->Ai[p]; if (rowpres[r]) z -= P->Ax[p] * y[r]; }
        return z;
    }
    double z = **D; (*D)++;
    for (int t = 0; t < cnt; t++) { const int r = (*I)[t]; if (rowpres[r]) z -= (*D)[t] * y[r]; }
    *I += cnt; *D += cnt;
    return z;
}

void sxpre_restrict(const SxPre *P, const double *x, const double *y, double *xr, double *yr) {
    /* the records replayed in the presolve's order on a copy (23 October 2026): a merged pair of parallel columns is
     * x_j + lambda x_k in the reduced problem, a merged pair of parallel rows has the dual y_i + lambda y_k, a cost moved
     * to its row shifts the row's dual by -c_j / a (the inverses of the postsolve's steps); the other reductions leave the
     * kept values as they are. Not exact in general (tools/sxpre_restrict_check.c: the reductions by the costs choose one
     * optimal point among several; on 13 Netlib LPs an optimal basic point restricted so violates the reduced problem's
     * dual conditions): the crossover route no longer uses it (struct SxOrig in simplex.c); kept for that check */
    double *xo = xmalloc(sizeof(double) * (P->n + 1)), *yo = xcalloc(P->m + 1, sizeof(double));
    memcpy(xo, x, sizeof(double) * P->n); if (y) memcpy(yo, y, sizeof(double) * P->m);
    char *allrows = NULL;
    for (int r = 0; r < P->nrec; r++) {
        const int *I = P->ibuf + P->ristart[r]; const double *D = P->dbuf + P->rdstart[r];
        switch (P->rtype[r]) {
        case R_PARCOL: xo[I[0]] += D[0] * xo[I[1]]; break;
        case R_PARROW: yo[I[0]] += D[0] * yo[I[1]]; break;
        case R_ROWDIFF: yo[I[0]] += D[0] * yo[I[1]]; break;      /* (the inverse of the postsolve's y_r -= t y_s) */
        case R_COSTSING: yo[I[0]] -= D[1] / D[0]; break;
        case R_AGGI: {          /* x_j' = x_j - (rhs - s) / a */
            const int cnt = I[2];
            double sx = D[1];
            for (int t = 0; t < cnt; t++) sx -= D[2 + t] * xo[I[3 + t]];
            xo[I[1]] -= sx / D[0];
            break; }
        case R_AGGB: {
            const int i = I[0], j = I[1], cnt = I[2];
            if (!allrows) { allrows = xmalloc(P->m + 1); memset(allrows, 1, P->m); }
            const int *Ic = I + 3 + cnt; const double *Dc = D + 2 + cnt;
            const double zt = zt_rec(P, j, &Ic, &Dc, yo, allrows);
            yo[i] -= zt / D[0];
            break; }
        default: break;
        }
    }
    for (int k = 0; k < P->nr; k++) xr[k] = xo[P->cmap[k]];
    for (int k = 0; k < P->mr; k++) yr[k] = y ? yo[P->rmap[k]] : 0;
    free(xo); free(yo); free(allrows);
}

/* the postsolve: x, the row duals y (as minimization) and the statuses of the problem as read from those of the reduced
 * problem (xr, yr, cstr, rstr; yr as minimization). */
void sxpre_postsolve(const SxPre *P, const double *xr, const double *yr, const int *cstr, const int *rstr, double *x, double *y, int *cst, int *rst) {
    const int m = P->m, n = P->n;
    char *rowpres = xcalloc(m + 1, 1);
    double *rrun = xcalloc(m + 1, sizeof(double));     /* (a run's row rest: R_SLACKREST, R_SLACK2) */
    for (int i = 0; i < m; i++) { y[i] = 0; rst[i] = SX_BASIC; }
    for (int j = 0; j < n; j++) { x[j] = 0; cst[j] = SX_LOWER; }
    for (int k = 0; k < P->mr; k++) { const int i = P->rmap[k]; y[i] = yr ? yr[k] : 0; rst[i] = rstr[k]; rowpres[i] = 1; }
    for (int k = 0; k < P->nr; k++) { const int j = P->cmap[k]; x[j] = xr[k]; cst[j] = cstr[k]; }
    if (getenv("BRISK_LPDUALDBG")) { int cntr[32] = {0}; for (int r = 0; r < P->nrec; r++) if (P->rtype[r] < 32) cntr[P->rtype[r]]++;
        fprintf(stderr, "[duals] the simplex presolve's records:"); for (int t = 1; t < 32; t++) if (cntr[t]) fprintf(stderr, " %d:%d", t, cntr[t]); fprintf(stderr, "\n"); }
    for (int r = P->nrec - 1; r >= 0; r--) {
        const int *I = P->ibuf + P->ristart[r];
        const double *D = P->dbuf + P->rdstart[r];
        switch (P->rtype[r]) {
        case R_EMPTYROW: case R_REDROW: {
            const int i = I[0];
            y[i] = 0; rst[i] = SX_BASIC; rowpres[i] = 1;
            break; }
        case R_NEGL: {
            /* (the row with its negligible entries, as it was: the point restored onto it by the column of its largest entry
             * that can move, the deficit over that entry) */
            const int ne = I[1]; double act = 0;
            for (int t = 0; t < ne; t++) act += D[3 * t] * x[I[2 + t]];
            const double rl = D[3 * ne], ru = D[3 * ne + 1];
            const double def = rl > -PHUGE && act < rl ? rl - act : ru < PHUGE && act > ru ? ru - act : 0.0;
            if (def != 0) {
                int tb = -1; double ab = 0;
                for (int t = 0; t < ne; t++) {
                    const int j = I[2 + t]; const double a = D[3 * t], xn = x[j] + def / a;
                    if (fabs(a) > ab && xn >= D[3 * t + 1] && xn <= D[3 * t + 2]) { ab = fabs(a); tb = t; }
                }
                if (tb >= 0) x[I[2 + tb]] += def / D[3 * tb];
                if (getenv("BRISK_SXPRE_NEGLDBG")) fprintf(stderr, "   [negligible postsolve: row %d act %.6g sides [%.3g, %.3g] deficit %.3g column %d moved by %.3g]\n", I[0], act, rl, ru, def, tb >= 0 ? I[2 + tb] : -1, tb >= 0 ? def / D[3 * tb] : 0.0);
            }
            break; }
        case R_FIXCOL: {
            const int j = I[0];
            x[j] = D[0]; cst[j] = I[1];
            break; }
        case R_ZFREE: {
            /* x_j at the largest of its lower bounds (its own, the rows'): at its own bound nonbasic, the rows basic; at a
             * row's, x_j basic and that row nonbasic at the side that gave it. The rows' duals zero (x_j's reduced cost is
             * its cost, zero) */
            const int j = I[0], k = I[1];
            const double lj = D[0], uj = D[1];
            const int *Ir = I + 2; const double *Dr = D + 2;
            double lo = lj, hi = uj; int ilo = -1, ihi = -1, slo = 0, shi = 0;
            for (int t = 0; t < k; t++) {
                const int i = Ir[0], cnt = Ir[1]; const double a = Dr[0], rl = Dr[1], ru = Dr[2];
                double s = 0;
                for (int q = 0; q < cnt; q++) s += Dr[3 + q] * x[Ir[2 + q]];
                Ir += 2 + cnt; Dr += 3 + cnt;
                y[i] = 0; rst[i] = SX_BASIC; rowpres[i] = 1;
                const double bl = a > 0 ? rl : ru, bu = a > 0 ? ru : rl;
                const int sl = rl == ru ? SX_FIXED : a > 0 ? SX_LOWER : SX_UPPER, su = rl == ru ? SX_FIXED : a > 0 ? SX_UPPER : SX_LOWER;
                if (fabs(bl) < PHUGE) { const double v = (bl - s) / a; if (lo <= -PHUGE || v > lo + 1e-12 * (1 + fabs(lo))) { lo = v; ilo = i; slo = sl; } }
                if (fabs(bu) < PHUGE) { const double v = (bu - s) / a; if (hi >= PHUGE || v < hi - 1e-12 * (1 + fabs(hi))) { hi = v; ihi = i; shi = su; } }
            }
            /* (7 November 2026; BRISK_SXPRE_ZFREEBND, on by default since that day; 0: off) x_j at its own upper bound when no row bounds it below that and its
             * lower bound is not feasible: the same reduced cost (zero), its rows basic with zero duals, x_j nonbasic, so the
             * basis keeps the reduced problem's basic columns. The largest lower bound alone made x_j basic at a value that
             * the rows' other columns decide: on a zero objective (lectsched-1, MILP: big-M rows -103 x_j <= -2 - s) a
             * fraction where its bound 1 is as good, 1,070 of the 1,338 fractional columns of the postsolved point */
            static int zfb = -1; if (zfb < 0) { const char *e_ = getenv("BRISK_SXPRE_ZFREEBND"); zfb = !(e_ && atoi(e_) == 0); }   /* (on by default since 7 November 2026: the default route, 4 threads, 23 problem LPs 1.004, 40 development LPs 1.012, 46 rescaled copies 1.005, the iterations the same on 108 of 109; lectsched-1's root LP, MILP, fewer fractional columns, ISSUES 842); BRISK_SXPRE_ZFREEBND=0: off */
            if (zfb && ilo >= 0 && ihi < 0 && uj < PHUGE && uj == P->u[j] && lo <= uj + 1e-9 * (1 + fabs(uj))) { x[j] = uj; cst[j] = fixstat(uj, lj, uj, P->l[j], P->u[j]); break; }
            if (lo > -PHUGE) { x[j] = lo; if (ilo < 0) cst[j] = fixstat(lo, lj, uj, P->l[j], P->u[j]); else { cst[j] = SX_BASIC; rst[ilo] = slo; } }
            else if (hi < PHUGE) { x[j] = hi; if (ihi < 0) cst[j] = fixstat(hi, lj, uj, P->l[j], P->u[j]); else { cst[j] = SX_BASIC; rst[ihi] = shi; } }
            else { x[j] = 0; cst[j] = SX_ZERO; }
            break; }
        case R_SINGROW: {
            const int i = I[0], j = I[1], fromrow = I[2];
            const double a = D[0], nl = D[1], nu = D[2];
            const int *Ic = I + 3; const double *Dc = D + 3;
            const double zt = zt_rec(P, j, &Ic, &Dc, y, rowpres);
            rowpres[i] = 1;
            /* active at the bound the row gave to the column, with a reduced cost that holds it there (zt >= 0 at a
             * lower bound, <= 0 at an upper)? A column fixed from both sides (two singleton rows, or a row and its own
             * bound) is held by one of them only: the other leaves it (supportcase40: 5,100 rows with duals of the wrong
             * sign, then 9,285 iterations on the problem as read) */
            int act = 0;
            if (cst[j] != SX_BASIC) {
                const double tz = 1e-9 * (1 + fabs(zt));
                const int atl = (fromrow & 1) && fabs(x[j] - nl) <= 1e-9 * (1 + fabs(nl));
                const int atu = (fromrow & 2) && fabs(x[j] - nu) <= 1e-9 * (1 + fabs(nu));
                if (atl && zt >= -tz) act = -1;          /* at the row's lower */
                else if (atu && zt <= tz) act = 1;       /* at the row's upper */
                else if (atl || atu) {                   /* held by the column's own bound, if it is there */
                    if (zt < 0 && P->u[j] < PHUGE && fabs(x[j] - P->u[j]) <= 1e-9 * (1 + fabs(P->u[j]))) cst[j] = P->l[j] == P->u[j] ? SX_FIXED : SX_UPPER;
                    else if (zt > 0 && P->l[j] > -PHUGE && fabs(x[j] - P->l[j]) <= 1e-9 * (1 + fabs(P->l[j]))) cst[j] = P->l[j] == P->u[j] ? SX_FIXED : SX_LOWER;
                }
            }
            if (act) {
                y[i] = zt / a;
                cst[j] = SX_BASIC;
                /* (fixed only if the row was an equation when it was removed: an equation as read whose slack column had
                 * made it a range is at one of its bounds; "fixed" sent the slack's postsolve to the wrong bound, and on
                 * gfd-schedulen180f7d50m30k18 93 rows were infeasible after the postsolve, 11,902 iterations followed) */
                rst[i] = (fromrow & 4) ? SX_FIXED : (act < 0) == (a > 0) ? SX_LOWER : SX_UPPER;
            } else { y[i] = 0; rst[i] = SX_BASIC; }
            break; }
        case R_FREESING: {
            const int i = I[0], j = I[1], cnt = I[2];
            const double a = D[0], rhs = D[1], cj = D[2], tight = D[3];
            double s = rhs;
            for (int t = 0; t < cnt; t++) s -= D[4 + t] * x[I[3 + t]];
            x[j] = s / a; cst[j] = SX_BASIC;
            y[i] = cj / a; rst[i] = tight == 1 ? SX_LOWER : tight == 2 ? SX_UPPER : SX_FIXED; rowpres[i] = 1;
            break; }
        case R_COSTSING: {      /* (after its R_SLACK, which set x_j and the statuses for the equation) */
            const int i = I[0];
            const double a = D[0], cj = D[1], rl = D[2], ru = D[3], rhs = D[4];
            y[i] += cj / a;
            if (rl != ru && rst[i] != SX_BASIC) rst[i] = rhs == rl ? SX_LOWER : SX_UPPER;
            break; }
        case R_PARCOL: {
            const int j = I[0], k = I[1];
            const double lam = D[0], lj = D[1], uj = D[2], lk = D[3], uk = D[4];
            const double v = x[j]; int st = cst[j];
            const double klo = lam > 0 ? lk : uk, kup = lam > 0 ? uk : lk;
            const int sklo = lam > 0 ? SX_LOWER : SX_UPPER, skup = lam > 0 ? SX_UPPER : SX_LOWER;
            /* (27 October 2026, the night) a status at a bound of the merged column is taken only when its value is there:
             * the rows see only the merged value, which the split must keep. A primal point without a basis (the
             * interior-point method's route of -lpmethod 1, all reduced statuses basic) reaches here with statuses that
             * earlier steps of the postsolve gave from their own rules: n13-3 (LP, MIPLIB 2017 collection), 1,020
             * parallel columns, a merged column "at its lower bound" with its value elsewhere, both parts put at their
             * lower bounds; a row was violated by 98 % of its activity and the objective was 4,087.6 for 4,287.6 */
            {
                const double vlo = lj > -PHUGE && fabs(klo) < PHUGE ? lj + lam * klo : -PINF, vup = uj < PHUGE && fabs(kup) < PHUGE ? uj + lam * kup : PINF;
                const int atlo = vlo > -PHUGE && fabs(v - vlo) <= 1e-9 * (1 + fabs(vlo)), atup = vup < PHUGE && fabs(v - vup) <= 1e-9 * (1 + fabs(vup));
                if ((st == SX_LOWER && !atlo) || (st == SX_UPPER && !atup) || (st == SX_FIXED && !atlo && !atup) || (st == SX_ZERO && fabs(v) > 1e-9)) st = SX_BASIC;
                else if (st == SX_FIXED) st = atlo ? SX_LOWER : SX_UPPER;
            }
            if (st == SX_LOWER || (st == SX_FIXED && lj > -PHUGE && fabs(klo) < PHUGE)) {
                x[j] = lj; x[k] = klo; cst[j] = lj == uj ? SX_FIXED : SX_LOWER; cst[k] = lk == uk ? SX_FIXED : sklo;
            } else if (st == SX_UPPER || st == SX_FIXED) {
                x[j] = uj; x[k] = kup; cst[j] = lj == uj ? SX_FIXED : SX_UPPER; cst[k] = lk == uk ? SX_FIXED : skup;
            } else if (st == SX_ZERO) {
                x[j] = 0; x[k] = 0; cst[j] = SX_ZERO; cst[k] = SX_ZERO;
            } else {
                /* inside: k at a bound with j basic, or j at a bound with k basic, whichever fits */
                int done = 0;
                for (int b = 0; b < 2 && !done; b++) {
                    const double kb = b == 0 ? lk : uk;
                    if (fabs(kb) >= PHUGE) continue;
                    const double xj = v - lam * kb;
                    if ((lj <= -PHUGE || xj >= lj - FTOL * (1 + fabs(lj))) && (uj >= PHUGE || xj <= uj + FTOL * (1 + fabs(uj)))) {
                        x[j] = xj; x[k] = kb; cst[j] = SX_BASIC; cst[k] = b == 0 ? SX_LOWER : SX_UPPER; done = 1;
                    }
                }
                for (int b = 0; b < 2 && !done; b++) {
                    const double jb = b == 0 ? lj : uj;
                    if (fabs(jb) >= PHUGE) continue;
                    const double xk = (v - jb) / lam;
                    if ((lk <= -PHUGE || xk >= lk - FTOL * (1 + fabs(lk))) && (uk >= PHUGE || xk <= uk + FTOL * (1 + fabs(uk)))) {
                        x[k] = xk; x[j] = jb; cst[k] = SX_BASIC; cst[j] = b == 0 ? SX_LOWER : SX_UPPER; done = 1;
                    }
                }
                if (!done) { x[k] = 0; cst[k] = SX_ZERO; x[j] = v; cst[j] = SX_BASIC; }
            }
            break; }
        case R_ROWDIFF: y[I[0]] -= D[0] * y[I[1]]; break;
        case R_PARROW: {
            const int i = I[0], k = I[1];
            const double lam = D[0], li = D[1], ui = D[2], lk = D[3], uk = D[4];
            rowpres[k] = 1;
            int st = rst[i];
            if (st == SX_FIXED) {
                if (li == ui) st = -1;                                    /* row i's own equation */
                else if (lk == uk) st = -2;                               /* row k's */
                else st = y[i] >= 0 ? SX_LOWER : SX_UPPER;
            }
            if (st == SX_BASIC || st == SX_ZERO || st == -1) { rst[k] = SX_BASIC; y[k] = 0; break; }
            const int fromk = st == -2 || (st == SX_LOWER && lk > li) || (st == SX_UPPER && uk < ui);
            if (!fromk) { rst[i] = st; rst[k] = SX_BASIC; y[k] = 0; break; }
            /* row k carries the active bound: its own lower (rl_k) or upper (ru_k) as lambda's sign maps it */
            y[k] = y[i] / lam; y[i] = 0; rst[i] = SX_BASIC;
            rst[k] = st == -2 ? SX_FIXED : ((st == SX_LOWER) == (lam > 0)) ? SX_LOWER : SX_UPPER;
            break; }
        case R_SLACKREST: {      /* (the rest of a run's row, summed once; the run's slacks add theirs below) */
            const int i = I[0], cnt = I[1];
            double r = 0; for (int t = 0; t < cnt; t++) r += D[t] * x[I[2 + t]];
            rrun[i] = r;
            break; }
        case R_SLACK: case R_SLACK2: {
            const int sl2 = P->rtype[r] == R_SLACK2;
            const int i = I[0], j = I[1], cnt = sl2 ? 0 : I[2];
            const double a = D[0], l = D[1], u = D[2], rl = D[3], ru = D[4];
            double r = 0;
            if (sl2) r = rrun[i]; else for (int t = 0; t < cnt; t++) r += D[5 + t] * x[I[3 + t]];
            const int eq = rl == ru;
            const double sup = a > 0 ? u : l, slo = a > 0 ? l : u;      /* the s giving max a s, min a s */
            /* (24 October 2026, the night) "fixed" for a row that was not an equation here: made one later (a row the cost
             * makes tight) at either side of its widened range; the side is the one its activity is at (physiciansched6-2,
             * LP: s put at the wrong bound, 15 rows violated by up to 960, 1,532 iterations of the final check) */
            /* (and for a row that was an equation: widened by a slack with one infinite bound, it is one-sided, and made an
             * equation later at its finite side; dfl001, LP, s in [0, inf) put at +inf. Before the rule only a fixed slack
             * gave a "fixed" widened row, whose two sides are one) */
            if (rst[i] == SX_FIXED) {
                const double lw = rl > -PHUGE && fabs(sup) < PHUGE ? rl - a * sup : -PINF, uw = ru < PHUGE && fabs(slo) < PHUGE ? ru - a * slo : PINF;
                rst[i] = fabs(r - lw) <= fabs(r - uw) ? SX_LOWER : SX_UPPER;
            }
            if (rst[i] == SX_LOWER || rst[i] == SX_FIXED) {
                x[j] = sup; cst[j] = sup == u ? SX_UPPER : SX_LOWER; rst[i] = eq ? SX_FIXED : SX_LOWER;
            } else if (rst[i] == SX_UPPER) {
                x[j] = slo; cst[j] = slo == u ? SX_UPPER : SX_LOWER; rst[i] = eq ? SX_FIXED : SX_UPPER;
            } else if (eq) {
                double v = (rl - r) / a; if (l > -PHUGE && v < l) v = l; if (u < PHUGE && v > u) v = u;
                x[j] = v; cst[j] = SX_BASIC; rst[i] = SX_FIXED;
            } else {
                /* the row stays basic with s at a bound when one fits, else s basic and the row at its nearer bound */
                const double tol = 1e-9 * (1 + fabs(r));
                int done = 0;
                for (int b = 0; b < 2 && !done; b++) {
                    const double sb = b == 0 ? l : u;
                    if (sb <= -PHUGE || sb >= PHUGE) continue;
                    const double act = r + a * sb;
                    if ((rl <= -PHUGE || act >= rl - tol) && (ru >= PHUGE || act <= ru + tol)) { x[j] = sb; cst[j] = b == 0 ? SX_LOWER : SX_UPPER; done = 1; }
                }
                if (!done) {
                    const double s0 = l > -PHUGE ? l : u < PHUGE ? u : 0, act0 = r + a * s0;
                    const double t = rl > -PHUGE && act0 < rl ? rl : ru;
                    double v = (t - r) / a; if (l > -PHUGE && v < l) v = l; if (u < PHUGE && v > u) v = u;
                    x[j] = v; cst[j] = SX_BASIC; rst[i] = t == rl ? SX_LOWER : SX_UPPER;
                }
            }
            if (sl2) rrun[i] += a * x[j];
            break; }
        case R_AGG: {
            const int i = I[0], j = I[1], cnt = I[2];
            const double a = D[0], rhs = D[1];
            double sx = rhs;
            for (int t = 0; t < cnt; t++) sx -= D[2 + t] * x[I[3 + t]];
            const int *Ic = I + 3 + cnt; const double *Dc = D + 2 + cnt;
            const double zt = zt_rec(P, j, &Ic, &Dc, y, rowpres);
            x[j] = sx / a; cst[j] = SX_BASIC;
            y[i] = zt / a; rst[i] = SX_FIXED; rowpres[i] = 1;
            break; }
        case R_AGGI: {
            /* x_j = (rhs - s) / a + x_j', x_j' the column's value in the reduced problem: x_j basic, and row i takes x_j''s
             * status (x_j' at its bound 0: the row at its finite side; basic: the row basic). The row's dual: x_j''s reduced
             * cost (the column without row i) over a, as for an equation */
            const int i = I[0], j = I[1], cnt = I[2];
            const double a = D[0], rhs = D[1];
            double sx = rhs;
            for (int t = 0; t < cnt; t++) sx -= D[2 + t] * x[I[3 + t]];
            const int fl = I[3 + cnt];
            const int *Ic = I + 4 + cnt; const double *Dc = D + 2 + cnt;
            const double zt = zt_rec(P, j, &Ic, &Dc, y, rowpres);
            const int sp = cst[j];
            x[j] = sx / a + x[j]; cst[j] = SX_BASIC;
            if (sp == SX_BASIC) { y[i] = 0; rst[i] = SX_BASIC; }
            else { y[i] = zt / a; rst[i] = (fl & 4) ? SX_UPPER : SX_LOWER; }
            rowpres[i] = 1;
            break; }
        case R_AGGB: {
            /* x_j = (rhs - s) / a, s the row's other terms; row i (present: a range in the reduced problem) is the
             * equality again. Its reduced status gives x_j's: the range basic, x_j basic; at the lower end of s (rhs -
             * a u_j when a > 0) x_j at its upper bound, at the upper end at its lower one. The equality's dual: the
             * column's reduced cost without row i, over a, plus the range's dual (x_j's reduced cost is then -a y_r) */
            const int i = I[0], j = I[1], cnt = I[2];
            const double a = D[0], rhs = D[1];
            double sx = rhs;
            for (int t = 0; t < cnt; t++) sx -= D[2 + t] * x[I[3 + t]];
            const int *Ic = I + 3 + cnt; const double *Dc = D + 2 + cnt;
            const double zt = zt_rec(P, j, &Ic, &Dc, y, rowpres);
            x[j] = sx / a;
            const int rs = rst[i];
            if (rs == SX_LOWER) cst[j] = a > 0 ? SX_UPPER : SX_LOWER;
            else if (rs == SX_UPPER) cst[j] = a > 0 ? SX_LOWER : SX_UPPER;
            else if (rs == SX_FIXED) {
                /* (23 October 2026) the row nonbasic, its side not named: a later record of the same row read the row's
                 * bounds as read, where it is the equality (a forcing row: "fixed" with one of its columns made basic);
                 * the column is at the bound nearer its value (basic as before, the basis had two columns for the row:
                 * singular, 47 columns replaced on greenbea, LP) */
                const double lj = P->l[j], uj = P->u[j];
                const double dl = lj > -1e30 ? fabs(x[j] - lj) : 1e300, du = uj < 1e30 ? fabs(x[j] - uj) : 1e300;
                cst[j] = dl >= 1e300 && du >= 1e300 ? SX_ZERO : dl <= du ? SX_LOWER : SX_UPPER;
            }
            else cst[j] = SX_BASIC;
            if (cst[j] != SX_BASIC && P->l[j] == P->u[j]) cst[j] = SX_FIXED;
            y[i] = zt / a + y[i]; rst[i] = SX_FIXED; rowpres[i] = 1;
            break; }
        case R_DOUBLETON: {
            const int i = I[0], j = I[1], k = I[2], from = I[3];
            const double a = D[0], b = D[1], rhs = D[2], klo = D[3], kup = D[4];
            const int *Ic = I + 4; const double *Dc = D + 5;
            const double ztj = zt_rec(P, j, &Ic, &Dc, y, rowpres);
            const double ztk = zt_rec(P, k, &Ic, &Dc, y, rowpres);
            x[j] = (rhs - b * x[k]) / a;
            rowpres[i] = 1; rst[i] = SX_FIXED;
            /* x_k at a bound that x_j's bounds gave: x_j is at its bound, x_k basic; otherwise x_j basic. At both its
             * bounds (fixed in the reduced problem), x_k is held by the one its reduced cost there points to (>= 0: the
             * lower): pds-20, -x_j - x_k = 0 with both >= 0, x_k fixed at 0 by its own lower bound and by x_j's through
             * the upper; its reduced cost +39 held it at its own bound, the upper was taken, and 362 columns ended with
             * reduced costs of the wrong sign, 464 iterations after the postsolve */
            int atj = 0;
            if (cst[k] != SX_BASIC) {
                const int atlo = fabs(x[k] - klo) <= 1e-9 * (1 + fabs(klo)), atup = fabs(x[k] - kup) <= 1e-9 * (1 + fabs(kup));
                int side = atlo && atup ? (ztk - b * ztj / a >= 0 ? -1 : 1) : atlo ? -1 : atup ? 1 : 0;
                atj = (side == -1 && (from & 1)) || (side == 1 && (from & 2));
                /* (held by its own bound on that side: the status names that side; piperout-08: fixed at 210 from below
                 * through x_j and from above by its own bound 210, it kept the reduced problem's "lower" and went to its
                 * lower bound -16) */
                if (!atj && atlo && atup && cst[k] != SX_FIXED) cst[k] = side < 0 ? SX_LOWER : SX_UPPER;
            }
            if (atj) {
                y[i] = ztk / b;
                cst[k] = SX_BASIC;
                cst[j] = fixstat(x[j], P->l[j], P->u[j], P->l[j], P->u[j]);
            } else {
                y[i] = ztj / a;
                cst[j] = SX_BASIC;
            }
            break; }
        case R_FORCING: {
            const int i = I[0], side = I[1], cnt = I[2];
            /* y_i >= 0 at the lower side (side +1), <= 0 at the upper; each column dual feasible at its bound */
            double yi = 0; int jb = -1;
            const int *Ic = I + 3; const double *Dc = D;
            for (int t = 0; t < cnt; t++) {
                const int j = Ic[0], stj = Ic[1];
                const double a = Dc[0], v = Dc[1];
                Ic += 2; Dc += 2;
                const double zt = zt_rec(P, j, &Ic, &Dc, y, rowpres);
                x[j] = v; cst[j] = stj;
                const double rr = zt / a;
                if (side > 0 ? rr > yi : rr < yi) { yi = rr; jb = j; }
            }
            rowpres[i] = 1;
            y[i] = yi;
            if (jb >= 0) { cst[jb] = SX_BASIC; rst[i] = P->rl[i] == P->ru[i] ? SX_FIXED : side > 0 ? SX_LOWER : SX_UPPER; }
            else rst[i] = SX_BASIC;
            break; }
        }
    }
    /* (24 October 2026, the night) a row whose status names a side it does not have in the problem as read: one made an
     * equation by the presolve (a row the cost makes tight, 27.7), its records' status "fixed" or the side of the equation.
     * It is at the side it has; a ranged row at the side its dual's sign says (physiciansched6-2, LP: 15 rows at the
     * wrong side, the final check 1,532 iterations) */
    for (int i = 0; i < m; i++) {
        if (rst[i] == SX_BASIC || P->rl[i] == P->ru[i]) continue;
        const int nolo = P->rl[i] <= -PHUGE, noup = P->ru[i] >= PHUGE;
        if (rst[i] == SX_FIXED || (rst[i] == SX_LOWER && nolo) || (rst[i] == SX_UPPER && noup))
            rst[i] = nolo ? SX_UPPER : noup ? SX_LOWER : y[i] >= 0 ? SX_LOWER : SX_UPPER;
    }
    free(rowpres); free(rrun);
}

/* ---------------------------------------------------------------- the dual problem (tall problems)
 * With x = s + x' (s_j the finite lower bound, else the finite upper bound, else 0) the problem is
 *     min c'x'  s.t.  rl' <= A x' <= ru',  x'_j >= 0 (lower bound only), <= 0 (upper only), free, or in [0, u_j - l_j],
 * rl' = rl - A s, ru' = ru - A s; fixed columns go into the shift. Its dual, as a minimization, has a variable v_i per
 * row (y_i: >= 0 for a row with a lower bound only, <= 0 for one with an upper bound only, free for an equality; a
 * ranged row gives two, v+ >= 0 and v- >= 0 with the columns a_i and -a_i), a variable w_j >= 0 per boxed column,
 *     min  -sum rl'_i v_i (rows with a lower bound) - sum ru'_i v_i (upper only) + sum ru'_i v-_i + sum (u_j - l_j) w_j
 *     s.t. a_j' v - w_j  =  c_j (free x_j),  <= c_j (lower bound, boxed),  >= c_j (upper bound only),
 * one row per column of A. Its basis gives one of the problem by complementarity: a row of the problem is basic when
 * none of its v is basic, else at the bound of the basic v; a column is basic when its dual row is tight (its logical
 * nonbasic) and its w nonbasic, at its upper bound when its w is basic, else at the bound of its shift. */
struct SxDual { int m, n, md, nd; int *vrow, *vkind, *wcol, *drow; };   /* D's column: row of A (kind 0 plain, 1 v+, 2 v-) or w of a column */
int sxpre_dualize(const LpProb *A0, LpProb *D, SxDual **out) {
    const int m = A0->m, n = A0->n;
    const double sg = A0->maxim ? -1.0 : 1.0;
    SxDual *M = xcalloc(1, sizeof(SxDual));
    M->m = m; M->n = n;
    double *s = xcalloc(n + 1, sizeof(double)), *rl = xmalloc(sizeof(double) * (m + 1)), *ru = xmalloc(sizeof(double) * (m + 1));
    for (int i = 0; i < m; i++) { rl[i] = A0->rl[i]; ru[i] = A0->ru[i]; }
    M->drow = xmalloc(sizeof(int) * (n + 1));
    int md = 0, nbox = 0;
    for (int j = 0; j < n; j++) {
        const double l = A0->l[j], u = A0->u[j];
        s[j] = l > -PHUGE ? l : u < PHUGE ? u : 0;
        M->drow[j] = l == u ? -1 : md++;
        if (l > -PHUGE && u < PHUGE && l != u) nbox++;
        if (s[j] != 0) for (int p = A0->Ap[j]; p < A0->Ap[j + 1]; p++) { const int i = A0->Ai[p]; if (rl[i] > -PHUGE) rl[i] -= A0->Ax[p] * s[j]; if (ru[i] < PHUGE) ru[i] -= A0->Ax[p] * s[j]; }
    }
    /* the columns of D: the rows of A (two for a ranged row), then the w of the boxed columns */
    int nv = 0;
    for (int i = 0; i < m; i++) { const int lf = rl[i] > -PHUGE, uf = ru[i] < PHUGE; nv += (lf && uf && rl[i] != ru[i]) ? 2 : (lf || uf) ? 1 : 0; }
    const int nd = nv + nbox;
    M->md = md; M->nd = nd;
    M->vrow = xmalloc(sizeof(int) * (nd + 1)); M->vkind = xmalloc(sizeof(int) * (nd + 1)); M->wcol = xmalloc(sizeof(int) * (nd + 1));
    int *rcnt = xcalloc(m + 1, sizeof(int));
    for (int j = 0; j < n; j++) if (M->drow[j] >= 0) for (int p = A0->Ap[j]; p < A0->Ap[j + 1]; p++) rcnt[A0->Ai[p]]++;
    memset(D, 0, sizeof *D);
    D->m = md; D->n = nd; D->maxim = 0;
    D->Ap = xmalloc(sizeof(int) * (nd + 1)); D->c = xmalloc(sizeof(double) * (nd + 1)); D->l = xmalloc(sizeof(double) * (nd + 1)); D->u = xmalloc(sizeof(double) * (nd + 1));
    D->rl = xmalloc(sizeof(double) * (md + 1)); D->ru = xmalloc(sizeof(double) * (md + 1));
    long nnz = 0; int k = 0;
    for (int i = 0; i < m; i++) {
        const int lf = rl[i] > -PHUGE, uf = ru[i] < PHUGE;
        if (!lf && !uf) continue;
        if (lf && uf && rl[i] != ru[i]) {
            M->vrow[k] = i; M->vkind[k] = 1; M->wcol[k] = -1; D->c[k] = -rl[i]; D->l[k] = 0; D->u[k] = PINF; k++;
            M->vrow[k] = i; M->vkind[k] = 2; M->wcol[k] = -1; D->c[k] = ru[i]; D->l[k] = 0; D->u[k] = PINF; k++;
            nnz += 2L * rcnt[i];
        } else {
            M->vrow[k] = i; M->vkind[k] = 0; M->wcol[k] = -1;
            D->c[k] = lf ? -rl[i] : -ru[i];
            D->l[k] = lf && uf ? -PINF : lf ? 0 : -PINF; D->u[k] = lf && uf ? PINF : lf ? PINF : 0;
            k++; nnz += rcnt[i];
        }
    }
    for (int j = 0; j < n; j++) if (A0->l[j] > -PHUGE && A0->u[j] < PHUGE && A0->l[j] != A0->u[j]) {
        M->vrow[k] = -1; M->vkind[k] = 0; M->wcol[k] = j; D->c[k] = A0->u[j] - A0->l[j]; D->l[k] = 0; D->u[k] = PINF; k++; nnz++;
    }
    /* D's columns of the v: the rows of A, by a transpose */
    int *rstart = xmalloc(sizeof(int) * (m + 2));
    rstart[0] = 0; for (int i = 0; i < m; i++) rstart[i + 1] = rstart[i] + rcnt[i];
    int *ti = xmalloc(sizeof(int) * (rstart[m] + 1)); double *tv = xmalloc(sizeof(double) * (rstart[m] + 1));
    for (int i = 0; i <= m; i++) rcnt[i] = rstart[i];
    for (int j = 0; j < n; j++) if (M->drow[j] >= 0) for (int p = A0->Ap[j]; p < A0->Ap[j + 1]; p++) { const int i = A0->Ai[p]; ti[rcnt[i]] = M->drow[j]; tv[rcnt[i]] = A0->Ax[p]; rcnt[i]++; }
    D->Ai = xmalloc(sizeof(int) * (nnz + 1)); D->Ax = xmalloc(sizeof(double) * (nnz + 1));
    long q = 0;
    for (k = 0; k < nd; k++) {
        D->Ap[k] = (int)q;
        if (M->wcol[k] >= 0) { D->Ai[q] = M->drow[M->wcol[k]]; D->Ax[q] = -1; q++; continue; }
        const int i = M->vrow[k]; const double sgn = M->vkind[k] == 2 ? -1.0 : 1.0;
        for (int p = rstart[i]; p < rstart[i + 1]; p++) { D->Ai[q] = ti[p]; D->Ax[q] = sgn * tv[p]; q++; }
    }
    D->Ap[nd] = (int)q;
    /* the rows of D: the dual constraints of the columns */
    for (int j = 0; j < n; j++) {
        const int r = M->drow[j]; if (r < 0) continue;
        const double c = sg * A0->c[j], l = A0->l[j], u = A0->u[j];
        if (l <= -PHUGE && u >= PHUGE) { D->rl[r] = c; D->ru[r] = c; }
        else if (l > -PHUGE) { D->rl[r] = -PINF; D->ru[r] = c; }
        else { D->rl[r] = c; D->ru[r] = PINF; }
    }
    snprintf(D->name, sizeof D->name, "dual");
    free(s); free(rl); free(ru); free(rcnt); free(rstart); free(ti); free(tv);
    *out = M;
    return 0;
}
/* the basis statuses of the problem from those of its dual (cstd over D's columns, rstd over D's rows; row statuses as
 * in simplex.h, SX_LOWER: the row at rl) */
void sxpre_dual_basis(const SxDual *M, const LpProb *A0, const int *cstd, const int *rstd, int *cst, int *rst) {
    const int m = M->m, n = M->n;
    for (int i = 0; i < m; i++) rst[i] = SX_BASIC;
    for (int k = 0; k < M->nd; k++) {
        if (M->wcol[k] >= 0 || cstd[k] != SX_BASIC) continue;
        const int i = M->vrow[k];
        const int lf = A0->rl[i] > -PHUGE, uf = A0->ru[i] < PHUGE;
        rst[i] = lf && uf && A0->rl[i] == A0->ru[i] ? SX_FIXED : M->vkind[k] == 2 || (M->vkind[k] == 0 && !lf) ? SX_UPPER : SX_LOWER;
    }
    char *wb = xcalloc(n + 1, 1);
    for (int k = 0; k < M->nd; k++) if (M->wcol[k] >= 0 && cstd[k] == SX_BASIC) wb[M->wcol[k]] = 1;
    for (int j = 0; j < n; j++) {
        const double l = A0->l[j], u = A0->u[j];
        const int r = M->drow[j];
        const int atlow = l > -PHUGE ? SX_LOWER : u < PHUGE ? SX_UPPER : SX_ZERO;
        if (r < 0) cst[j] = SX_FIXED;
        else if (wb[j]) cst[j] = SX_UPPER;
        else if (rstd[r] == SX_BASIC) cst[j] = atlow;
        else cst[j] = SX_BASIC;
    }
    free(wb);
}
void sxpre_dual_free(SxDual *M) { if (!M) return; free(M->vrow); free(M->vkind); free(M->wcol); free(M->drow); free(M); }

/* ---------------------------------------------------------------- symmetry: the coarsest equitable partition
 * Colour refinement of the rows and columns: rows start coloured by their bounds, columns by cost and bounds; a row's
 * colour is then refined by the multiset of (colour, coefficient) of its entries' columns, a column's by that of its
 * entries' rows, until no class splits. In the partition reached, the rows of a class have the same multiset of
 * (column class, coefficient), and the columns of a class the same multiset of (row class, coefficient) (checked
 * exactly at the end). Averaging any feasible point over the column classes then gives a feasible point of the same
 * objective (each row's activity becomes the average over its class), so the problem restricted to points constant on
 * the classes has the same optimal value: it is the problem with one variable per column class and one row per row
 * class (Grohe, Kersting, Mladenov and Selman, "Dimension reduction via colour refinement", 2014). */
struct SxSym { int m, n, mq, nq; int *rcls, *ccls; double *rs, *cs; };   /* (rs, cs: the factors of the scaled data, or NULL) */
static unsigned long long mix64(unsigned long long z) {
    z += 0x9e3779b97f4a7c15ULL; z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL; z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL; return z ^ (z >> 31);
}
static unsigned long long dbits(double v) { unsigned long long b; if (v == 0) v = 0; memcpy(&b, &v, sizeof b); return b; }
/* a's entries (index, value) of one line as (class, value) pairs sorted, for the exact check */
typedef struct { int c; double v; } CV;
static int cmp_cv(const void *a, const void *b) {
    const CV *x = a, *y = b;
    if (x->c != y->c) return x->c < y->c ? -1 : 1;
    return x->v < y->v ? -1 : x->v > y->v;
}
static int lines_equitable(int k, const int *start, const int *idx, const double *val, const int *cls, const int *ocls, int nclass) {
    int *rep = xmalloc(sizeof(int) * (nclass + 1)); for (int c = 0; c < nclass; c++) rep[c] = -1;
    int maxl = 0; for (int i = 0; i < k; i++) if (start[i + 1] - start[i] > maxl) maxl = start[i + 1] - start[i];
    CV *a = xmalloc(sizeof(CV) * (maxl + 1)), *b = xmalloc(sizeof(CV) * (maxl + 1));
    int ok = 1;
    for (int i = 0; i < k && ok; i++) {
        const int c = cls[i];
        if (rep[c] < 0) { rep[c] = i; continue; }
        const int r = rep[c], la = start[i + 1] - start[i], lb = start[r + 1] - start[r];
        if (la != lb) { ok = 0; break; }
        for (int p = 0; p < la; p++) { a[p].c = ocls[idx[start[i] + p]]; a[p].v = val[start[i] + p]; b[p].c = ocls[idx[start[r] + p]]; b[p].v = val[start[r] + p]; }
        qsort(a, la, sizeof(CV), cmp_cv); qsort(b, lb, sizeof(CV), cmp_cv);
        for (int p = 0; p < la; p++) if (a[p].c != b[p].c || a[p].v != b[p].v) { ok = 0; break; }
    }
    free(rep); free(a); free(b);
    return ok;
}
/* a key of a pair of values (as main.c's lp_bounds_key: both values' bits mixed into a finite double) */
static double canon_key(double lo, double hi) {
    double l = lo > -1e30 ? lo : -INFINITY, h = hi < 1e30 ? hi : INFINITY;
    if (l == 0) l = 0;
    if (h == 0) h = 0;
    unsigned long long a, b; memcpy(&a, &l, 8); memcpy(&b, &h, 8);
    unsigned long long z = (a * 0x9e3779b97f4a7c15ull) ^ ((b + 0x632be59bd9b4e019ull) * 0xbf58476d1ce4e5b9ull);
    z ^= z >> 29; z &= 0x3fffffffffffffffull;
    double v; memcpy(&v, &z, 8);
    return v;
}
/* (7 November 2026) the problem as read on the scaling that does not depend on the input's (lpsolve.c, lp_canon_scale), its
 * values equal up to the rounding of that scaling made equal (lp_snap_values; the two sides of a row together, the two
 * bounds of a column together), for the folding check; 0 when the pattern (the signs, the zero and infinite data) has too
 * many classes or the scaling does not converge. S shares L's pattern and names; its Ax, rl (ru = rl + m), c, l (u = l + n)
 * are its own */
static double canon_sgn(double v) { return v == 0 ? 0.0 : fabs(v) >= 1e29 ? (v > 0 ? 3.0 : -3.0) : v > 0 ? 1.0 : -1.0; }
/* the order of doubles as unsigned keys (the sign bit flipped for positives, all bits for negatives; -0 as +0) */
static inline unsigned long long canon_okey(double v) { if (v == 0) v = 0.0; unsigned long long u; memcpy(&u, &v, 8); return (u >> 63) ? ~u : (u | 0x8000000000000000ull); }
/* idx (k) sorted by key (least significant byte first, stable; the passes whose byte is the same in every key skipped) */
static void canon_radix(const unsigned long long *key, int k, int *idx) {
    int *tmp = malloc(sizeof(int) * ((size_t)k + 1)); size_t cnt[256];
    for (int t = 0; t < k; t++) idx[t] = t;
    for (int sh = 0; sh < 64; sh += 8) {
        memset(cnt, 0, sizeof cnt);
        for (int t = 0; t < k; t++) cnt[(key[t] >> sh) & 255]++;
        int one = 0; for (int b = 0; b < 256; b++) if (cnt[b] == (size_t)k) one = 1;
        if (one) continue;
        size_t s = 0; for (int b = 0; b < 256; b++) { const size_t c = cnt[b]; cnt[b] = s; s += c; }
        for (int t = 0; t < k; t++) { const int e = idx[t]; tmp[cnt[(key[e] >> sh) & 255]++] = e; }
        memcpy(idx, tmp, sizeof(int) * (size_t)k);
    }
    free(tmp);
}
/* group numbers of k values: sorted, a group goes on while two neighbours differ by at most tol relative (chains); zeros,
 * and values of 1e29 or more in size, by their sign. The groups number at most the values left distinct by lp_snap_values
 * at the same tol (which leaves the members of a group wider than 8 tol apart), so a count of classes on them is at most
 * the count on the snapped values */
static unsigned long long canon_mix(unsigned long long x);
static void canon_groups(const double *v, int k, double tol, int *gid) {
    /* zeros group 0, +-infinite (1e29 or more) groups 1 and 2; the distinct finite nonzero values (a hash set: values repeat
     * often) sorted and chained from 3 on, and every value given its distinct value's group */
    int nk = 0;
    for (int t = 0; t < k; t++) { if (v[t] == 0) gid[t] = 0; else if (v[t] >= 1e29) gid[t] = 1; else if (v[t] <= -1e29) gid[t] = 2; else nk++; }
    if (nk == 0) return;
    size_t cap = 16; while (cap < 2 * (size_t)nk) cap <<= 1;
    int *slot = malloc(sizeof(int) * cap); for (size_t h = 0; h < cap; h++) slot[h] = -1;
    double *w = malloc(sizeof(double) * ((size_t)nk + 1)); int nd = 0;
    int *dv = malloc(sizeof(int) * ((size_t)k + 1));      /* (each value's distinct number) */
    for (int t = 0; t < k; t++) {
        if (v[t] == 0 || fabs(v[t]) >= 1e29) continue;
        unsigned long long u; memcpy(&u, &v[t], 8);
        size_t h = (size_t)(canon_mix(u) & (cap - 1));
        while (slot[h] >= 0 && w[slot[h]] != v[t]) h = (h + 1) & (cap - 1);
        if (slot[h] < 0) { slot[h] = nd; w[nd++] = v[t]; }
        dv[t] = slot[h];
    }
    free(slot);
    unsigned long long *key = malloc(sizeof(unsigned long long) * ((size_t)nd + 1)); int *idx = malloc(sizeof(int) * ((size_t)nd + 1)), *dg = malloc(sizeof(int) * ((size_t)nd + 1));
    for (int q = 0; q < nd; q++) key[q] = canon_okey(w[q]);
    canon_radix(key, nd, idx);
    int g = 3;
    for (int q = 0; q < nd; q++) {
        if (q > 0) { const double x = w[idx[q - 1]], y = w[idx[q]];
            if (!(x == y || ((x > 0) == (y > 0) && y - x <= tol * fmax(fabs(x), fabs(y))))) g++; }
        dg[idx[q]] = g;
    }
    for (int t = 0; t < k; t++) if (v[t] != 0 && fabs(v[t]) < 1e29) gid[t] = dg[dv[t]];
    free(key); free(idx); free(dg); free(w); free(dv);
}
/* the number of distinct values (a hash set of open addressing; the order of the values does not matter) */
static long canon_ndistinct_u(unsigned long long *v, int k) {
    if (k <= 0) return 0;
    size_t cap = 16; while (cap < 2 * (size_t)k) cap <<= 1;
    unsigned long long *hs = malloc(sizeof(unsigned long long) * cap); unsigned char *used = calloc(cap, 1);
    long d = 0;
    for (int t = 0; t < k; t++) {
        size_t h = (size_t)(canon_mix(v[t]) & (cap - 1));
        while (used[h] && hs[h] != v[t]) h = (h + 1) & (cap - 1);
        if (!used[h]) { used[h] = 1; hs[h] = v[t]; d++; }
    }
    free(hs); free(used);
    return d;
}
static unsigned long long canon_mix(unsigned long long x) { x += 0x9e3779b97f4a7c15ull; x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ull; x = (x ^ (x >> 27)) * 0x94d049bb133111ebull; return x ^ (x >> 31); }
/* the classes by the sides (rows) and by the costs and bounds (columns) of the data scaled by rs, cs, grouped at tol: the
 * classes the refinement starts from, or fewer (canon_groups) */
static long canon_start_classes(const LpProb *L, const double *rs, const double *cs, double tol) {
    const int m = L->m, n = L->n, mx = m > n ? m : n;
    double *v = malloc(sizeof(double) * (2 * (size_t)mx + 1)); int *g = malloc(sizeof(int) * (2 * (size_t)mx + 1)), *gc = malloc(sizeof(int) * ((size_t)n + 1));
    unsigned long long *key = malloc(sizeof(unsigned long long) * ((size_t)mx + 1));
    for (int i = 0; i < m; i++) { v[i] = fabs(L->rl[i]) < 1e29 ? rs[i] * L->rl[i] : L->rl[i]; v[m + i] = fabs(L->ru[i]) < 1e29 ? rs[i] * L->ru[i] : L->ru[i]; }
    canon_groups(v, 2 * m, tol, g);
    for (int i = 0; i < m; i++) key[i] = canon_mix((unsigned long long)g[i] * 0x100000001b3ull) ^ canon_mix(~(unsigned long long)g[m + i]);
    long ncls = canon_ndistinct_u(key, m);
    for (int j = 0; j < n; j++) v[j] = cs[j] * L->c[j];
    canon_groups(v, n, tol, gc);
    for (int j = 0; j < n; j++) { v[j] = fabs(L->l[j]) < 1e29 ? L->l[j] / cs[j] : L->l[j]; v[n + j] = fabs(L->u[j]) < 1e29 ? L->u[j] / cs[j] : L->u[j]; }
    canon_groups(v, 2 * n, tol, g);
    for (int j = 0; j < n; j++) key[j] = canon_mix((unsigned long long)gc[j] * 0x100000001b3ull) ^ canon_mix((unsigned long long)g[j] * 0x9e3779b97f4a7c15ull + 7) ^ canon_mix(~(unsigned long long)g[n + j]);
    ncls += canon_ndistinct_u(key, n);
    free(v); free(g); free(gc); free(key);
    return ncls;
}
/* (7 November 2026, later) a lower bound on the classes of the scaled partition from quantities that no scaling of the rows
 * and columns changes, computed from the data as given (no scaling solved): for an entry a_ij of row i and column j and a
 * finite side s_i of the row (the lower one when both are finite: the other follows from the ratio), t_ij = s_i c_j / a_ij (row
 * i multiplied by r_i and column j by d_j: (r s)(d c)/(r a d)); for a
 * column, c_j l_j, c_j u_j and l_j / u_j; for a row, rl_i / ru_i; and the ratios c_j / a_ij within a row over the row's
 * largest, s_i / a_ij within a column over the column's largest. Two rows of one class of the scaled partition have equal
 * scaled sides and, entry by entry, equal scaled entries and costs, hence the same multiset of t_ij; two columns of one
 * class likewise (and equal products). So the classes of rows keyed by their sides' signs, their ratio and the multiset of
 * their t_ij (grouped at tol), and of columns keyed by the signs of their data, the products, the ratio and their t_ij, are
 * at most the classes of any equitable partition on the scaled data: when they exceed stop (m + n), no fold. One pass over
 * the entries (n3707, LP: 10,151 such classes as its scaled sides show, found without the scaling) */
static long canon_invariant_classes(const LpProb *L, double tol, int **codes, unsigned long long **rko, unsigned long long **cko, int lite) {
    const int m = L->m, n = L->n, nz = L->Ap[n];
    double tci[8] = {0}; tci[0] = omp_get_wtime();
    double *tl = malloc(sizeof(double) * ((size_t)nz + 1)); int *gl = malloc(sizeof(int) * ((size_t)nz + 1));
#define CI_FIN(v) ((v) != 0 && fabs(v) < 1e29)
    if (lite) for (int k = 0; k < nz; k++) { gl[k] = 0; tl[k] = 0.0; }    /* (lite: the entries by their signs only, below) */
    if (!lite) {
    for (int j = 0; j < n; j++) for (int k = L->Ap[j]; k < L->Ap[j + 1]; k++) {
        const int i = L->Ai[k]; const double a = L->Ax[k], s = CI_FIN(L->rl[i]) ? L->rl[i] : CI_FIN(L->ru[i]) ? L->ru[i] : 0.0;   /* (the other side: the row's ratio, below) */
        tl[k] = a != 0 && s != 0 ? s * L->c[j] / a : 0.0;
    }
    tci[1] = omp_get_wtime();
    canon_groups(tl, nz, tol, gl);
    tci[2] = omp_get_wtime();
    /* the ratios within a row, c_j / a_ij over the row's largest (the row's scale cancels, each column's in its own term),
     * and within a column, s_i / a_ij over the column's largest */
    {   double *rmx = calloc((size_t)m + 1, sizeof(double)), *cmx = calloc((size_t)n + 1, sizeof(double)), *tw = malloc(sizeof(double) * ((size_t)nz + 1));
        int *g2 = malloc(sizeof(int) * ((size_t)nz + 1));
        for (int j = 0; j < n; j++) for (int k = L->Ap[j]; k < L->Ap[j + 1]; k++) {
            const int i = L->Ai[k]; const double a = L->Ax[k], s = CI_FIN(L->rl[i]) ? L->rl[i] : CI_FIN(L->ru[i]) ? L->ru[i] : 0.0;
            if (a != 0 && L->c[j] != 0) rmx[i] = fmax(rmx[i], fabs(L->c[j] / a));
            if (a != 0 && s != 0) cmx[j] = fmax(cmx[j], fabs(s / a));
        }
        for (int j = 0; j < n; j++) for (int k = L->Ap[j]; k < L->Ap[j + 1]; k++) { const int i = L->Ai[k]; tw[k] = L->Ax[k] != 0 && L->c[j] != 0 && rmx[i] > 0 ? L->c[j] / L->Ax[k] / rmx[i] : 0.0; }
        tci[3] = omp_get_wtime();
        canon_groups(tw, nz, tol, g2);
        tci[4] = omp_get_wtime();
        for (int k = 0; k < nz; k++) gl[k] = (int)(canon_mix(((unsigned long long)gl[k] << 32) ^ (unsigned long long)g2[k]) & 0x3fffffff);
        for (int j = 0; j < n; j++) for (int k = L->Ap[j]; k < L->Ap[j + 1]; k++) { const int i = L->Ai[k]; const double s = CI_FIN(L->rl[i]) ? L->rl[i] : CI_FIN(L->ru[i]) ? L->ru[i] : 0.0;
            tw[k] = L->Ax[k] != 0 && s != 0 && cmx[j] > 0 ? s / L->Ax[k] / cmx[j] : 0.0; }
        canon_groups(tw, nz, tol, g2);
        for (int k = 0; k < nz; k++) gl[k] = (int)(canon_mix(((unsigned long long)gl[k] << 32) ^ (unsigned long long)g2[k] ^ 0x9e37ull) & 0x3fffffff);
        free(rmx); free(cmx); free(tw); free(g2);
    }
    }
    for (int k = 0; k < nz; k++) if (L->Ax[k] < 0) gl[k] ^= 0x40000000;     /* (the entry's sign) */
    free(tl);
    const int mx = m > n ? m : n;
    double *v = malloc(sizeof(double) * (3 * (size_t)mx + 1)); int *g = malloc(sizeof(int) * (3 * (size_t)mx + 1));
    unsigned long long *rk = calloc((size_t)m + 1, sizeof(unsigned long long)), *ck = calloc((size_t)n + 1, sizeof(unsigned long long));
    for (int j = 0; j < n; j++) for (int k = L->Ap[j]; k < L->Ap[j + 1]; k++) {
        const unsigned long long e = canon_mix((unsigned long long)gl[k] * 0x100000001b3ull);
        rk[L->Ai[k]] += canon_mix(e ^ 0x51ed27ull); ck[j] += canon_mix(e ^ 0x2545f4914f6cdd1dull);
    }
    /* the rows' sides: signs, and the ratio */
    for (int i = 0; i < m; i++) v[i] = CI_FIN(L->rl[i]) && CI_FIN(L->ru[i]) ? L->rl[i] / L->ru[i] : 0.0;
    canon_groups(v, m, tol, g);
    for (int i = 0; i < m; i++) rk[i] = canon_mix(rk[i] ^ canon_mix((unsigned long long)g[i] * 0x9e3779b97f4a7c15ull + (unsigned long long)(3 + canon_sgn(L->rl[i])) * 7 + (unsigned long long)(3 + canon_sgn(L->ru[i]))));
    /* the columns: signs, the two products, the ratio */
    for (int j = 0; j < n; j++) { v[j] = CI_FIN(L->l[j]) ? L->c[j] * L->l[j] : 0.0; v[n + j] = CI_FIN(L->u[j]) ? L->c[j] * L->u[j] : 0.0; v[2 * n + j] = CI_FIN(L->l[j]) && CI_FIN(L->u[j]) ? L->l[j] / L->u[j] : 0.0; }
    canon_groups(v, 3 * n, tol, g);
    for (int j = 0; j < n; j++) ck[j] = canon_mix(ck[j] ^ canon_mix((unsigned long long)g[j] * 0x100000001b3ull) ^ canon_mix((unsigned long long)g[n + j] * 0x9e3779b97f4a7c15ull + 1)
                                              ^ canon_mix((unsigned long long)g[2 * n + j] * 0xbf58476d1ce4e5b9ull + 2) ^ (unsigned long long)((3 + canon_sgn(L->c[j])) * 49 + (3 + canon_sgn(L->l[j])) * 7 + (3 + canon_sgn(L->u[j]))));
#undef CI_FIN
    free(v); free(g);
    const long ncls = canon_ndistinct_u(rk, m) + canon_ndistinct_u(ck, n);
    if (getenv("BRISK_LPFOLDPROF") && !lite) printf("[invariant classes: values %.3f, their grouping %.3f, the ratios %.3f, their grouping %.3f, the rest %.3f s]\n", tci[1] - tci[0], tci[2] - tci[1], tci[3] - tci[2], tci[4] - tci[3], omp_get_wtime() - tci[4]);
    if (codes) *codes = gl; else free(gl);
    if (rko) *rko = rk; else free(rk);
    if (cko) *cko = ck; else free(ck);
    return ncls;
}
/* (7 November 2026, later) the checks in order, each a necessary condition for a fold: (1) the classes by the quantities no
 * scaling changes (canon_invariant_classes: one pass over the entries, no scaling solved), then the refinement started from
 * them with the entries labelled by theirs (it refines the partition of the pattern, the signs and the zero and infinite
 * data, which the scaled partition refines too, and it is itself coarser than the scaled partition); (2) the scaling, and the
 * classes the refinement would start from (the scaled sides, costs and bounds, grouped: canon_start_classes); then the
 * scaled problem is built.
 * (A loose scaling first, the conjugate gradient method to 1e-3, was tried as a cheaper rejection and dropped: its factors
 * are 1e-2 to 1e-1 off in their logarithms, the grouping it needs merges the scaled sides into a few dozen groups, 72
 * against 10,151 on n3707 (LP), and it rejects nothing.) Every step is sequential and in a fixed order: the decision does
 * not depend on the threads. */
static int canon_core(const LpProb *L, LpProb *S, double **rso, double **cso, double stop, int verbose) {
    const int m = L->m, n = L->n, nz = L->Ap[n];
    const double t0 = omp_get_wtime();
    const double tol = getenv("BRISK_LPFOLDSCALETOL") ? atof(getenv("BRISK_LPFOLDSCALETOL")) : 1e-9;
    /* (7 November 2026, evening; BRISK_LPFOLDSCALELITE=1, off) the cheapest refinement first: the rows and columns keyed by
     * their own scale-free data (the sides' ratio, the products and ratio of cost and bounds, the signs) and the entries by
     * their signs alone, a refinement that the full one below refines; the entries' scale-free values (three groupings of all
     * the entries) only when it leaves enough classes (square47, LP, 27 million entries: those values 2.0 s, which delayed
     * the interior-point method behind the dual simplex method beside it) */
    static int fcost = -1; if (fcost < 0) fcost = getenv("BRISK_LPFOLDSCALELITE") && atoi(getenv("BRISK_LPFOLDSCALELITE")) > 0;
    for (int lite = fcost ? 1 : 0; lite >= 0; lite--) {   int *code = NULL; unsigned long long *rk = NULL, *ck = NULL;
        const long nci = canon_invariant_classes(L, tol, &code, &rk, &ck, lite);
        if (verbose) printf("   [the folding on the scaled data: %ld classes by the quantities no scaling changes (%.4fs)]\n", nci, omp_get_wtime() - t0);
        if ((double)nci > stop * ((double)m + n)) { free(code); free(rk); free(ck); return 0; }
        /* the refinement from them: the entries labelled by their codes (sign included), the rows and columns started from
         * their keys; it refines the partition of the pattern (signs, zero and infinite data), and every class of the
         * scaled partition lies within one of its classes */
        double *sx = malloc(sizeof(double) * ((size_t)nz + 1)), *hb = malloc(sizeof(double) * ((size_t)m + 1)), *hc = malloc(sizeof(double) * ((size_t)n + 1));
        int *rcl = malloc(sizeof(int) * ((size_t)m + 1)), *ccl = malloc(sizeof(int) * ((size_t)n + 1)), nr = m, nc = n;
        for (int k = 0; k < nz; k++) sx[k] = (double)(code[k] + 1);
        for (int i = 0; i < m; i++) hb[i] = (double)(rk[i] >> 12);
        for (int j = 0; j < n; j++) hc[j] = (double)(ck[j] >> 12);
        free(code); free(rk); free(ck);
        const int f = lp_fold_partition_stop(m, n, 0, L->Ap, L->Ai, sx, hb, hc, NULL, rcl, ccl, &nr, &nc, stop);
        free(sx); free(hb); free(hc); free(rcl); free(ccl);
        if (verbose) printf("   [the folding on the scaled data: their refinement%s %d row and %d column classes (%.4fs)]\n", lite ? " (the entries by their signs)" : "", f ? nr : m, f ? nc : n, omp_get_wtime() - t0);
        if (!(f && (double)nr + nc <= stop * ((double)m + n))) return 0;
    }
    double *rs = malloc(sizeof(double) * ((size_t)m + 1)), *cs = malloc(sizeof(double) * ((size_t)n + 1));
    int its = 0; double rr = 0;
    const int ok = lp_canon_scale(m, n, L->Ap, L->Ai, L->Ax, L->rl, L->ru, L->c, L->l, L->u, rs, cs, &its, &rr);
    if (verbose) printf("   [the folding on the scaled data: the scaling %d conjugate gradient steps, relative residual %.1e (%.4fs)]\n", its, rr, omp_get_wtime() - t0);
    if (!ok) { free(rs); free(cs); return 0; }
    {   const long ncls = canon_start_classes(L, rs, cs, tol);
        if (verbose) printf("   [the folding on the scaled data: %ld classes by the scaled sides, costs and bounds (%.4fs)]\n", ncls, omp_get_wtime() - t0);
        if ((double)ncls > stop * ((double)m + n)) { free(rs); free(cs); return 0; }
    }
    *S = *L;
    S->Ax = malloc(sizeof(double) * ((size_t)nz + 1)); S->rl = malloc(sizeof(double) * (2 * (size_t)m + 1)); S->c = malloc(sizeof(double) * ((size_t)n + 1)); S->l = malloc(sizeof(double) * (2 * (size_t)n + 1));
    S->ru = S->rl + m; S->u = S->l + n;
    for (int i = 0; i < m; i++) { S->rl[i] = fabs(L->rl[i]) < 1e29 ? rs[i] * L->rl[i] : L->rl[i]; S->ru[i] = fabs(L->ru[i]) < 1e29 ? rs[i] * L->ru[i] : L->ru[i]; }
    for (int j = 0; j < n; j++) { S->c[j] = cs[j] * L->c[j]; S->l[j] = fabs(L->l[j]) < 1e29 ? L->l[j] / cs[j] : L->l[j]; S->u[j] = fabs(L->u[j]) < 1e29 ? L->u[j] / cs[j] : L->u[j]; }
    lp_snap_values(S->rl, 2 * m, tol); lp_snap_values(S->c, n, tol); lp_snap_values(S->l, 2 * n, tol);
    for (int j = 0; j < n; j++) for (int k = L->Ap[j]; k < L->Ap[j + 1]; k++) S->Ax[k] = rs[L->Ai[k]] * L->Ax[k] * cs[j];
    lp_snap_values(S->Ax, nz, tol);
    if (rso) *rso = rs; else free(rs);
    if (cso) *cso = cs; else free(cs);
    return 1;
}
int sxpre_canon_problem(const LpProb *L, LpProb *S, double stop, int verbose) { return canon_core(L, S, NULL, NULL, stop, verbose); }
/* (7 November 2026, later) the scaled problem computed once per problem as read and shared by the threads that ask for it
 * (main.c's folding check of the problem as read and the dual simplex jobs' sym_first): the first to ask computes it under
 * the lock, the others wait for it and take the same result. Whoever computes, the result is the same (canon_core is
 * deterministic); only the waiting depends on the schedule. A problem other than the one the object was made for (another
 * size or other data: a job's dual problem) is not served */
void sxpre_canon_init(SxCanon *C, const LpProb *L, double stop) { memset(C, 0, sizeof *C); pthread_mutex_init(&C->mu, NULL); C->L = L; C->stop = stop; C->on = 1; }
static int canon_same(const LpProb *a, const LpProb *b) {
    if (a == b) return 1;
    if (a->m != b->m || a->n != b->n || a->Ap[a->n] != b->Ap[b->n]) return 0;
    const int m = a->m, n = a->n, nz = a->Ap[n];
    return !memcmp(a->Ap, b->Ap, sizeof(int) * (n + 1)) && !memcmp(a->Ai, b->Ai, sizeof(int) * nz) && !memcmp(a->Ax, b->Ax, sizeof(double) * nz)
        && !memcmp(a->rl, b->rl, sizeof(double) * m) && !memcmp(a->ru, b->ru, sizeof(double) * m) && !memcmp(a->c, b->c, sizeof(double) * n)
        && !memcmp(a->l, b->l, sizeof(double) * n) && !memcmp(a->u, b->u, sizeof(double) * n);
}
int sxpre_canon_get(SxCanon *C, const LpProb *L, const LpProb **S, const double **rs, const double **cs, int verbose) {
    if (!C || !C->on || !canon_same(C->L, L)) return -1;     /* (-1: not served; the caller computes its own) */
    pthread_mutex_lock(&C->mu);
    if (!C->done) { C->ok = canon_core(C->L, &C->S, &C->rs, &C->cs, C->stop, verbose); C->done = 1; }
    pthread_mutex_unlock(&C->mu);
    if (C->ok) { if (S) *S = &C->S; if (rs) *rs = C->rs; if (cs) *cs = C->cs; }
    return C->ok;
}
void sxpre_canon_free(SxCanon *C) {
    if (!C || !C->on) return;
    if (C->ok) { free(C->S.Ax); free(C->S.rl); free(C->S.c); free(C->S.l); free(C->rs); free(C->cs); }
    pthread_mutex_destroy(&C->mu); C->on = 0;
}
/* the classes of one side (rows or columns): sizes, the key every member had when the class was formed, and the lines
 * whose class changed in the last split */
typedef struct { int ncls, cap; int *size; unsigned long long *key; int *cnt; int *chg, nchg; } Refine;
static void ref_init(Refine *R, int k, const unsigned long long *key) {
    (void)key;
    R->cap = k + 1; R->ncls = k > 0 ? 1 : 0;
    R->size = xcalloc(R->cap, sizeof(int)); R->key = xcalloc(R->cap, sizeof(unsigned long long)); R->cnt = xcalloc(R->cap, sizeof(int));
    R->chg = xmalloc(sizeof(int) * (k + 1)); R->nchg = 0;
    if (k > 0) { R->size[0] = k; R->key[0] = 0x5bd1e995ULL; }   /* (a key no line has: the first split moves all) */
}
static void ref_free(Refine *R) { free(R->size); free(R->key); free(R->cnt); free(R->chg); }
static _Thread_local const int *g_rcls; static _Thread_local const unsigned long long *g_rkey;   /* (per thread, ISSUES 454) */
static int cmp_aff(const void *a, const void *b) {
    const int i = *(const int *)a, j = *(const int *)b;
    if (g_rcls[i] != g_rcls[j]) return g_rcls[i] < g_rcls[j] ? -1 : 1;
    if (g_rkey[i] != g_rkey[j]) return g_rkey[i] < g_rkey[j] ? -1 : 1;
    return i < j ? -1 : i > j;
}
/* split the classes of the touched lines (a, na; their keys already new): in a class, the touched lines whose key is the
 * class' key stay, the others form new classes by key; when every member of a class is touched and none has its key,
 * the first group keeps the class */
static void ref_split(Refine *R, int *cls, const unsigned long long *key, int *a, int na) {
    R->nchg = 0;
    if (na == 0) return;
    static int oldsort = -1; if (oldsort < 0) oldsort = getenv("BRISK_SXSYMQSORT") != NULL;   /* (the qsort as before) */
    if (oldsort) { g_rcls = cls; g_rkey = key; qsort(a, na, sizeof(int), cmp_aff); }
    else {
        RefKey *v = xmalloc(sizeof(RefKey) * (size_t)na);
        for (int t = 0; t < na; t++) {
            const unsigned i = (unsigned)a[t]; const unsigned long long k = key[i];
            v[t].a = ((unsigned long long)(unsigned)cls[i] << 32) | (k >> 32); v[t].b = (k << 32) | i;
        }
        rk_sort(v, na);
        for (int t = 0; t < na; t++) a[t] = (int)(unsigned)(v[t].b & 0xffffffffULL);
        free(v);
    }
    for (int t = 0; t < na; t++) R->cnt[cls[a[t]]]++;
    int t = 0;
    while (t < na) {
        const int c = cls[a[t]];
        int e = t; while (e < na && cls[a[e]] == c) e++;
        const int all = R->cnt[c] == R->size[c];
        /* (23 October 2026) when every member is touched, the largest group keeps c (Hopcroft's rule: only the lines
         * that move are propagated, the smaller parts); otherwise the members not touched keep it with its key */
        static int large = -1; if (large < 0) large = !(getenv("BRISK_SXSYMLARGE") && atoi(getenv("BRISK_SXSYMLARGE")) == 0);
        if (all && large) {
            int best = -1; unsigned long long kb = 0;
            for (int g = t; g < e; ) {
                int h = g; while (h < e && key[a[h]] == key[a[g]]) h++;
                if (h - g > best) { best = h - g; kb = key[a[g]]; }
                g = h;
            }
            R->key[c] = kb;
        }
        int keep = 0;   /* whether some group keeps c */
        for (int g = t; g < e; ) {
            int h = g; while (h < e && key[a[h]] == key[a[g]]) h++;
            if (key[a[g]] == R->key[c]) keep = 1;
            g = h;
        }
        int first = !keep && all;
        for (int g = t; g < e; ) {
            int h = g; while (h < e && key[a[h]] == key[a[g]]) h++;
            const unsigned long long kg = key[a[g]];
            if (kg == R->key[c]) { g = h; continue; }
            if (first) { R->key[c] = kg; first = 0; g = h; continue; }
            const int nw = R->ncls++;
            R->size[nw] = h - g; R->key[nw] = kg; R->cnt[nw] = 0; R->size[c] -= h - g;
            for (int q = g; q < h; q++) { cls[a[q]] = nw; R->chg[R->nchg++] = a[q]; }
            g = h;
        }
        R->cnt[c] = 0;
        t = e;
    }
}
static int sym_core(const LpProb *A0, LpProb *Q, SxSym **out, int verbose, double budget, const LpProb *V, double maxcls) {
    const int m = A0->m, n = A0->n;
    *out = NULL;
    if (m == 0 || n == 0) return 0;
    const double tsym0_ = verbose ? omp_get_wtime() : 0;   /* (12 November 2026: the search's time in its verbose line) */
    const int nz = A0->Ap[n];
    int *rc = xcalloc(m + 1, sizeof(int)), *cc = xcalloc(n + 1, sizeof(int));
    unsigned long long *rk = xmalloc(sizeof(unsigned long long) * (m + 1)), *ck = xmalloc(sizeof(unsigned long long) * (n + 1));
    for (int i = 0; i < m; i++) rk[i] = mix64(dbits(A0->rl[i]) ^ mix64(dbits(A0->ru[i])));
    for (int j = 0; j < n; j++) ck[j] = mix64(dbits(A0->c[j]) ^ mix64(dbits(A0->l[j]) ^ mix64(dbits(A0->u[j]))));
    const double ratio = getenv("BRISK_SXSYM") ? atof(getenv("BRISK_SXSYM")) : 0.7;   /* (experiments) */
    /* the refinement, incremental: a line's key is recomputed only when the class of one of its neighbours changed,
     * and only the classes holding such lines are split (their other members keep the class' key) */
    Refine Rr, Rc;
    ref_init(&Rr, m, rk);
    ref_init(&Rc, n, ck);
    int *alist = xmalloc(sizeof(int) * ((m > n ? m : n) + 1)); char *amark = xcalloc((m > n ? m : n) + 1, 1);
    int rounds = 0;
    /* the classes by the data: every line touched */
    for (int i = 0; i < m; i++) alist[i] = i;
    ref_split(&Rr, rc, rk, alist, m);
    for (int j = 0; j < n; j++) alist[j] = j;
    ref_split(&Rc, cc, ck, alist, n);
    /* (23 October 2026) the refinement only splits classes: when the classes by the data alone are already too many,
     * nothing of use can follow (lp_transport_1000, LP, 1,000,000 columns of distinct costs: 0.5 s for the rounds, twice
     * since the problem as read is tried too, sym_first) */
    /* (12 November 2026) the caller's limit on the classes (below), off with BRISK_SXSYMCAP=0 */
    static int capon = -1; if (capon < 0) { const char *e_ = getenv("BRISK_SXSYMCAP"); capon = !(e_ && atoi(e_) == 0); }
    if (!capon) maxcls = 0;
    if ((double)(Rr.ncls + Rc.ncls) > ratio * (m + n) || (maxcls > 0 && (double)(Rr.ncls + Rc.ncls) > maxcls)) {
        if (verbose) printf("symmetry: %d x %d in classes %d x %d by the data (not used)\n", m, n, Rr.ncls, Rc.ncls);
        ref_free(&Rr); ref_free(&Rc); free(alist); free(amark); free(rc); free(cc); free(rk); free(ck);
        return 0;
    }
    /* rows of A */
    int *rp = xcalloc(m + 2, sizeof(int)), *ri = xmalloc(sizeof(int) * (nz + 1)); double *rv = xmalloc(sizeof(double) * (nz + 1));
    for (int p = 0; p < nz; p++) rp[A0->Ai[p] + 1]++;
    for (int i = 0; i < m; i++) rp[i + 1] += rp[i];
    { int *f = xmalloc(sizeof(int) * (m + 1)); memcpy(f, rp, sizeof(int) * m);
      for (int j = 0; j < n; j++) for (int p = A0->Ap[j]; p < A0->Ap[j + 1]; p++) { const int q = f[A0->Ai[p]]++; ri[q] = j; rv[q] = A0->Ax[p]; }
      free(f); }
    int first = 1, overbudget = 0, capped = 0; double work = 0;
    for (;;) {
        rounds++;
        /* rows next to columns that changed class (all in the first round) */
        int na = 0;
        if (first) { for (int i = 0; i < m; i++) alist[na++] = i; }
        else for (int t = 0; t < Rc.nchg; t++) { const int j = Rc.chg[t]; for (int p = A0->Ap[j]; p < A0->Ap[j + 1]; p++) { const int i = A0->Ai[p]; if (!amark[i]) { amark[i] = 1; alist[na++] = i; } } }
        for (int t = 0; t < na; t++) { const int i = alist[t]; amark[i] = 0; unsigned long long h = 0; for (int p = rp[i]; p < rp[i + 1]; p++) h += mix64(((unsigned long long)cc[ri[p]] << 32) ^ mix64(dbits(rv[p]))); rk[i] = h; work += rp[i + 1] - rp[i] + 1; }
        ref_split(&Rr, rc, rk, alist, na);
        /* columns next to rows that changed class */
        na = 0;
        if (first) { for (int j = 0; j < n; j++) alist[na++] = j; }
        else for (int t = 0; t < Rr.nchg; t++) { const int i = Rr.chg[t]; for (int p = rp[i]; p < rp[i + 1]; p++) { const int j = ri[p]; if (!amark[j]) { amark[j] = 1; alist[na++] = j; } } }
        for (int t = 0; t < na; t++) { const int j = alist[t]; amark[j] = 0; unsigned long long h = 0; for (int p = A0->Ap[j]; p < A0->Ap[j + 1]; p++) h += mix64(((unsigned long long)rc[A0->Ai[p]] << 32) ^ mix64(dbits(A0->Ax[p]))); ck[j] = h; work += A0->Ap[j + 1] - A0->Ap[j] + 1; }
        ref_split(&Rc, cc, ck, alist, na);
        first = 0;
        if (getenv("BRISK_SXSYMDBG")) fprintf(stderr, "  [symmetry round %d: %d + %d classes of %d + %d (%.3f of the size; caps %.0f, %.0f), %.3f s]\n", rounds, Rr.ncls, Rc.ncls, m, n, (double)(Rr.ncls + Rc.ncls) / (m + n), ratio * (m + n), maxcls, omp_get_wtime() - tsym0_);
        if (Rc.nchg == 0) break;     /* (the rows were refined against these column classes: stable) */
        if ((double)(Rr.ncls + Rc.ncls) > ratio * (m + n)) break;   /* (too many classes to be of use: the check below fails) */
        /* (12 November 2026) the caller's own limit on the classes (sym_first: the problem as read is folded only when its classes
         * are at most 0.7 of the presolved problem's size): the refinement only splits classes, so once past it the result cannot
         * be used. neos-799716 (LP): the problem as read, 40,418 x 30,619, took five rounds to pass 0.7 of its own size, 7.6 % of
         * the instructions of the dual simplex method's solve, while the presolved problem's limit was passed after the first */
        if (maxcls > 0 && (double)(Rr.ncls + Rc.ncls) > maxcls) { capped = 1; break; }
        if (budget > 0 && work > budget) { overbudget = 1; break; }   /* (a caller's cap on the entries the rounds touch) */
        if (SXPRE_STOPPED()) { overbudget = 1; break; }                  /* (the caller's stop flag: no symmetry) */
    }
    int nr = Rr.ncls, nc = Rc.ncls;
    ref_free(&Rr); ref_free(&Rc); free(alist); free(amark);
    int use = !overbudget && !capped && (double)(nr + nc) <= ratio * (m + n) && (nr < m || nc < n);
    if (use) {
        const int e1 = lines_equitable(m, rp, ri, rv, rc, cc, nr), e2 = lines_equitable(n, A0->Ap, A0->Ai, A0->Ax, cc, rc, nc);
        if (getenv("SXDBG")) printf("symmetry check: rows %d columns %d\n", e1, e2);
        use = e1 && e2;
    }
    if (use) {   /* (the data of each class the same: the colours started from hashes of it) */
        int *rr = xmalloc(sizeof(int) * (nr + 1)), *cr = xmalloc(sizeof(int) * (nc + 1));
        for (int c = 0; c < nr; c++) rr[c] = -1;
        for (int c = 0; c < nc; c++) cr[c] = -1;
        for (int i = 0; i < m && use; i++) { const int r = rr[rc[i]]; if (r < 0) rr[rc[i]] = i; else if (A0->rl[i] != A0->rl[r] || A0->ru[i] != A0->ru[r]) use = 0; }
        for (int j = 0; j < n && use; j++) { const int r = cr[cc[j]]; if (r < 0) cr[cc[j]] = j; else if (A0->c[j] != A0->c[r] || A0->l[j] != A0->l[r] || A0->u[j] != A0->u[r]) use = 0; }
        free(rr); free(cr);
    }
    if (verbose) printf("symmetry: %d x %d in classes %d x %d after %d rounds%s (%.3f s)\n", m, n, nr, nc, rounds, use ? "" : " (not used)", omp_get_wtime() - tsym0_);
    if (use) {
        /* the problem on the classes: one row per row class (its first row: the sums of its coefficients over each column
         * class), one column per column class (cost: the class' size times the cost) */
        int *rrep = xmalloc(sizeof(int) * (nr + 1)), *csz = xcalloc(nc + 1, sizeof(int)), *crep = xmalloc(sizeof(int) * (nc + 1));
        for (int c = 0; c < nr; c++) rrep[c] = -1;
        for (int i = m - 1; i >= 0; i--) rrep[rc[i]] = i;
        for (int c = 0; c < nc; c++) crep[c] = -1;
        for (int j = n - 1; j >= 0; j--) { crep[cc[j]] = j; csz[cc[j]]++; }
        /* the entries: column class by column class, rows of each in order */
        int *cnt = xcalloc(nc + 2, sizeof(int)); double *acc = xcalloc(nc + 1, sizeof(double)); int *mark = xmalloc(sizeof(int) * (nc + 1));
        for (int c = 0; c < nc; c++) mark[c] = -1;
        long tot = 0;
        for (int R = 0; R < nr; R++) { const int i = rrep[R]; for (int p = rp[i]; p < rp[i + 1]; p++) { const int C = cc[ri[p]]; if (mark[C] != R) { mark[C] = R; cnt[C + 1]++; tot++; } } }
        for (int c = 0; c < nc; c++) cnt[c + 1] += cnt[c];
        memset(Q, 0, sizeof *Q);
        Q->m = nr; Q->n = nc; Q->maxim = A0->maxim; Q->c0 = A0->c0;
        Q->Ap = xmalloc(sizeof(int) * (nc + 1)); Q->Ai = xmalloc(sizeof(int) * (tot + 1)); Q->Ax = xmalloc(sizeof(double) * (tot + 1));
        /* (7 November 2026, evening) V: the values the problem on the classes takes (the scaled data before the grouping, the
         * partition being that of the grouped data A0): the grouping moves values by up to its width, which no reduction of
         * the presolve that compares values to rounding survives (the parallel rows of neos-4264598-oueme's column-rescaled
         * copy, LP: 1,651 found on the grouped data, against 3,752 on the original's classes) */
        if (V && V != A0) { int *f = xmalloc(sizeof(int) * (m + 1)); memcpy(f, rp, sizeof(int) * m);
            for (int j = 0; j < n; j++) for (int p = V->Ap[j]; p < V->Ap[j + 1]; p++) { const int q = f[V->Ai[p]]++; rv[q] = V->Ax[p]; }
            free(f); }
        else V = A0;
        memcpy(Q->Ap, cnt, sizeof(int) * (nc + 1));
        int *fill = xmalloc(sizeof(int) * (nc + 1)); memcpy(fill, cnt, sizeof(int) * nc);
        for (int c = 0; c < nc; c++) mark[c] = -1;
        for (int R = 0; R < nr; R++) {
            const int i = rrep[R];
            for (int p = rp[i]; p < rp[i + 1]; p++) { const int C = cc[ri[p]]; if (mark[C] != R) { mark[C] = R; acc[C] = 0; } acc[C] += rv[p]; }
            for (int p = rp[i]; p < rp[i + 1]; p++) { const int C = cc[ri[p]]; if (mark[C] == R) { Q->Ai[fill[C]] = R; Q->Ax[fill[C]] = acc[C]; fill[C]++; mark[C] = -2 - R; } }
        }
        Q->c = xmalloc(sizeof(double) * (nc + 1)); Q->l = xmalloc(sizeof(double) * (nc + 1)); Q->u = xmalloc(sizeof(double) * (nc + 1));
        for (int c = 0; c < nc; c++) { const int j = crep[c]; Q->c[c] = csz[c] * V->c[j]; Q->l[c] = V->l[j]; Q->u[c] = V->u[j]; }
        Q->rl = xmalloc(sizeof(double) * (nr + 1)); Q->ru = xmalloc(sizeof(double) * (nr + 1));
        for (int R = 0; R < nr; R++) { Q->rl[R] = V->rl[rrep[R]]; Q->ru[R] = V->ru[rrep[R]]; }
        snprintf(Q->name, sizeof Q->name, "classes");
        SxSym *M = xcalloc(1, sizeof(SxSym));
        M->m = m; M->n = n; M->mq = nr; M->nq = nc; M->rcls = rc; M->ccls = cc; rc = cc = NULL;
        *out = M;
        free(rrep); free(csz); free(crep); free(cnt); free(acc); free(mark); free(fill);
    }
    free(rp); free(ri); free(rv); free(rc); free(cc); free(rk); free(ck);
    return use;
}
/* the point of the problem from that of the problem on the classes: each column at the value of its class */
/* the point of the problem from that of the problem on the classes: each column at the value of its class, each row's
 * dual that of its class divided by the class' size (the class' row stands for the sum of its rows) */
void sxpre_sym_lift(const SxSym *M, const double *xq, const double *yq, double *x, double *y) {
    for (int j = 0; j < M->n; j++) x[j] = xq[M->ccls[j]] * (M->cs ? M->cs[j] : 1.0);
    if (!yq || !y) return;
    int *sz = xcalloc(M->mq + 1, sizeof(int));
    for (int i = 0; i < M->m; i++) sz[M->rcls[i]]++;
    for (int i = 0; i < M->m; i++) y[i] = yq[M->rcls[i]] / sz[M->rcls[i]] * (M->rs ? M->rs[i] : 1.0);
    free(sz);
}
void sxpre_sym_free(SxSym *M) { if (!M) return; free(M->rcls); free(M->ccls); free(M->rs); free(M->cs); free(M); }
/* (7 November 2026) BRISK_LPFOLDSCALE=1: when the data as given have no symmetry of use, the partition of the data on the
 * scaling that does not depend on the input's (sxpre_canon_problem; lpsolve.c, lp_canon_scale): the problem on the classes
 * is then that of the scaled data, and its point is carried back through the scaling (x_j = cs_j x~_j, y_i = rs_i y~_i).
 * Tried on the problem as read only (sym_first), after sxpre_symmetry finds nothing there */
int sxpre_symmetry(const LpProb *A0, LpProb *Q, SxSym **out, int verbose, double budget) { return sym_core(A0, Q, out, verbose, budget, NULL, 0); }
int sxpre_symmetry_cap(const LpProb *A0, LpProb *Q, SxSym **out, int verbose, double budget, double maxcls) { return sym_core(A0, Q, out, verbose, budget, NULL, maxcls); }
int sxpre_symmetry_canon(const LpProb *A0, LpProb *Q, SxSym **out, int verbose, double budget, SxCanon *shared) {
    static int fsc = -1; if (fsc < 0) fsc = getenv("BRISK_LPFOLDSCALE") && atoi(getenv("BRISK_LPFOLDSCALE")) > 0;
    if (!fsc || A0->m == 0 || A0->n == 0 || SXPRE_STOPPED()) return 0;
    const double ratio = getenv("BRISK_SXSYM") ? atof(getenv("BRISK_SXSYM")) : 0.7;
    const int m = A0->m, n = A0->n;
    const LpProb *Sp = NULL; const double *rs0 = NULL, *cs0 = NULL;
    LpProb S; double *rso = NULL, *cso = NULL;
    const int g = sxpre_canon_get(shared, A0, &Sp, &rs0, &cs0, verbose && getenv("BRISK_LPDBG"));
    if (g < 0) { if (!canon_core(A0, &S, &rso, &cso, fmin(ratio, 0.5), verbose && getenv("BRISK_LPDBG"))) return 0; Sp = &S; rs0 = rso; cs0 = cso; }
    else if (g == 0) return 0;
    /* (7 November 2026, evening; BRISK_LPFOLDSCALEQV=1, off) the problem on the classes from the scaled values before the
     * grouping (sym_core's V) */
    static int fqv = -1; if (fqv < 0) fqv = getenv("BRISK_LPFOLDSCALEQV") && atoi(getenv("BRISK_LPFOLDSCALEQV")) > 0;
    LpProb U; int hasu = 0;
    if (fqv) {
        U = *A0; const int nz = A0->Ap[n]; hasu = 1;
        U.Ax = xmalloc(sizeof(double) * (nz + 1)); U.rl = xmalloc(sizeof(double) * (2 * m + 1)); U.ru = U.rl + m; U.c = xmalloc(sizeof(double) * (n + 1)); U.l = xmalloc(sizeof(double) * (2 * n + 1)); U.u = U.l + n;
        for (int j = 0; j < n; j++) for (int k = A0->Ap[j]; k < A0->Ap[j + 1]; k++) U.Ax[k] = rs0[A0->Ai[k]] * A0->Ax[k] * cs0[j];
        for (int i = 0; i < m; i++) { U.rl[i] = fabs(A0->rl[i]) < 1e29 ? rs0[i] * A0->rl[i] : A0->rl[i]; U.ru[i] = fabs(A0->ru[i]) < 1e29 ? rs0[i] * A0->ru[i] : A0->ru[i]; }
        for (int j = 0; j < n; j++) { U.c[j] = cs0[j] * A0->c[j]; U.l[j] = fabs(A0->l[j]) < 1e29 ? A0->l[j] / cs0[j] : A0->l[j]; U.u[j] = fabs(A0->u[j]) < 1e29 ? A0->u[j] / cs0[j] : A0->u[j]; }
    }
    const int r = sym_core(Sp, Q, out, verbose, budget, hasu ? &U : NULL, 0);
    if (hasu) { free(U.Ax); free(U.rl); free(U.c); free(U.l); }
    if (r) {
        double *rs = xmalloc(sizeof(double) * (m + 1)), *cs = xmalloc(sizeof(double) * (n + 1));
        memcpy(rs, rs0, sizeof(double) * m); memcpy(cs, cs0, sizeof(double) * n);
        (*out)->rs = rs; (*out)->cs = cs;
        snprintf(Q->name, sizeof Q->name, "classes (scaled)");
        if (verbose) printf("symmetry: on the scaled data%s\n", g > 0 ? " (shared with the folding check of the problem as read)" : "");
    }
    if (g < 0) { free(S.Ax); free(S.rl); free(S.c); free(S.l); free(rso); free(cso); }
    return r;
}
