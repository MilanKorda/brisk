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
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define PINF 1e30
#define PHUGE 1e29
#define FTOL 1e-9               /* feasibility tolerance of the reductions */
#define ZTOL 1e-12              /* entries below this (relative) vanish when the substitution sums them */

static void *xmalloc(size_t n) { void *p = malloc(n ? n : 1); if (!p) { fprintf(stderr, "brisk: out of memory (presolve)\n"); exit(1); } return p; }
static void *xcalloc(size_t n, size_t s) { void *p = calloc(n ? n : 1, s ? s : 1); if (!p) { fprintf(stderr, "brisk: out of memory (presolve)\n"); exit(1); } return p; }
static void *xrealloc(void *q, size_t n) { void *p = realloc(q, n ? n : 1); if (!p) { fprintf(stderr, "brisk: out of memory (presolve)\n"); exit(1); } return p; }

enum { R_EMPTYROW = 1, R_REDROW, R_SINGROW, R_FIXCOL, R_FREESING, R_FORCING, R_DOUBLETON, R_AGG, R_SLACK, R_PARROW, R_PARCOL, R_COSTSING, R_AGGB, R_SLACK2, R_SLACKREST };

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
};
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
    char *rowon, *colon;
    char *cmod;                                 /* the column's entries or cost changed since it was read */
    int *rcnt, *ccnt;                           /* entries in active rows and columns */
} W;

static void wpush(int **idx, double **val, int *len, int *cap, int k, int t, double v) {
    if (len[k] == cap[k]) { cap[k] = 2 * cap[k] + 4; idx[k] = xrealloc(idx[k], sizeof(int) * cap[k]); val[k] = xrealloc(val[k], sizeof(double) * cap[k]); }
    idx[k][len[k]] = t; val[k][len[k]] = v; len[k]++;
}
static int wfind(int *const *idx, const int *len, int k, int t) { for (int p = 0; p < len[k]; p++) if (idx[k][p] == t) return p; return -1; }
static void wdel(int **idx, double **val, int *len, int k, int p) { len[k]--; idx[k][p] = idx[k][len[k]]; val[k][p] = val[k][len[k]]; }

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
static void rec_col(SxPre *P, const W *w, int j, int skip);
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
        if (w->l[j] > w->u[j] + FTOL * (1 + fabs(w->u[j]))) return 1;
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
            if (w->rl[i] > FTOL * (1 + fabs(w->rl[i])) || w->ru[i] < -FTOL * (1 + fabs(w->ru[i]))) { *status = 1; snprintf(why, nwhy, "row %d is empty and excludes 0", i); break; }
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
static void rec_col(SxPre *P, const W *w, int j, int skip) {
    if (!w->cmod[j]) { rec_i(P, -1); return; }
    int cnt = 0;
    for (int p = 0; p < w->clen[j]; p++) if (w->rowon[w->ci[j][p]] && w->ci[j][p] != skip) cnt++;
    rec_i(P, cnt); rec_d(P, w->c[j]);
    for (int p = 0; p < w->clen[j]; p++) { const int r = w->ci[j][p]; if (w->rowon[r] && r != skip) { rec_i(P, r); rec_d(P, w->cv[j][p]); } }
}
/* a_rk += d for an active row r and column k (an entry added, changed, or removed when it cancels) */
static void add_entry(W *w, int r, int k, double d, double scale) {
    w->cmod[k] = 1;
    const int pr = wfind(w->ri, w->rlen, r, k);
    if (pr < 0) {
        wpush(w->ri, w->rv, w->rlen, w->rcap, r, k, d);
        wpush(w->ci, w->cv, w->clen, w->ccap, k, r, d);
        w->rcnt[r]++; w->ccnt[k]++;
        return;
    }
    const int pc = wfind(w->ci, w->clen, k, r);
    const double v = w->rv[r][pr] + d;
    if (fabs(v) <= ZTOL * scale) {
        wdel(w->ri, w->rv, w->rlen, r, pr); wdel(w->ci, w->cv, w->clen, k, pc);
        w->rcnt[r]--; w->ccnt[k]--;
    } else { w->rv[r][pr] = v; w->cv[k][pc] = v; }
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
static unsigned long long pmix(unsigned long long z) {
    z += 0x9e3779b97f4a7c15ULL; z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL; z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL; return z ^ (z >> 31);
}
static unsigned long long pbits(double v) { unsigned long long b; if (v == 0) v = 0; memcpy(&b, &v, sizeof b); return b; }
static const unsigned long long *g_ph;
static int cmp_ph(const void *a, const void *b) { const int i = *(const int *)a, k = *(const int *)b; if (g_ph[i] != g_ph[k]) return g_ph[i] < g_ph[k] ? -1 : 1; return i < k ? -1 : i > k; }
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
        for (int p = 0; p < w->rlen[i]; p++) { const int j = w->ri[i][p]; if (w->colon[j] && w->rv[i][p] != 0) { hh += pmix(((unsigned long long)j << 32) ^ pmix(pbits(w->rv[i][p] * sc[i]))); c++; } }
        h[i] = hh + pmix((unsigned long long)c); ord[no++] = i;
    }
    g_ph = h; qsort(ord, no, sizeof(int), cmp_ph);
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
                for (int p = 0; p < w->rlen[k] && same; p++) { const int j = w->ri[k][p]; if (!w->colon[j] || w->rv[k][p] == 0) continue; ck++; if (!mk[j] || val[j] != w->rv[k][p] * sc[k]) same = 0; }
                if (!same || ck != ci) continue;
                const double lam = sc[i] / sc[k];          /* a_k = lam a_i */
                const double lk = lam > 0 ? (w->rl[k] > -PHUGE ? w->rl[k] / lam : -PINF) : (w->ru[k] < PHUGE ? w->ru[k] / lam : -PINF);
                const double uk = lam > 0 ? (w->ru[k] < PHUGE ? w->ru[k] / lam : PINF) : (w->rl[k] > -PHUGE ? w->rl[k] / lam : PINF);
                rec_begin(P, R_PARROW); rec_i(P, i); rec_i(P, k); rec_d(P, lam); rec_d(P, w->rl[i]); rec_d(P, w->ru[i]); rec_d(P, lk); rec_d(P, uk);
                double nl = w->rl[i] > lk ? w->rl[i] : lk, nu = w->ru[i] < uk ? w->ru[i] : uk;
                if (nl > nu) { if (nl - nu > FTOL * (1 + fabs(nl))) { *infeas = 1; break; } nu = nl; }
                w->rl[i] = nl; w->ru[i] = nu;
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
/* parallel columns: columns whose active entries and costs are proportional (a_k = lambda a_j, c_k = lambda c_j) act
 * as one variable x_j + lambda x_k: column k is dropped and column j takes the bounds of that sum. The postsolve splits
 * the value: at a bound of the sum both are at the bounds that give it; inside, one of them is basic. */
static int parallel_cols(W *w, SxPre *P) {
    const int n = w->n;
    unsigned long long *h = xmalloc(sizeof(unsigned long long) * (n + 1)); double *sc = xmalloc(sizeof(double) * (n + 1));
    int *ord = xmalloc(sizeof(int) * (n + 1)), no = 0;
    for (int j = 0; j < n; j++) {
        if (!w->colon[j] || w->ccnt[j] < 1) continue;
        int imin = -1; double amin = 0;
        for (int p = 0; p < w->clen[j]; p++) { const int i = w->ci[j][p]; if (w->rowon[i] && w->cv[j][p] != 0 && (imin < 0 || i < imin)) { imin = i; amin = w->cv[j][p]; } }
        if (imin < 0) continue;
        sc[j] = 1.0 / amin;
        unsigned long long hh = 0; int c = 0;
        for (int p = 0; p < w->clen[j]; p++) { const int i = w->ci[j][p]; if (w->rowon[i] && w->cv[j][p] != 0) { hh += pmix(((unsigned long long)i << 32) ^ pmix(pbits(w->cv[j][p] * sc[j]))); c++; } }
        h[j] = hh + pmix((unsigned long long)c); ord[no++] = j;
    }
    g_ph = h; qsort(ord, no, sizeof(int), cmp_ph);
    double *val = xcalloc(w->m + 1, sizeof(double)); char *mk = xcalloc(w->m + 1, 1);
    int ndrop = 0;
    for (int t = 0; t < no; ) {
        int e = t; while (e < no && h[ord[e]] == h[ord[t]]) e++;
        if (e - t > 1) {
            const int j = ord[t];
            int cj = 0;
            for (int p = 0; p < w->clen[j]; p++) { const int i = w->ci[j][p]; if (w->rowon[i] && w->cv[j][p] != 0) { val[i] = w->cv[j][p] * sc[j]; mk[i] = 1; cj++; } }
            for (int u = t + 1; u < e; u++) {
                const int k = ord[u];
                int ck = 0, same = 1;
                for (int p = 0; p < w->clen[k] && same; p++) { const int i = w->ci[k][p]; if (!w->rowon[i] || w->cv[k][p] == 0) continue; ck++; if (!mk[i] || val[i] != w->cv[k][p] * sc[k]) same = 0; }
                if (!same || ck != cj) continue;
                const double lam = sc[j] / sc[k];        /* a_k = lam a_j */
                if (fabs(w->c[k] - lam * w->c[j]) > 1e-12 * (1 + fabs(w->c[k]))) continue;
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
    for (int p = 0; p < w->rlen[i]; p++) { const int k = w->ri[i][p]; if (k != j && w->colon[k]) { w->c[k] -= cj * w->rv[i][p] / a; w->cmod[k] = 1; } }
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
int sxpre_presolve(const LpProb *L, LpProb *R, SxPre **out, int verbose, char *why, size_t nwhy) {
    return sxpre_presolve_m(L, R, out, verbose, -1, why, nwhy);
}
/* (23 October 2026) aggbmode 1: the bounded substitution of columns without cost, at most 4 new entries (the simplex
 * method's own presolve of the race, simplex.c aggb_route); 0: none; -1: as the environment says (experiments) */
int sxpre_presolve_ex(const LpProb *L, LpProb *R, SxPre **out, int verbose, int aggbmode, char *why, size_t nwhy) {
    return sxpre_presolve_m(L, R, out, verbose, aggbmode, why, nwhy);
}
static int sxpre_presolve_m(const LpProb *L, LpProb *R, SxPre **out, int verbose, int aggbmode, char *why, size_t nwhy) {
    const int m = L->m, n = L->n;
    SxPre *P = xcalloc(1, sizeof(SxPre));
    P->m = m; P->n = n; P->run_i = -1;
    const double sg = L->maxim ? -1.0 : 1.0;
    W w; memset(&w, 0, sizeof w);
    w.m = m; w.n = n;
    w.ci = xcalloc(n + 1, sizeof(int *)); w.cv = xcalloc(n + 1, sizeof(double *)); w.clen = xcalloc(n + 1, sizeof(int)); w.ccap = xcalloc(n + 1, sizeof(int));
    w.ri = xcalloc(m + 1, sizeof(int *)); w.rv = xcalloc(m + 1, sizeof(double *)); w.rlen = xcalloc(m + 1, sizeof(int)); w.rcap = xcalloc(m + 1, sizeof(int));
    for (int j = 0; j < n; j++) {
        int c = 0; for (int p = L->Ap[j]; p < L->Ap[j + 1]; p++) if (L->Ax[p] != 0) c++;
        w.ccap[j] = c + 2; w.ci[j] = xmalloc(sizeof(int) * w.ccap[j]); w.cv[j] = xmalloc(sizeof(double) * w.ccap[j]);
        for (int p = L->Ap[j]; p < L->Ap[j + 1]; p++) if (L->Ax[p] != 0) { w.ci[j][w.clen[j]] = L->Ai[p]; w.cv[j][w.clen[j]] = L->Ax[p]; w.clen[j]++; w.rcap[L->Ai[p]]++; }
    }
    for (int i = 0; i < m; i++) { w.rcap[i] += 2; w.ri[i] = xmalloc(sizeof(int) * w.rcap[i]); w.rv[i] = xmalloc(sizeof(double) * w.rcap[i]); }
    for (int j = 0; j < n; j++) for (int p = 0; p < w.clen[j]; p++) { const int i = w.ci[j][p]; w.ri[i][w.rlen[i]] = j; w.rv[i][w.rlen[i]] = w.cv[j][p]; w.rlen[i]++; }
    w.c = xmalloc(sizeof(double) * (n + 1)); w.l = xmalloc(sizeof(double) * (n + 1)); w.u = xmalloc(sizeof(double) * (n + 1));
    w.rl = xmalloc(sizeof(double) * (m + 1)); w.ru = xmalloc(sizeof(double) * (m + 1));
    P->l = xmalloc(sizeof(double) * (n + 1)); P->u = xmalloc(sizeof(double) * (n + 1)); P->rl = xmalloc(sizeof(double) * (m + 1)); P->ru = xmalloc(sizeof(double) * (m + 1));
    {   const long nz = L->Ap[n];
        P->Ap = xmalloc(sizeof(int) * (n + 1)); P->Ai = xmalloc(sizeof(int) * (nz + 1)); P->Ax = xmalloc(sizeof(double) * (nz + 1)); P->c = xmalloc(sizeof(double) * (n + 1));
        memcpy(P->Ap, L->Ap, sizeof(int) * (n + 1)); memcpy(P->Ai, L->Ai, sizeof(int) * nz); memcpy(P->Ax, L->Ax, sizeof(double) * nz);
        for (int j = 0; j < n; j++) P->c[j] = sg * L->c[j]; }
    for (int j = 0; j < n; j++) { w.c[j] = sg * L->c[j]; w.l[j] = L->l[j] <= -PHUGE ? -PINF : L->l[j]; w.u[j] = L->u[j] >= PHUGE ? PINF : L->u[j]; }
    for (int i = 0; i < m; i++) { w.rl[i] = L->rl[i] <= -PHUGE ? -PINF : L->rl[i]; w.ru[i] = L->ru[i] >= PHUGE ? PINF : L->ru[i]; }
    memcpy(P->l, w.l, sizeof(double) * n); memcpy(P->u, w.u, sizeof(double) * n); memcpy(P->rl, w.rl, sizeof(double) * m); memcpy(P->ru, w.ru, sizeof(double) * m);
    w.rowon = xmalloc(m + 1); w.colon = xmalloc(n + 1); memset(w.rowon, 1, m); memset(w.colon, 1, n);
    w.cmod = xcalloc(n + 1, 1);
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
    int *dropsrc = xmalloc(sizeof(int) * (n + 1)); for (int j = 0; j < n; j++) dropsrc[j] = -1;
    char *trole = xcalloc(m + 1, 1);    /* the rows the cost made tight (2) and the rows such an argument rested on (1) */
    char *rowdropsrc = xcalloc(m + 1, 1), *rowanydrop = xcalloc(m + 1, 1);   /* (the guard of the dropped bounds) */   /* the row whose implication dropped x_j's bound */
    const int nocostsing = getenv("BRISK_SXPRE_NOCOSTSING") != NULL;
    const int noparallel = getenv("BRISK_SXPRE_NOPARALLEL") != NULL, nocascade = getenv("BRISK_SXPRE_NOCASCADE") != NULL;
    int nslack = 0, ntight = 0, npar = 0, nparc = 0, ncostsing = 0;
    int nagg = 0, naggb = 0;
    const int aggb = aggbmode >= 0 ? aggbmode == 1 : getenv("BRISK_SXAGGB") && atoi(getenv("BRISK_SXAGGB")) == 1;   /* (experiment) */
    const long aggbfill = aggbmode >= 0 ? 4 : getenv("BRISK_SXAGGBFILL") ? atol(getenv("BRISK_SXAGGBFILL")) : 16;   /* (experiment: the fill allowed a bounded substitution) */
    const int aggbzc = aggbmode >= 0 ? 1 : getenv("BRISK_SXAGGBZC") && atoi(getenv("BRISK_SXAGGBZC")) == 1;   /* (experiment: only columns without cost) */
    int ndom = 0;
    int parstall = 0;
    /* (22 October 2026) a step that ran on the problem as it is now and found nothing finds nothing again: nch counts
     * the reductions, and the column pass, the dominated columns, the aggregator and the activity bounds are skipped
     * while nch is the count at which they last found nothing. A pass after a few reductions elsewhere repeated them in
     * full: lp_cheb_10000x300 (6.04 million nonzeros), one fixed column removed, a second pass of whole-matrix sweeps */
    long nch = 0, col_clean = -1, dom_clean = -1, agg_clean = -1, act_clean = -1;
    /* (25 October 2026) three costly steps that found nothing in their last run wait while the other steps still change
     * the problem (the previous pass changed it), and run again in a pass of every step before the presolve ends: the rows
     * the cost makes tight, the aggregator and the rows redundant by implied bounds sweep every row with the implied
     * bounds. A chain of
     * dominated columns and redundant rows took 14 passes on neos-3402294-bobin (LP: 591,076 rows, 586,984 -> 32,932 over
     * the passes), each sweeping all rows with both steps that found nothing. BRISK_SXPRE_NODEFER=1: every step each pass */
    static int nodefer = -1; if (nodefer < 0) nodefer = getenv("BRISK_SXPRE_NODEFER") != NULL;
    int tr_idle = 0, ir_idle = 0, ag_idle = 0, prev_changed = 0;
    for (int pass = 0; pass < 100 && status == 0; pass++) {
        int changed = 0, deferred = 0;
        const int defer = !nodefer && prev_changed;
        int act0 = 0; for (int i = 0; i < m; i++) act0 += w.rowon[i]; for (int j = 0; j < n; j++) act0 += w.colon[j];
        /* rows: empty, singleton */
        for (int i = 0; i < m && status == 0; i++) {
            if (!w.rowon[i]) continue;
            if (w.rl[i] > w.ru[i] + FTOL * (1 + fabs(w.ru[i]))) { status = 1; snprintf(why, nwhy, "row %d: lower bound above upper bound", i); break; }
            if (w.rcnt[i] == 0) {
                if (w.rl[i] > FTOL * (1 + fabs(w.rl[i])) || w.ru[i] < -FTOL * (1 + fabs(w.ru[i]))) { status = 1; snprintf(why, nwhy, "row %d is empty and excludes 0", i); break; }
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
        /* columns: fixed, empty, dual fixing, free and implied free singletons in equations */
        const long nch_c = nch; const int col_run = nch != col_clean;
        for (int j = 0; j < n && status == 0 && col_run; j++) {
            if (!w.colon[j]) continue;
            if (ISFIXED(&w, j)) { fix_col(&w, P, j); nfix++; changed = 1, nch++; continue; }
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
                drop_col(&w, j, w.l[j]); ndual++; changed = 1, nch++; continue;
            }
            if ((up_ok && cj < 0) || (up_ok && cj == 0 && w.ccnt[j] == 0 && w.u[j] < PHUGE)) {
                if (w.u[j] >= PHUGE) { status = 2; snprintf(why, nwhy, "column %d: unbounded above with a negative cost", j); break; }
                rec_begin(P, R_FIXCOL); rec_i(P, j); rec_i(P, P->l[j] == P->u[j] ? SX_FIXED : SX_UPPER); rec_d(P, w.u[j]);
                drop_col(&w, j, w.u[j]); ndual++; changed = 1, nch++; continue;
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
                    activity(&w, i, j, &lmin, &lmax, &nmin, &nmax);
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
                if (!eqrow && !tight) { if (cj == 0 && w.l[j] < w.u[j] && !noslack) { slack_singleton(&w, P, i, j, a); nslack++; changed = 1, nch++; } continue; }
                int freeok = w.l[j] <= -PHUGE && w.u[j] >= PHUGE;
                if (!freeok && !noimpl) {
                    /* implied free: the row and the other columns' bounds keep x_j within its own bounds */
                    double lmin, lmax; int nmin, nmax;
                    activity(&w, i, j, &lmin, &lmax, &nmin, &nmax);
                    double xlo = -PINF, xup = PINF;          /* x_j = (rhs - rest) / a, rest in [lmin, lmax] */
                    if (a > 0) { if (!nmax) xlo = (rhs - lmax) / a; if (!nmin) xup = (rhs - lmin) / a; }
                    else { if (!nmin) xlo = (rhs - lmin) / a; if (!nmax) xup = (rhs - lmax) / a; }
                    const int lok = w.l[j] <= -PHUGE || (xlo > -PHUGE && xlo >= w.l[j] - FTOL * (1 + fabs(w.l[j])));
                    const int uok = w.u[j] >= PHUGE || (xup < PHUGE && xup <= w.u[j] + FTOL * (1 + fabs(w.u[j])));
                    freeok = lok && uok;
                }
                if (!freeok) {
                    if (cj == 0 && w.l[j] < w.u[j] && !noslack) { slack_singleton(&w, P, i, j, a); nslack++; changed = 1, nch++; }
                    /* (only when the row has one other column, which the row then bounds: the row goes too. In
                     * longer rows the problem is no smaller, only its costs move, and the method's path changed for
                     * the worse: 55 MIPLIB relaxations 0.543 -> 0.584 s in the shifted geometric mean, nw04 1.1 ->
                     * 3.2 s; in a doubleton row thor50dday 49.5 -> 0.15 s) */
                    else if (cj != 0 && w.l[j] < w.u[j] && !nocostsing && w.rcnt[i] == 2) { cost_singleton(&w, P, i, j, a, rhs); slack_singleton(&w, P, i, j, a); ncostsing++; changed = 1, nch++; }
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
                    w.c[k] -= cj * w.rv[i][p] / a; w.cmod[k] = 1;
                }
                w.off += cj * rhs / a;
                w.colon[j] = 0;
                drop_row(&w, i);
                if (tight) ntight++; else nfree++;
                changed = 1, nch++;
            }
        }
        if (status) break;
        if (col_run && nch == nch_c) col_clean = nch;
        /* dominated columns: with the implied bounds on the duals, a reduced cost that keeps its sign puts the column at
         * the bound it prefers (Andersen and Andersen 1995) */
        const long nch_d = nch; const int dom_run = !nodom && nch != dom_clean;
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
        /* doubleton equations: a x_j + b x_k = rhs, x_j substituted (the column with fewer entries, if its coefficient
         * is not small against the other's) */
        if (!nodbl) for (int i = 0; i < m && status == 0; i++) {
            if (!w.rowon[i] || w.rcnt[i] != 2 || w.rl[i] != w.ru[i] || w.rl[i] <= -PHUGE) continue;
            int jj[2] = {-1, -1}; double aa[2] = {0, 0}; int t = 0;
            for (int p = 0; p < w.rlen[i] && t < 2; p++) if (w.colon[w.ri[i][p]]) { jj[t] = w.ri[i][p]; aa[t] = w.rv[i][p]; t++; }
            if (t < 2) continue;
            int s = w.ccnt[jj[0]] <= w.ccnt[jj[1]] ? 0 : 1;
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
                if (w.l[k] > w.u[k] + FTOL * (1 + fabs(w.u[k]))) { status = 1; snprintf(why, nwhy, "row %d: the doubleton's bounds are empty", i); break; }
                const double mid = 0.5 * (w.l[k] + w.u[k]); w.l[k] = w.u[k] = mid;
            }
            /* costs and the other rows */
            w.c[k] -= w.c[j] * b / a; w.cmod[k] = 1;
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
                double scale = fabs(arj * b / a); if (scale < 1) scale = 1;
                add_entry(&w, rr, k, -arj * b / a, scale);
            }
            ndbl++; changed = 1, nch++;
        }
        if (status) break;
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
            implied_col_bounds_c(&w, &ibc, nch, ilo, iup, srl, sru);
            char *stale = xcalloc(n + 1, 1);
            for (int j = 0; j < n && status == 0; j++) {
                if (!w.colon[j] || w.ccnt[j] < 2 || stale[j]) continue;
                const int lok = w.l[j] <= -PHUGE || ilo[j] >= w.l[j] - FTOL * (1 + fabs(w.l[j]));
                const int uok = w.u[j] >= PHUGE || iup[j] <= w.u[j] + FTOL * (1 + fabs(w.u[j]));
                /* (an experiment, BRISK_SXAGGB=1, 23 October 2026) a column with bounds the rows do not imply is
                 * substituted too, its bounds kept by the row, which stays as a range (R_AGGB) */
                const int bnd = !(lok && uok);
                if (bnd && (!aggb || (aggbzc && w.c[j] != 0.0))) continue;
                int ib = -1; double ab = 0; long fb = 0;
                for (int p = 0; p < w.clen[j]; p++) {
                    const int i = w.ci[j][p];
                    if (!w.rowon[i] || w.rl[i] != w.ru[i] || w.rl[i] <= -PHUGE || w.rcnt[i] > 24) continue;
                    const long fill = (long)(w.ccnt[j] - 1) * (w.rcnt[i] - 2);
                    if (fill > (bnd ? aggbfill : 16) || (ib >= 0 && fill >= fb)) continue;
                    double amax = 0;
                    for (int q = 0; q < w.rlen[i]; q++) if (w.colon[w.ri[i][q]] && fabs(w.rv[i][q]) > amax) amax = fabs(w.rv[i][q]);
                    if (fabs(w.cv[j][p]) < 0.01 * amax) continue;
                    ib = i; ab = w.cv[j][p]; fb = fill;
                }
                if (ib < 0) continue;
                const int i = ib; const double a = ab, cj = w.c[j], rhs = w.rl[i];
                if (bnd) w.cmod[j] = 1;      /* (the record lists the column without row i, which stays) */
                if (bnd && getenv("SXDBG") && naggb < 40) printf("    bounded substitution: column %d (bounds %g, %g) by row %d (a %g, rhs %g, %d entries)\n", j, w.l[j], w.u[j], i, ab, w.rl[i], w.rcnt[i]);
                rec_begin(P, bnd ? R_AGGB : R_AGG); rec_i(P, i); rec_i(P, j); rec_d(P, a); rec_d(P, rhs);
                int cnt = 0;
                for (int q = 0; q < w.rlen[i]; q++) if (w.ri[i][q] != j && w.colon[w.ri[i][q]]) cnt++;
                rec_i(P, cnt);
                for (int q = 0; q < w.rlen[i]; q++) { const int k = w.ri[i][q]; if (k != j && w.colon[k]) { rec_i(P, k); rec_d(P, w.rv[i][q]); } }
                rec_col(P, &w, j, i);
                /* the row's other columns and entries (a copy: the substitution changes the rows) */
                int *kk = xmalloc(sizeof(int) * (cnt + 1)); double *ak = xmalloc(sizeof(double) * (cnt + 1)); int t = 0;
                for (int q = 0; q < w.rlen[i]; q++) { const int k = w.ri[i][q]; if (k != j && w.colon[k]) { kk[t] = k; ak[t] = w.rv[i][q]; t++; } }
                for (int q = 0; q < t; q++) { w.c[kk[q]] -= cj * ak[q] / a; w.cmod[kk[q]] = 1; }
                for (int p = 0; p < w.clen[j]; p++) {
                    const int r = w.ci[j][p];
                    if (!w.rowon[r]) continue;
                    for (int q = 0; q < w.rlen[r]; q++) {
                        const int k = w.ri[r][q];
                        if ((srl[k] == r && w.l[k] > -PHUGE) || (sru[k] == r && w.u[k] < PHUGE)) stale[k] = 1;
                    }
                }
                w.off += cj * rhs / a;
                w.colon[j] = 0;
                if (!bnd) drop_row(&w, i);
                else {
                    /* row i without x_j: x_j = (rhs - s) / a in [l_j, u_j] is s in [rhs - a u_j, rhs - a l_j] (a > 0) */
                    const double lj = w.l[j], uj = w.u[j];
                    const double s1 = uj >= PHUGE ? (a > 0 ? -PINF : PINF) : rhs - a * uj, s2 = lj <= -PHUGE ? (a > 0 ? PINF : -PINF) : rhs - a * lj;
                    w.rl[i] = fmin(s1, s2); w.ru[i] = fmax(s1, s2);
                    if (w.rl[i] <= -PHUGE) w.rl[i] = -PINF;
                    if (w.ru[i] >= PHUGE) w.ru[i] = PINF;
                    w.rcnt[i]--;
                    naggb++;
                }
                for (int p = 0; p < w.clen[j]; p++) {
                    const int r = w.ci[j][p];
                    if (!w.rowon[r] || r == i) continue;
                    const double f = w.cv[j][p] / a;
                    w.rcnt[r]--;
                    if (w.rl[r] > -PHUGE) w.rl[r] -= f * rhs;
                    if (w.ru[r] < PHUGE) w.ru[r] -= f * rhs;
                    for (int q = 0; q < t; q++) { double sc = fabs(f * ak[q]); if (sc < 1) sc = 1; add_entry(&w, r, kk[q], -f * ak[q], sc); }
                }
                free(kk); free(ak);
                nagg++; changed = 1, nch++;
            }
            free(stale); free(srl); free(sru);
        }
        if (status) break;
        if (agg_run && nch == nch_a) agg_clean = nch;
        if (agg_run) ag_idle = nch == nch_a;
        /* rows: activity bounds */
        const long nch_r = nch; const int act_run = nch != act_clean;
        for (int i = 0; i < m && status == 0 && act_run; i++) {
            if (!w.rowon[i] || w.rcnt[i] < 2) continue;
            double lmin, lmax; int ninfmin, ninfmax;
            activity(&w, i, -1, &lmin, &lmax, &ninfmin, &ninfmax);
            const double rl = w.rl[i], ru = w.ru[i];
            const double tl = FTOL * (1 + fabs(rl)), tu = FTOL * (1 + fabs(ru));
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
        /* (24 October 2026, the night) rows redundant by the bounds the other rows imply (not only the columns' own): a row
         * whose activity, with each column's bounds tightened by the rows that imply tighter ones, stays within its sides is
         * dropped; the bounds themselves are not changed. In one pass a row that gave such a bound is not dropped, and a
         * dropped row gives none, so no two rows justify each other's removal (neos-1171448, LP: x0 + x567 <= 8 with
         * x0 <= 3 and x567 <= 3 from rows of 22 entries; Gurobi drops them). BRISK_SXPRE_NOIMPLRED=1: off */
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
        /* a pass that removed less than a thousandth of the rows and columns: a chain that moves one link a pass
         * (snp-02-004-104: a dominated column, then a column singleton, 9 rows a pass for over 100 passes, 2.9 s of
         * full passes). The parallel rows and columns once more, then the end. */
        if (changed && pass >= 2) {
            int act1 = 0; for (int i = 0; i < m; i++) act1 += w.rowon[i]; for (int j = 0; j < n; j++) act1 += w.colon[j];
            if ((double)(act0 - act1) < 1e-3 * act0) { if (parstall) break; parstall = 1; changed = 0; }
        }
        if (!changed && !noparallel) {
            int inf = 0;
            const int d = parallel_rows(&w, P, &inf);
            if (inf) { status = 1; snprintf(why, nwhy, "two parallel rows have contradicting bounds"); break; }
            if (d) { npar += d; changed = 1, nch++; }
            const int dc = parallel_cols(&w, P);
            if (dc) { nparc += dc; changed = 1, nch++; }
        }
        /* (a pass without a change that deferred a step: one more pass with every step) */
        if (!changed && !deferred) break;
        prev_changed = changed;
    }
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
        R->c0 = w.off; R->maxim = 0;
        snprintf(R->name, sizeof R->name, "%.48s (presolved)", L->name);
        free(rnew); free(cnew);
        if (verbose) printf("presolve: %d of %d rows and %d of %d columns left (%d singleton rows, %d fixed columns, %d by dual fixing, %d column singletons in equations, %d in inequalities their cost makes tight, %d slack columns, %d column singletons whose cost moved to their row and bounds to its range, %d doubleton equations, %d substituted columns (%d with their bounds kept by their row), %d dominated columns, %d parallel rows, %d parallel columns, %d forcing and %d redundant rows, %d empty, %d sides of ranged rows dropped, %d rows made equations by a cost, %d rows redundant by implied bounds)\n",
                            mr, m, nr, n, nsing, nfix, ndual, nfree, ntight, nslack, ncostsing, ndbl, nagg, naggb, ndom, npar, nparc, nforce, nred, nempty, nside, ntightrow, nimplred);
        if (getenv("BRISK_SXPRE_STATS")) presolved_stats(R);
    }
    for (int j = 0; j < n; j++) { free(w.ci[j]); free(w.cv[j]); }
    for (int i = 0; i < m; i++) { free(w.ri[i]); free(w.rv[i]); }
    free(w.ci); free(w.cv); free(w.clen); free(w.ccap); free(w.ri); free(w.rv); free(w.rlen); free(w.rcap);
    free(w.c); free(w.l); free(w.u); free(w.rl); free(w.ru); free(w.rowon); free(w.colon); free(w.cmod); free(w.rcnt); free(w.ccnt);
    free(ylo); free(yup); free(ilo); free(iup); free(ibc.ilo); free(ibc.iup); free(ibc.sl); free(ibc.su); free(y0l); free(y0u); free(ysl); free(ysu); free(inu); free(inl); free(dropsrc); free(trole); free(rowdropsrc); free(rowanydrop);
    *out = P;
    return status;
}

/* a column's reduced cost at the stage of a record: its cost then minus its entries then times the duals of the rows
 * present now (restored, or in the reduced problem); the record's column at *I, *D (as rec_col wrote it), advanced */
static double zt_rec(const SxPre *P, int j, const int **I, const double **D, const double *y, const char *rowpres) {
    const int cnt = **I; (*I)++;
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
        case R_COSTSING: yo[I[0]] -= D[1] / D[0]; break;
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
    for (int r = P->nrec - 1; r >= 0; r--) {
        const int *I = P->ibuf + P->ristart[r];
        const double *D = P->dbuf + P->rdstart[r];
        switch (P->rtype[r]) {
        case R_EMPTYROW: case R_REDROW: {
            const int i = I[0];
            y[i] = 0; rst[i] = SX_BASIC; rowpres[i] = 1;
            break; }
        case R_FIXCOL: {
            const int j = I[0];
            x[j] = D[0]; cst[j] = I[1];
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
            const double v = x[j]; const int st = cst[j];
            const double klo = lam > 0 ? lk : uk, kup = lam > 0 ? uk : lk;
            const int sklo = lam > 0 ? SX_LOWER : SX_UPPER, skup = lam > 0 ? SX_UPPER : SX_LOWER;
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
struct SxSym { int m, n, mq, nq; int *rcls, *ccls; };
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
static const int *g_rcls; static const unsigned long long *g_rkey;
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
    g_rcls = cls; g_rkey = key;
    qsort(a, na, sizeof(int), cmp_aff);
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
int sxpre_symmetry(const LpProb *A0, LpProb *Q, SxSym **out, int verbose, double budget) {
    const int m = A0->m, n = A0->n;
    *out = NULL;
    if (m == 0 || n == 0) return 0;
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
    if ((double)(Rr.ncls + Rc.ncls) > ratio * (m + n)) {
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
    int first = 1, overbudget = 0; double work = 0;
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
        if (Rc.nchg == 0) break;     /* (the rows were refined against these column classes: stable) */
        if ((double)(Rr.ncls + Rc.ncls) > ratio * (m + n)) break;   /* (too many classes to be of use: the check below fails) */
        if (budget > 0 && work > budget) { overbudget = 1; break; }   /* (a caller's cap on the entries the rounds touch) */
    }
    int nr = Rr.ncls, nc = Rc.ncls;
    ref_free(&Rr); ref_free(&Rc); free(alist); free(amark);
    int use = !overbudget && (double)(nr + nc) <= ratio * (m + n) && (nr < m || nc < n);
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
    if (verbose) printf("symmetry: %d x %d in classes %d x %d after %d rounds%s\n", m, n, nr, nc, rounds, use ? "" : " (not used)");
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
        memcpy(Q->Ap, cnt, sizeof(int) * (nc + 1));
        int *fill = xmalloc(sizeof(int) * (nc + 1)); memcpy(fill, cnt, sizeof(int) * nc);
        for (int c = 0; c < nc; c++) mark[c] = -1;
        for (int R = 0; R < nr; R++) {
            const int i = rrep[R];
            for (int p = rp[i]; p < rp[i + 1]; p++) { const int C = cc[ri[p]]; if (mark[C] != R) { mark[C] = R; acc[C] = 0; } acc[C] += rv[p]; }
            for (int p = rp[i]; p < rp[i + 1]; p++) { const int C = cc[ri[p]]; if (mark[C] == R) { Q->Ai[fill[C]] = R; Q->Ax[fill[C]] = acc[C]; fill[C]++; mark[C] = -2 - R; } }
        }
        Q->c = xmalloc(sizeof(double) * (nc + 1)); Q->l = xmalloc(sizeof(double) * (nc + 1)); Q->u = xmalloc(sizeof(double) * (nc + 1));
        for (int c = 0; c < nc; c++) { const int j = crep[c]; Q->c[c] = csz[c] * A0->c[j]; Q->l[c] = A0->l[j]; Q->u[c] = A0->u[j]; }
        Q->rl = xmalloc(sizeof(double) * (nr + 1)); Q->ru = xmalloc(sizeof(double) * (nr + 1));
        for (int R = 0; R < nr; R++) { Q->rl[R] = A0->rl[rrep[R]]; Q->ru[R] = A0->ru[rrep[R]]; }
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
    for (int j = 0; j < M->n; j++) x[j] = xq[M->ccls[j]];
    if (!yq || !y) return;
    int *sz = xcalloc(M->mq + 1, sizeof(int));
    for (int i = 0; i < M->m; i++) sz[M->rcls[i]]++;
    for (int i = 0; i < M->m; i++) y[i] = yq[M->rcls[i]] / sz[M->rcls[i]];
    free(sz);
}
void sxpre_sym_free(SxSym *M) { if (!M) return; free(M->rcls); free(M->ccls); free(M); }
