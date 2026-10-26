/* lpmodel.c - a linear program kept between solves, for a mixed-integer solver (lpmodel.h).
 *
 * The object holds the problem as given (an LpProb, kept up to date with every change) and, from the first solve on, the
 * simplex method's state (sxmodel.h, simplex.c): the scaled problem, the basis, its factorization with the updates, the
 * values and the steepest-edge weights. A bound or a cost change goes into both. Added or deleted rows change the size of
 * every array of that state: they are recorded (the basis statuses and the scale factors of the rows follow them), and
 * the state is built again on the new problem with the same scaling and that basis at the next solve or query (one setup
 * and one factorization for any number of row changes in between). */
#include "lpmodel.h"
#include "sxmodel.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <pthread.h>
#ifdef _OPENMP
#include <omp.h>
#endif
/* (22 October 2026) the first solve with the presolve goes through simplex_solve, whose presolve, folding and
 * interior-point factorization keep data in variables of their files (sort keys for qsort, the presolve's record, the
 * ordering's settings): two such solves at once would mix them. They take turns (ISSUES 422, open); the re-solves use
 * only the object's own state and run at the same time */
static pthread_mutex_t g_first_mu = PTHREAD_MUTEX_INITIALIZER;

struct LpModel {
    LpProb P;                       /* the problem as given, with every change */
    LmOpts opt;
    SxModel *E;                     /* the simplex method's state (NULL before the first solve or basis) */
    /* row changes since the state was built: the statuses and the row scale factors follow them */
    int rebuild;
    int *cst, *rst;                 /* statuses (n, m) */
    double *csc, *rsc, costsc;      /* scale factors: columns (n), the first msc rows */
    int msc;
    /* the last solve */
    int status, have;               /* have: the results below belong to the problem as it is */
    double obj;
    /* a trial */
    int trial, trial_status, trial_have; double trial_obj;
    SxmSaved *saved;
    double *sl, *su, *srl, *sru, *sc;
};

static void *xmalloc(size_t n) { void *p = malloc(n ? n : 1); if (!p) { fprintf(stderr, "brisk: out of memory (lpmodel)\n"); exit(1); } return p; }
static void *xcalloc(size_t n, size_t s) { void *p = calloc(n ? n : 1, s ? s : 1); if (!p) { fprintf(stderr, "brisk: out of memory (lpmodel)\n"); exit(1); } return p; }
static double *dupd(const double *a, int n) { double *b = xmalloc(sizeof(double) * ((size_t)n + 1)); if (n > 0) memcpy(b, a, sizeof(double) * (size_t)n); return b; }
static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + 1e-9 * t.tv_nsec; }
static double clampinf(double v) { return v >= LP_INF ? LP_INF : v <= -LP_INF ? -LP_INF : v; }

void lm_default_opts(LmOpts *o) {
    memset(o, 0, sizeof *o);
    o->tol_primal = 1e-7; o->tol_dual = 1e-7; o->threads = 1; o->first_presolve = 1;
}

LpModel *lm_create(int m, int n, const int *Ap, const int *Ai, const double *Ax, const double *c, double c0,
                   const double *l, const double *u, const double *rl, const double *ru, int maximize) {
    if (m < 0 || n < 0 || !Ap || (Ap[n] > 0 && (!Ai || !Ax)) || (n > 0 && (!c || !l || !u)) || (m > 0 && (!rl || !ru))) return NULL;
    for (int j = 0; j < n; j++) {
        if (Ap[j + 1] < Ap[j]) return NULL;
        for (int p = Ap[j]; p < Ap[j + 1]; p++) if (Ai[p] < 0 || Ai[p] >= m) return NULL;
    }
    LpModel *M = xcalloc(1, sizeof *M);
    LpProb *P = &M->P;
    P->m = m; P->n = n; P->c0 = c0; P->maxim = maximize ? 1 : 0;
    const int nz = Ap[n];
    P->Ap = xmalloc(sizeof(int) * ((size_t)n + 1)); memcpy(P->Ap, Ap, sizeof(int) * ((size_t)n + 1));
    P->Ai = xmalloc(sizeof(int) * ((size_t)nz + 1)); if (nz) memcpy(P->Ai, Ai, sizeof(int) * (size_t)nz);
    P->Ax = dupd(Ax, nz);
    P->c = dupd(c, n); P->l = dupd(l, n); P->u = dupd(u, n); P->rl = dupd(rl, m); P->ru = dupd(ru, m);
    for (int j = 0; j < n; j++) { P->l[j] = clampinf(P->l[j]); P->u[j] = clampinf(P->u[j]); }
    for (int i = 0; i < m; i++) { P->rl[i] = clampinf(P->rl[i]); P->ru[i] = clampinf(P->ru[i]); }
    lm_default_opts(&M->opt);
    M->status = LM_NOTSOLVED;
    return M;
}
LpModel *lm_create_lp(const LpProb *L) {
    if (!L) return NULL;
    return lm_create(L->m, L->n, L->Ap, L->Ai, L->Ax, L->c, L->c0, L->l, L->u, L->rl, L->ru, L->maxim);
}
static void trial_free(LpModel *M) {
    sxm_saved_free(M->saved); M->saved = NULL;
    free(M->sl); free(M->su); free(M->srl); free(M->sru); free(M->sc);
    M->sl = M->su = M->srl = M->sru = M->sc = NULL;
    M->trial = 0;
}
void lm_free(LpModel *M) {
    if (!M) return;
    trial_free(M);
    sxm_free(M->E);
    free(M->cst); free(M->rst); free(M->csc); free(M->rsc);
    lp_free(&M->P);
    free(M);
}
int lm_rows(const LpModel *M) { return M ? M->P.m : -1; }
int lm_cols(const LpModel *M) { return M ? M->P.n : -1; }
int lm_set_opts(LpModel *M, const LmOpts *o) {
    if (!M || !o) return -1;
    M->opt = *o;
    if (M->opt.threads <= 0) M->opt.threads = 1;
    return 0;
}

/* ---- the state built again after row changes: same column scaling, the old rows' factors, the statuses carried */
static void ensure_state(LpModel *M) {
    if (!M->E || !M->rebuild) return;
    SxModel *E = sxm_new(&M->P, M->csc, M->rsc, M->msc, M->costsc);
    sxm_set_basis(E, M->cst, M->rst);
    sxm_free(M->E);
    M->E = E;
    M->rebuild = 0;
    free(M->cst); free(M->rst); free(M->csc); free(M->rsc);
    M->cst = M->rst = NULL; M->csc = M->rsc = NULL;
}
/* the first row change since the state was built: its statuses and scale factors taken, to follow the changes */
static void begin_rebuild(LpModel *M) {
    if (!M->E || M->rebuild) return;
    const int m = M->P.m, n = M->P.n;
    M->cst = xmalloc(sizeof(int) * ((size_t)n + 1)); M->rst = xmalloc(sizeof(int) * ((size_t)m + 1));
    sxm_get_basis(M->E, M->cst, M->rst);
    M->csc = xmalloc(sizeof(double) * ((size_t)n + 1)); M->rsc = xmalloc(sizeof(double) * ((size_t)m + 1));
    sxm_scaling(M->E, M->csc, M->rsc, &M->costsc);
    M->msc = m;
    M->rebuild = 1;
}
static void changed(LpModel *M) { M->have = 0; M->status = LM_NOTSOLVED; }

/* ---- changes */
int lm_set_col_bounds(LpModel *M, int j, double l, double u) {
    if (!M || j < 0 || j >= M->P.n) return -1;
    l = clampinf(l); u = clampinf(u);
    M->P.l[j] = l; M->P.u[j] = u;
    if (M->E && !M->rebuild) sxm_col_bounds(M->E, j, l, u);
    changed(M);
    return 0;
}
int lm_set_col_bounds_list(LpModel *M, int k, const int *cols, const double *l, const double *u) {
    if (!M || k < 0 || (k > 0 && (!cols || !l || !u))) return -1;
    for (int t = 0; t < k; t++) if (cols[t] < 0 || cols[t] >= M->P.n) return -1;
    for (int t = 0; t < k; t++) lm_set_col_bounds(M, cols[t], l[t], u[t]);
    return 0;
}
int lm_set_row_bounds(LpModel *M, int i, double rl, double ru) {
    if (!M || i < 0 || i >= M->P.m) return -1;
    rl = clampinf(rl); ru = clampinf(ru);
    M->P.rl[i] = rl; M->P.ru[i] = ru;
    if (M->E && !M->rebuild) sxm_row_bounds(M->E, i, rl, ru);
    changed(M);
    return 0;
}
int lm_set_obj(LpModel *M, int j, double c) {
    if (!M || j < 0 || j >= M->P.n) return -1;
    M->P.c[j] = c;
    if (M->E && !M->rebuild) sxm_cost(M->E, j, c);
    changed(M);
    return 0;
}
int lm_add_rows(LpModel *M, int k, const int *start, const int *cols, const double *val, const double *rl, const double *ru) {
    if (!M || M->trial || k < 0) return -1;
    if (k == 0) return 0;
    if (!start || !rl || !ru || (start[k] > start[0] && (!cols || !val))) return -1;
    LpProb *P = &M->P;
    const int m = P->m, n = P->n;
    for (int t = 0; t < k; t++) {
        if (start[t + 1] < start[t]) return -1;
        for (int q = start[t]; q < start[t + 1]; q++) if (cols[q] < 0 || cols[q] >= n) return -1;
    }
    begin_rebuild(M);
    /* the new matrix by columns: the old entries of each column, then the new rows' */
    int *cnt = xcalloc((size_t)n + 1, sizeof(int));
    for (int q = start[0]; q < start[k]; q++) if (val[q] != 0) cnt[cols[q]]++;
    const int nz0 = P->Ap[n], nz1 = nz0 + (start[k] - start[0]);
    int *Ap = xmalloc(sizeof(int) * ((size_t)n + 1)), *Ai = xmalloc(sizeof(int) * ((size_t)nz1 + 1));
    double *Ax = xmalloc(sizeof(double) * ((size_t)nz1 + 1));
    int e = 0;
    for (int j = 0; j < n; j++) {
        Ap[j] = e;
        for (int p = P->Ap[j]; p < P->Ap[j + 1]; p++) { Ai[e] = P->Ai[p]; Ax[e] = P->Ax[p]; e++; }
        e += cnt[j];
    }
    Ap[n] = e;
    int *fill = xmalloc(sizeof(int) * ((size_t)n + 1));
    for (int j = 0; j < n; j++) fill[j] = Ap[j] + (P->Ap[j + 1] - P->Ap[j]);
    for (int t = 0; t < k; t++) for (int q = start[t]; q < start[t + 1]; q++) if (val[q] != 0) { const int j = cols[q]; Ai[fill[j]] = m + t; Ax[fill[j]] = val[q]; fill[j]++; }
    free(fill); free(cnt);
    free(P->Ap); free(P->Ai); free(P->Ax);
    P->Ap = Ap; P->Ai = Ai; P->Ax = Ax;
    P->rl = realloc(P->rl, sizeof(double) * ((size_t)m + k + 1)); P->ru = realloc(P->ru, sizeof(double) * ((size_t)m + k + 1));
    for (int t = 0; t < k; t++) { P->rl[m + t] = clampinf(rl[t]); P->ru[m + t] = clampinf(ru[t]); }
    P->m = m + k;
    if (M->rebuild) {
        M->rst = realloc(M->rst, sizeof(int) * ((size_t)m + k + 1));
        for (int t = 0; t < k; t++) M->rst[m + t] = SX_BASIC;
    }
    changed(M);
    return 0;
}
int lm_del_rows(LpModel *M, int k, const int *rows) {
    if (!M || M->trial || k < 0) return -1;
    if (k == 0) return 0;
    LpProb *P = &M->P;
    const int m = P->m, n = P->n;
    char *del = xcalloc((size_t)m + 1, 1);
    for (int t = 0; t < k; t++) { if (!rows || rows[t] < 0 || rows[t] >= m || del[rows[t]]) { free(del); return -1; } del[rows[t]] = 1; }
    begin_rebuild(M);
    int *map = xmalloc(sizeof(int) * ((size_t)m + 1)), mm = 0;
    for (int i = 0; i < m; i++) map[i] = del[i] ? -1 : mm++;
    int e = 0;
    for (int j = 0; j < n; j++) {
        const int p0 = P->Ap[j], p1 = P->Ap[j + 1];
        P->Ap[j] = e;
        for (int p = p0; p < p1; p++) if (map[P->Ai[p]] >= 0) { P->Ai[e] = map[P->Ai[p]]; P->Ax[e] = P->Ax[p]; e++; }
    }
    P->Ap[n] = e;
    /* the kept rows' data moved down; the statuses and scale factors with them. A deleted row whose activity was
     * nonbasic leaves a basic variable too many: the last basic one in the statuses becomes nonbasic (lm_set_basis' rule) */
    int nbas_del = 0;
    if (M->rebuild) for (int i = 0; i < m; i++) if (del[i] && M->rst[i] != SX_BASIC) nbas_del++;
    int msc_new = 0;
    for (int i = 0; i < m; i++) {
        if (del[i]) continue;
        const int t = map[i];
        P->rl[t] = P->rl[i]; P->ru[t] = P->ru[i];
        if (M->rebuild) { M->rst[t] = M->rst[i]; if (i < M->msc) { M->rsc[t] = M->rsc[i]; msc_new = t + 1; } }
    }
    if (M->rebuild) M->msc = msc_new;
    P->m = mm;
    (void)nbas_del;     /* (sxm_set_basis drops the basic variables beyond the number of rows) */
    free(map); free(del);
    changed(M);
    return 0;
}

/* ---- solve */
static int map_status(int st, const volatile int *stop) {
    switch (st) {
    case 0: return LM_OPTIMAL;
    case 1: return LM_INFEASIBLE;
    case 2: return LM_UNBOUNDED;
    case 3: return LM_ITERLIMIT;
    case 6: return stop && *stop ? LM_STOPPED : LM_TIMELIMIT;
    case 7: return LM_CUTOFF;
    default: return LM_NUMERICAL;
    }
}
static void sxm_opts_of(const LpModel *M, SxmOpts *so, double t0, long itdone) {
    memset(so, 0, sizeof *so);
    so->maxit = M->opt.maxit > 0 ? (M->opt.maxit > itdone ? M->opt.maxit - itdone : 1) : 0;
    so->timelimit = M->opt.timelimit > 0 ? fmax(M->opt.timelimit - (now() - t0), 1e-6) : 0;
    so->stop = M->opt.stop;
    so->tol_primal = M->opt.tol_primal; so->tol_dual = M->opt.tol_dual;
    so->cutoff_on = M->opt.cutoff_on; so->cutoff = M->opt.cutoff - M->P.c0;
}
int lm_solve(LpModel *M, LmInfo *info) {
    if (!M) return -1;
    LmInfo dummy; if (!info) info = &dummy;
    memset(info, 0, sizeof *info);
    const double t0 = now();
    long it0 = 0, itd0 = 0, itp0 = 0; int nf0 = 0; double wk0 = 0;
    if (!M->E) {
        info->first = 1;
        if (M->opt.first_presolve && M->P.m > 0 && M->P.n > 0) {
            /* the first solve with everything, on the threads asked for (the setting is the calling thread's own) */
            SimplexOpts so; memset(&so, 0, sizeof so);
            so.method = 0; so.deterministic = 1;
            so.maxit = M->opt.maxit; so.timelimit = M->opt.timelimit; so.stop = M->opt.stop;
            so.tol_primal = M->opt.tol_primal; so.tol_dual = M->opt.tol_dual; so.verbose = M->opt.verbose;
            int *cs = xmalloc(sizeof(int) * ((size_t)M->P.n + 1)), *rs = xmalloc(sizeof(int) * ((size_t)M->P.m + 1));
            lm_get_basis(M, cs, rs);      /* (the basis of the row activities, kept when the presolve ends the solve) */
            SimplexInfo si; memset(&si, 0, sizeof si);
            pthread_mutex_lock(&g_first_mu);
#ifdef _OPENMP
            const int nt0 = omp_get_max_threads();
            omp_set_num_threads(M->opt.threads > 0 ? M->opt.threads : 1);
#endif
            const int st = simplex_solve(&M->P, &so, NULL, NULL, NULL, NULL, cs, rs, &si);
#ifdef _OPENMP
            omp_set_num_threads(nt0);
#endif
            pthread_mutex_unlock(&g_first_mu);
            it0 = si.iters; itd0 = si.iters_dual; itp0 = si.iters_primal; nf0 = si.nfactor; wk0 = si.work;
            M->E = sxm_new(&M->P, NULL, NULL, 0, 0);
            if (st != 3 && st != 6) sxm_set_basis(M->E, cs, rs);
            free(cs); free(rs);
            if (st == 3 || st == 6) {
                /* a limit: no basis to take; the state stays at the basis of the row activities */
                M->status = map_status(st, M->opt.stop); M->have = 0; M->obj = 0;
                info->status = M->status; info->iters = it0; info->iters_dual = itd0; info->iters_primal = itp0; info->nfactor = nf0; info->work = wk0;
                info->time = now() - t0;
                return M->status;
            }
        } else M->E = sxm_new(&M->P, NULL, NULL, 0, 0);
    }
    ensure_state(M);
    SxmOpts so; sxm_opts_of(M, &so, t0, it0);
    SxmRes r;
    const int st = sxm_solve(M->E, &so, &r);
    M->status = map_status(st, M->opt.stop);
    M->obj = M->P.c0 + sxm_objective(M->E);
    M->have = 1;
    info->status = M->status; info->obj = M->obj;
    info->iters = it0 + r.iters; info->iters_dual = itd0 + r.iters_dual; info->iters_primal = itp0 + r.iters_primal;
    info->nfactor = nf0 + r.nfactor;
    info->work = wk0 + r.work;      /* (22 October 2026, the MIP developer) the deterministic work, SxmRes.work (and the first solve's simplex method's) */
    info->time = now() - t0;
    return M->status;
}

/* ---- results */
int lm_status(const LpModel *M) { return M ? M->status : -1; }
double lm_obj(const LpModel *M) { return M && M->have ? M->obj : NAN; }
int lm_get_x(const LpModel *M, double *x) { if (!M || !M->have || !x) return -1; sxm_values(M->E, x, NULL, NULL); return 0; }
int lm_get_duals(const LpModel *M, double *y) { if (!M || !M->have || !y) return -1; sxm_values(M->E, NULL, y, NULL); return 0; }
int lm_get_redcosts(const LpModel *M, double *z) { if (!M || !M->have || !z) return -1; sxm_values(M->E, NULL, NULL, z); return 0; }
int lm_get_activity(const LpModel *M, double *r) {
    if (!M || !M->have || !r) return -1;
    const LpProb *P = &M->P;
    double *x = xmalloc(sizeof(double) * ((size_t)P->n + 1));
    sxm_values(M->E, x, NULL, NULL);
    for (int i = 0; i < P->m; i++) r[i] = 0;
    for (int j = 0; j < P->n; j++) if (x[j] != 0) for (int p = P->Ap[j]; p < P->Ap[j + 1]; p++) r[P->Ai[p]] += P->Ax[p] * x[j];
    free(x);
    return 0;
}
int lm_get_farkas(const LpModel *M, double *y) { if (!M || !M->have || !y || M->status != LM_INFEASIBLE) return -1; return sxm_farkas(M->E, y); }
int lm_get_ray(const LpModel *M, double *d) { if (!M || !M->have || !d || M->status != LM_UNBOUNDED) return -1; return sxm_ray(M->E, d); }

/* ---- basis */
int lm_get_basis(LpModel *M, int *cstat, int *rstat) {
    if (!M) return -1;
    if (!M->E) {        /* before the first solve: the basis of the row activities, the columns at a bound */
        const LpProb *P = &M->P;
        if (cstat) for (int j = 0; j < P->n; j++) cstat[j] = P->l[j] == P->u[j] ? SX_FIXED : P->l[j] > -LP_INF ? SX_LOWER : P->u[j] < LP_INF ? SX_UPPER : SX_ZERO;
        if (rstat) for (int i = 0; i < P->m; i++) rstat[i] = SX_BASIC;
        return 0;
    }
    if (M->rebuild) {   /* the statuses as they follow the row changes */
        if (cstat) memcpy(cstat, M->cst, sizeof(int) * (size_t)M->P.n);
        if (rstat) memcpy(rstat, M->rst, sizeof(int) * (size_t)M->P.m);
        return 0;
    }
    sxm_get_basis(M->E, cstat, rstat);
    return 0;
}
int lm_set_basis(LpModel *M, const int *cstat, const int *rstat) {
    if (!M || !cstat || !rstat) return -1;
    for (int j = 0; j < M->P.n; j++) if (cstat[j] < SX_BASIC || cstat[j] > SX_FIXED) return -1;
    for (int i = 0; i < M->P.m; i++) if (rstat[i] < SX_BASIC || rstat[i] > SX_FIXED) return -1;
    if (!M->E) M->E = sxm_new(&M->P, NULL, NULL, 0, 0);
    ensure_state(M);
    changed(M);
    return sxm_set_basis(M->E, cstat, rstat);
}

/* ---- the tableau */
static int tableau_ready(LpModel *M) {
    if (!M) return 0;
    if (!M->E) return 0;
    ensure_state(M);
    return 1;
}
int lm_get_head(LpModel *M, int *head) { if (!head || !tableau_ready(M)) return -1; sxm_head(M->E, head); return 0; }
int lm_binv_row(LpModel *M, int i, double *out) { if (!out || !tableau_ready(M) || i < 0 || i >= M->P.m) return -1; sxm_binv_row(M->E, i, out); return 0; }
int lm_binva_row(LpModel *M, int i, double *out) { if (!out || !tableau_ready(M) || i < 0 || i >= M->P.m) return -1; sxm_binva_row(M->E, i, out); return 0; }

/* ---- a trial of bound changes */
int lm_trial_begin(LpModel *M) {
    if (!M || M->trial || !M->E) return -1;
    ensure_state(M);
    const LpProb *P = &M->P;
    M->saved = sxm_save(M->E);
    M->sl = dupd(P->l, P->n); M->su = dupd(P->u, P->n); M->srl = dupd(P->rl, P->m); M->sru = dupd(P->ru, P->m); M->sc = dupd(P->c, P->n);
    M->trial_status = M->status; M->trial_have = M->have; M->trial_obj = M->obj;
    M->trial = 1;
    return 0;
}
int lm_trial_end(LpModel *M) {
    if (!M || !M->trial) return -1;
    LpProb *P = &M->P;
    memcpy(P->l, M->sl, sizeof(double) * (size_t)P->n); memcpy(P->u, M->su, sizeof(double) * (size_t)P->n);
    memcpy(P->rl, M->srl, sizeof(double) * (size_t)P->m); memcpy(P->ru, M->sru, sizeof(double) * (size_t)P->m);
    for (int j = 0; j < P->n; j++) if (P->c[j] != M->sc[j]) { P->c[j] = M->sc[j]; sxm_cost(M->E, j, P->c[j]); }
    sxm_restore(M->E, M->saved);
    M->status = M->trial_status; M->have = M->trial_have; M->obj = M->trial_obj;
    trial_free(M);
    return 0;
}
