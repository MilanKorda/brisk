/* qpdbl.c - doubleton equations of a QP substituted out before the interior-point method, and the solution mapped back
 * (the SOCP developer, 23 October 2026; IMPLEMENTATION 28.2).
 *
 * An equality row with two entries, a_ik x_k + a_ij x_j = b_i, gives x_k = alpha + beta x_j (alpha = b_i / a_ik,
 * beta = -a_ij / a_ik): column k and row i leave the problem (Andersen and Andersen, "Presolving in linear programming",
 * 1995). The column to eliminate is one whose H column has no entry off the diagonal, so that H keeps its pattern:
 * h_jj += h_kk beta^2, c_j += beta (c_k + h_kk alpha), c0 += h_kk alpha^2 / 2 + c_k alpha; each other row r of column k
 * gets beta a_rk in column j and its bounds shifted by a_rk alpha. Only a column whose bounds are implied by its
 * partner's through the row is eliminated (x_k implied free), so that no bound moves. One pass takes a set of such rows in which no column is both
 * eliminated and kept as another's partner; the passes repeat while they find rows (substitutions make new ones).
 *
 * Postsolve, the passes in reverse: x_k = alpha + beta x_j, z_k = 0 (x_j keeps the bound multiplier); the row's
 * multiplier from column k's dual equation as it was at its pass, a_ik y_i = h_kk x_k + c_k - sum_{r != i} a_rk y_r;
 * column j's dual equation then holds by
 * construction (its terms are those of the reduced problem). */
#include "qp.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

static int fin_(double v) { return v > -QP_INF && v < QP_INF; }

typedef struct {
    int i, k, j;                  /* original indices: the row, the eliminated column, its partner */
    double alpha, beta, aik, hkk, ck;
    long e0, e1;                  /* column k's other entries at the pass (rows original, values) in the arrays below */
} DblStep;

struct QpDbl {
    int n0, m0, n, m;             /* the original sizes and the reduced */
    int *col0, *row0;             /* the reduced problem's columns and rows, as original indices */
    DblStep *st; int ns, cap;
    int *er; double *ev; long ne, ecap;
};
typedef struct QpDbl QpDbl;

void qpdbl_free(QpDbl *D) { if (!D) return; free(D->col0); free(D->row0); free(D->st); free(D->er); free(D->ev); free(D); }

/* the working problem: CSC A, H's lower triangle, with the original indices of its rows and columns */
typedef struct { int n, m; int *Ap, *Ai, *Hp, *Hi; double *Ax, *Hx, *c, *rl, *ru, *xl, *xu, c0; int *col0, *row0; } W;
static void w_free(W *w) { free(w->Ap); free(w->Ai); free(w->Ax); free(w->Hp); free(w->Hi); free(w->Hx); free(w->c); free(w->rl); free(w->ru); free(w->xl); free(w->xu); free(w->col0); free(w->row0); }

static int push_step(QpDbl *D, const DblStep *s) {
    if (D->ns == D->cap) { const int c2 = D->cap ? 2 * D->cap : 256; DblStep *t = realloc(D->st, sizeof(DblStep) * (size_t)c2); if (!t) return -1; D->st = t; D->cap = c2; }
    D->st[D->ns++] = *s; return 0;
}
static int push_entry(QpDbl *D, int r, double v) {
    if (D->ne == D->ecap) { const long c2 = D->ecap ? 2 * D->ecap : 4096; int *a = realloc(D->er, sizeof(int) * (size_t)c2); double *b = realloc(D->ev, sizeof(double) * (size_t)c2); if (!a || !b) { if (a) D->er = a; if (b) D->ev = b; return -1; } D->er = a; D->ev = b; D->ecap = c2; }
    D->er[D->ne] = r; D->ev[D->ne] = v; D->ne++; return 0;
}

/* one pass: returns the number of rows eliminated (0: none), -1 on a failure (out of memory) */
static int pass(W *w, QpDbl *D) {
    const int n = w->n, m = w->m;
    int *rc = calloc((size_t)m + 1, sizeof(int)), *rp1 = malloc(sizeof(int) * ((size_t)m + 1)), *rp2 = malloc(sizeof(int) * ((size_t)m + 1));
    double *rv1 = malloc(sizeof(double) * ((size_t)m + 1)), *rv2 = malloc(sizeof(double) * ((size_t)m + 1));
    int *hoff = calloc((size_t)n + 1, sizeof(int)), *ccnt = calloc((size_t)n + 1, sizeof(int));
    double *hd = calloc((size_t)n + 1, sizeof(double));
    signed char *mark = calloc((size_t)n + 1, 1);       /* 1: eliminated this pass, 2: a partner this pass */
    int *elim_row = malloc(sizeof(int) * ((size_t)n + 1)), *partner = malloc(sizeof(int) * ((size_t)n + 1));
    char *rdead = calloc((size_t)m + 1, 1);
    if (!rc || !rp1 || !rp2 || !rv1 || !rv2 || !hoff || !ccnt || !hd || !mark || !elim_row || !partner || !rdead) { free(rc); free(rp1); free(rp2); free(rv1); free(rv2); free(hoff); free(ccnt); free(hd); free(mark); free(elim_row); free(partner); free(rdead); return -1; }
    for (int j = 0; j < n; j++) for (int p = w->Ap[j]; p < w->Ap[j + 1]; p++) if (w->Ax[p] != 0) {
        const int i = w->Ai[p]; if (rc[i] == 0) { rp1[i] = j; rv1[i] = w->Ax[p]; } else if (rc[i] == 1) { rp2[i] = j; rv2[i] = w->Ax[p]; }
        rc[i]++; ccnt[j]++;
    }
    if (w->Hp) for (int j = 0; j < n; j++) for (int p = w->Hp[j]; p < w->Hp[j + 1]; p++) {
        const int i = w->Hi[p]; if (w->Hx[p] == 0) continue;
        if (i == j) hd[j] += w->Hx[p]; else { hoff[i]++; hoff[j]++; }
    }
    int ne = 0;
    double *alpha = malloc(sizeof(double) * ((size_t)n + 1)), *beta = malloc(sizeof(double) * ((size_t)n + 1));
    if (!alpha || !beta) { ne = -1; goto done; }
    for (int i = 0; i < m; i++) {
        if (rc[i] != 2 || !fin_(w->rl[i]) || w->rl[i] != w->ru[i]) continue;
        int k = -1, j = -1; double ak = 0;
        const int p = rp1[i], q = rp2[i]; const double ap = rv1[i], aq = rv2[i];
        if (mark[p] || mark[q]) continue;
        /* the eliminated column: H diagonal there, a pivot of at least a tenth of the row's largest entry, its bounds
         * implied by the partner's through the row (x_k implied free: moving a bound of k onto j cost iterations, QSHIP12S,
         * QP: 13 -> 20, the 138 Maros-Meszaros QPs 16 more and 8 fewer, the cause not found); the shorter column first */
        double al = 0, be = 0;
        for (int o = 0; o < 2 && k < 0; o++) {
            const int first = ccnt[p] <= ccnt[q] ? o == 0 : o == 1;
            const int kk = first ? p : q, jj = first ? q : p; const double ak_ = first ? ap : aq, aj_ = first ? aq : ap;
            if (hoff[kk] != 0 || fabs(ak_) < 0.1 * fmax(fabs(ap), fabs(aq))) continue;
            const double a1 = w->rl[i] / ak_, b1 = -aj_ / ak_;
            double tl = -QP_INF, tu = QP_INF;      /* k's bounds on x_j */
            if (b1 > 0) { if (fin_(w->xl[kk])) tl = (w->xl[kk] - a1) / b1; if (fin_(w->xu[kk])) tu = (w->xu[kk] - a1) / b1; }
            else { if (fin_(w->xu[kk])) tl = (w->xu[kk] - a1) / b1; if (fin_(w->xl[kk])) tu = (w->xl[kk] - a1) / b1; }
            if (tl > w->xl[jj] || tu < w->xu[jj]) continue;
            k = kk; j = jj; ak = ak_; al = a1; be = b1;
        }
        if (k < 0) continue;
        if (fin_(w->xl[j]) && fin_(w->xu[j]) && w->xu[j] - w->xl[j] <= 1e-12 * (1 + fabs(w->xl[j]))) continue;   /* (j fixed: left to the reduction) */
        DblStep s; memset(&s, 0, sizeof s);
        s.i = w->row0[i]; s.k = w->col0[k]; s.j = w->col0[j]; s.alpha = al; s.beta = be; s.aik = ak; s.hkk = hd[k]; s.ck = w->c[k];
        s.e0 = D->ne;
        for (int p2 = w->Ap[k]; p2 < w->Ap[k + 1]; p2++) if (w->Ai[p2] != i && w->Ax[p2] != 0) if (push_entry(D, w->row0[w->Ai[p2]], w->Ax[p2])) { ne = -1; goto done; }
        s.e1 = D->ne;
        if (push_step(D, &s)) { ne = -1; goto done; }
        /* the problem's data that do not depend on the new matrix: cost, the rows' shifts */
        w->c[j] += be * (w->c[k] + hd[k] * al); w->c0 += 0.5 * hd[k] * al * al + w->c[k] * al;
        for (int p2 = w->Ap[k]; p2 < w->Ap[k + 1]; p2++) { const int r = w->Ai[p2]; if (r == i) continue; const double sh = w->Ax[p2] * al; if (fin_(w->rl[r])) w->rl[r] -= sh; if (fin_(w->ru[r])) w->ru[r] -= sh; }
        hd[j] += hd[k] * be * be;
        mark[k] = 1; mark[j] = 2; elim_row[k] = i; partner[k] = j; alpha[k] = al; beta[k] = be; rdead[i] = 1; ne++;
    }
    if (ne > 0) {
        /* the new A: each kept column's entries, a partner's merged with beta times its eliminated column's (row i left out) */
        int *nmap = malloc(sizeof(int) * ((size_t)n + 1)), *mmap = malloc(sizeof(int) * ((size_t)m + 1));
        int *pk = malloc(sizeof(int) * ((size_t)n + 1));           /* the eliminated column whose partner is j (-1: none) */
        double *acc = calloc((size_t)m + 1, sizeof(double)), *accm = calloc((size_t)m + 1, sizeof(double)); int *mk = malloc(sizeof(int) * ((size_t)m + 1)), *lst = malloc(sizeof(int) * ((size_t)m + 1));
        if (!nmap || !mmap || !pk || !acc || !accm || !mk || !lst) { free(nmap); free(mmap); free(pk); free(acc); free(accm); free(mk); free(lst); ne = -1; goto done; }
        int n2 = 0, m2 = 0;
        for (int j = 0; j < n; j++) { nmap[j] = mark[j] == 1 ? -1 : n2++; pk[j] = -1; }
        for (int k = 0; k < n; k++) if (mark[k] == 1) pk[partner[k]] = k;
        for (int i = 0; i < m; i++) { mmap[i] = rdead[i] ? -1 : m2++; mk[i] = -1; }
        long nz = 0; for (int j = 0; j < n; j++) nz += w->Ap[j + 1] - w->Ap[j];
        int *Ap = malloc(sizeof(int) * ((size_t)n2 + 1)), *Ai = malloc(sizeof(int) * ((size_t)nz + 1)); double *Ax = malloc(sizeof(double) * ((size_t)nz + 1));
        long t = 0; int c2 = 0;
        for (int j = 0; j < n; j++) {
            if (mark[j] == 1) continue;
            Ap[c2] = (int)t; int nl = 0;
            for (int p = w->Ap[j]; p < w->Ap[j + 1]; p++) { const int r = w->Ai[p]; if (rdead[r]) continue; if (mk[r] != j) { mk[r] = j; acc[r] = 0; accm[r] = 0; lst[nl++] = r; } acc[r] += w->Ax[p]; accm[r] += fabs(w->Ax[p]); }
            if (pk[j] >= 0) { const int k = pk[j]; for (int p = w->Ap[k]; p < w->Ap[k + 1]; p++) { const int r = w->Ai[p]; if (rdead[r]) continue; if (mk[r] != j) { mk[r] = j; acc[r] = 0; accm[r] = 0; lst[nl++] = r; } acc[r] += beta[k] * w->Ax[p]; accm[r] += fabs(beta[k] * w->Ax[p]); } }
            /* (rows in increasing order, as the CSC convention of the readers) */
            for (int a = 1; a < nl; a++) { const int v = lst[a]; int b = a - 1; while (b >= 0 && lst[b] > v) { lst[b + 1] = lst[b]; b--; } lst[b + 1] = v; }
            /* (23 October 2026, the night) an entry that cancels to the rounding of its terms is a zero: a_rj + beta a_rk with
             * a_rj = -beta a_rk exactly in the data leaves 1e-17 of them, kept it became a row of tiny entries that the
             * scaling multiplied by 9.8e16 (DCOPF_300_ieee, QP, a DC power flow of PGLib's 300-bus grid: two lines in series
             * through a bus without load; the iteration limit in 200 iterations, then the solve without the substitutions) */
            for (int a = 0; a < nl; a++) { const int r = lst[a]; if (acc[r] != 0 && fabs(acc[r]) > 1e-12 * accm[r]) { Ai[t] = mmap[r]; Ax[t] = acc[r]; t++; } }
            c2++;
        }
        Ap[n2] = (int)t;
        /* the new H: the kept columns' entries, the partners' diagonals set from hd */
        int *Hp = NULL, *Hi = NULL; double *Hx = NULL;
        {   long hz = 0; if (w->Hp) hz = w->Hp[n];
            Hp = malloc(sizeof(int) * ((size_t)n2 + 1)); Hi = malloc(sizeof(int) * ((size_t)hz + n2 + 1)); Hx = malloc(sizeof(double) * ((size_t)hz + n2 + 1));
            long h = 0; int cc = 0;
            for (int j = 0; j < n; j++) {
                if (mark[j] == 1) continue;
                Hp[cc] = (int)h; int had = 0;
                if (w->Hp) for (int p = w->Hp[j]; p < w->Hp[j + 1]; p++) {
                    const int i = w->Hi[p]; if (mark[i] == 1) continue;          /* (an eliminated column's H is diagonal only: none here) */
                    if (i == j) { if (!had && hd[j] != 0) { Hi[h] = cc; Hx[h] = hd[j]; h++; } had = 1; }
                    else { Hi[h] = nmap[i]; Hx[h] = w->Hx[p]; h++; }
                }
                if (!had && hd[j] != 0) {      /* (a new diagonal entry: placed first, the diagonal is the column's first row in a lower triangle) */
                    memmove(Hi + Hp[cc] + 1, Hi + Hp[cc], sizeof(int) * (size_t)(h - Hp[cc])); memmove(Hx + Hp[cc] + 1, Hx + Hp[cc], sizeof(double) * (size_t)(h - Hp[cc]));
                    Hi[Hp[cc]] = cc; Hx[Hp[cc]] = hd[j]; h++;
                }
                cc++;
            }
            Hp[n2] = (int)h;
        }
        double *c = malloc(sizeof(double) * ((size_t)n2 + 1)), *xl = malloc(sizeof(double) * ((size_t)n2 + 1)), *xu = malloc(sizeof(double) * ((size_t)n2 + 1));
        double *rl = malloc(sizeof(double) * ((size_t)m2 + 1)), *ru = malloc(sizeof(double) * ((size_t)m2 + 1));
        int *col0 = malloc(sizeof(int) * ((size_t)n2 + 1)), *row0 = malloc(sizeof(int) * ((size_t)m2 + 1));
        for (int j = 0; j < n; j++) if (nmap[j] >= 0) { const int a = nmap[j]; c[a] = w->c[j]; xl[a] = w->xl[j]; xu[a] = w->xu[j]; col0[a] = w->col0[j]; }
        for (int i = 0; i < m; i++) if (mmap[i] >= 0) { const int a = mmap[i]; rl[a] = w->rl[i]; ru[a] = w->ru[i]; row0[a] = w->row0[i]; }
        const double c0 = w->c0;
        w_free(w);
        w->n = n2; w->m = m2; w->Ap = Ap; w->Ai = Ai; w->Ax = Ax; w->Hp = Hp; w->Hi = Hi; w->Hx = Hx; w->c = c; w->xl = xl; w->xu = xu; w->rl = rl; w->ru = ru; w->col0 = col0; w->row0 = row0; w->c0 = c0;
        free(nmap); free(mmap); free(pk); free(acc); free(accm); free(mk); free(lst);
    }
done:
    free(rc); free(rp1); free(rp2); free(rv1); free(rv2); free(hoff); free(ccnt); free(hd); free(mark); free(elim_row); free(partner); free(rdead); free(alpha); free(beta);
    return ne;
}

/* the reduced problem P2 (its own arrays; qp_prob_free frees them) and the record for the postsolve; NULL when no row was
 * eliminated (or on a failure) */
QpDbl *qpdbl_presolve(const QpProb *P, QpProb *P2, int *nelim) {
    *nelim = 0;
    if (P->nqc > 0 || P->m <= 0 || P->n <= 0) return NULL;
    W w; memset(&w, 0, sizeof w);
    const int n = P->n, m = P->m; const long nz = P->Ap[n], hz = P->Hp ? P->Hp[n] : 0;
    w.n = n; w.m = m; w.c0 = P->c0;
    w.Ap = malloc(sizeof(int) * ((size_t)n + 1)); w.Ai = malloc(sizeof(int) * ((size_t)nz + 1)); w.Ax = malloc(sizeof(double) * ((size_t)nz + 1));
    w.Hp = malloc(sizeof(int) * ((size_t)n + 1)); w.Hi = malloc(sizeof(int) * ((size_t)hz + 1)); w.Hx = malloc(sizeof(double) * ((size_t)hz + 1));
    w.c = malloc(sizeof(double) * ((size_t)n + 1)); w.xl = malloc(sizeof(double) * ((size_t)n + 1)); w.xu = malloc(sizeof(double) * ((size_t)n + 1));
    w.rl = malloc(sizeof(double) * ((size_t)m + 1)); w.ru = malloc(sizeof(double) * ((size_t)m + 1));
    w.col0 = malloc(sizeof(int) * ((size_t)n + 1)); w.row0 = malloc(sizeof(int) * ((size_t)m + 1));
    QpDbl *D = calloc(1, sizeof *D);
    if (!w.Ap || !w.Ai || !w.Ax || !w.Hp || !w.Hi || !w.Hx || !w.c || !w.xl || !w.xu || !w.rl || !w.ru || !w.col0 || !w.row0 || !D) { w_free(&w); free(D); return NULL; }
    memcpy(w.Ap, P->Ap, sizeof(int) * ((size_t)n + 1)); memcpy(w.Ai, P->Ai, sizeof(int) * (size_t)nz); memcpy(w.Ax, P->Ax, sizeof(double) * (size_t)nz);
    if (P->Hp) { memcpy(w.Hp, P->Hp, sizeof(int) * ((size_t)n + 1)); memcpy(w.Hi, P->Hi, sizeof(int) * (size_t)hz); memcpy(w.Hx, P->Hx, sizeof(double) * (size_t)hz); }
    else for (int j = 0; j <= n; j++) w.Hp[j] = 0;
    memcpy(w.c, P->c, sizeof(double) * (size_t)n); memcpy(w.xl, P->xl, sizeof(double) * (size_t)n); memcpy(w.xu, P->xu, sizeof(double) * (size_t)n);
    memcpy(w.rl, P->rl, sizeof(double) * (size_t)m); memcpy(w.ru, P->ru, sizeof(double) * (size_t)m);
    for (int j = 0; j < n; j++) w.col0[j] = j;
    for (int i = 0; i < m; i++) w.row0[i] = i;
    D->n0 = n; D->m0 = m;
    for (int ps = 0; ps < 20; ps++) {
        const int e = pass(&w, D);
        if (e < 0) { w_free(&w); qpdbl_free(D); return NULL; }
        if (e == 0) break;
        *nelim += e;
    }
    if (*nelim == 0) { w_free(&w); qpdbl_free(D); return NULL; }
    memset(P2, 0, sizeof *P2);
    P2->n = w.n; P2->m = w.m; P2->Ap = w.Ap; P2->Ai = w.Ai; P2->Ax = w.Ax; P2->Hp = w.Hp; P2->Hi = w.Hi; P2->Hx = w.Hx;
    P2->c = w.c; P2->c0 = w.c0; P2->rl = w.rl; P2->ru = w.ru; P2->xl = w.xl; P2->xu = w.xu; P2->maxim = P->maxim;
    memcpy(P2->name, P->name, sizeof P2->name);
    D->n = w.n; D->m = w.m; D->col0 = w.col0; D->row0 = w.row0;
    return D;
}

/* the reduced problem's solution (x2, y2, z2) mapped to the original problem's (x, y, z: n0, m0, n0) */
void qpdbl_postsolve(const QpDbl *D, const QpProb *P, const double *x2, const double *y2, const double *z2, double *x, double *y, double *z) {
    (void)P;
    for (int j = 0; j < D->n0; j++) { x[j] = 0; z[j] = 0; }
    for (int i = 0; i < D->m0; i++) y[i] = 0;
    for (int a = 0; a < D->n; a++) { x[D->col0[a]] = x2[a]; z[D->col0[a]] = z2[a]; }
    for (int a = 0; a < D->m; a++) y[D->row0[a]] = y2[a];
    for (int t = D->ns - 1; t >= 0; t--) {
        const DblStep *s = &D->st[t];
        x[s->k] = s->alpha + s->beta * x[s->j];
        z[s->k] = 0;                  /* (k's bounds are implied by j's: their multiplier is j's) */
        double r = s->hkk * x[s->k] + s->ck;
        for (long e = s->e0; e < s->e1; e++) r -= D->ev[e] * y[D->er[e]];
        y[s->i] = r / s->aik;
    }
}
