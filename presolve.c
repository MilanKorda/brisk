/*
 * Facial-reduction presolve.
 *
 * If <A_i, X> = 0 and A_i is semidefinite, every feasible X satisfies A_i X = 0,
 * so the problem has no strictly feasible point (Slater fails) and the dual
 * optimal set is unbounded. Interior-point methods then lose accuracy.
 *
 * Handled cases (per block part of constraint i, all parts of one sign):
 *   - SDP part  s*a*a'  (rank one): substitute X = V W V' with the sparse basis
 *     v_k = e_k - (a_k/a_p) e_p of a-perp; block size shrinks by one.
 *   - SDP part  same-sign diagonal: the touched rows/columns of X vanish.
 *   - LP part   same-sign coefficients: the touched variables are fixed to 0.
 * The constraint is then removed. Empty constraints with b_i = 0 are removed too.
 * Objective values are unchanged: <C, V W V'> = <V'CV, W>.
 */
#include "brisk.h"
int g_fr_maxdepth = -1;   /* -frdepth (main.c sets it per solve; -1: no limit) */
int g_fr_depth_used = 0;  /* the depth of the facial reduction made in this run (for the advice; main.c resets it) */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#ifdef _OPENMP
#include <omp.h>
#endif

typedef struct { int r, c; double v; } Ent;

static int cmp_ent(const void *a, const void *b) {
    const Ent *x = a, *y = b;
    if (x->c != y->c) return x->c - y->c;
    return x->r - y->r;
}

static void sp_release(SpSym *S) {
    free(S->row); free(S->col); free(S->val);
    free(S->fr); free(S->fc); free(S->fv); free(S->rows);
    memset(S, 0, sizeof(*S));
}

/* rebuild S (upper storage) from an entry list; merges duplicates, drops zeros */
static void sp_from_entries(SpSym *S, Ent *E, size_t ne, int n, int is_lp) {
    int sorted = 1;
    for (size_t k = 0; k < ne; k++) {
        if (E[k].r > E[k].c) { int t = E[k].r; E[k].r = E[k].c; E[k].c = t; }
        if (k > 0 && sorted && cmp_ent(&E[k - 1], &E[k]) >= 0) sorted = 0;   /* (strictly increasing: duplicates keep qsort's order) */
    }
    if (!sorted) qsort(E, ne, sizeof(Ent), cmp_ent);   /* (a compression writes its entries in order: 13,000 a matrix) */
    size_t w = 0;
    for (size_t k = 0; k < ne; k++) {
        if (w > 0 && E[w-1].r == E[k].r && E[w-1].c == E[k].c) E[w-1].v += E[k].v;
        else E[w++] = E[k];
    }
    size_t z = 0;
    for (size_t k = 0; k < w; k++) if (E[k].v != 0.0) E[z++] = E[k];
    sp_release(S);
    S->nnz = (int)z;
    S->row = malloc(sizeof(int) * (z ? z : 1));
    S->col = malloc(sizeof(int) * (z ? z : 1));
    S->val = malloc(sizeof(double) * (z ? z : 1));
    for (size_t k = 0; k < z; k++) { S->row[k] = E[k].r; S->col[k] = E[k].c; S->val[k] = E[k].v; }
    /* full expansion + row set (mirrors problem.c) */
    S->fr = malloc(sizeof(int) * (2 * z + 1));
    S->fc = malloc(sizeof(int) * (2 * z + 1));
    S->fv = malloc(sizeof(double) * (2 * z + 1));
    int nf = 0;
    for (size_t k = 0; k < z; k++) {
        S->fr[nf] = S->row[k]; S->fc[nf] = S->col[k]; S->fv[nf] = S->val[k]; nf++;
        if (!is_lp && S->row[k] != S->col[k]) {
            S->fr[nf] = S->col[k]; S->fc[nf] = S->row[k]; S->fv[nf] = S->val[k]; nf++;
        }
    }
    S->ef = nf;
    char *mark = calloc(n > 0 ? n : 1, 1);
    S->nr = 0;
    for (int k = 0; k < nf; k++) if (!mark[S->fr[k]]) { mark[S->fr[k]] = 1; S->nr++; }
    S->rows = malloc(sizeof(int) * (S->nr ? S->nr : 1));
    for (int k = 0, c = 0; k < n; k++) if (mark[k]) S->rows[c++] = k;
    free(mark);
}

/* S == sgn * a a' ?  (a dense, length n) */
static int rank1_test(const SpSym *S, int n, double *a, double *sgn) {
    int p = -1;
    double best = 0, amax = 0;
    for (int k = 0; k < S->nnz; k++) {
        amax = fmax(amax, fabs(S->val[k]));
        if (S->row[k] == S->col[k] && fabs(S->val[k]) > best) { best = fabs(S->val[k]); p = S->row[k]; }
    }
    if (p < 0) return 0;
    double spp = 0;
    memset(a, 0, sizeof(double) * n);
    for (int k = 0; k < S->nnz; k++) if (S->row[k] == p && S->col[k] == p) spp = S->val[k];
    *sgn = spp > 0 ? 1.0 : -1.0;
    double ap = sqrt(fabs(spp));
    a[p] = ap;
    for (int k = 0; k < S->nnz; k++) {
        if (S->row[k] == S->col[k]) continue;
        if (S->row[k] == p) a[S->col[k]] = S->val[k] / (*sgn * ap);
        else if (S->col[k] == p) a[S->row[k]] = S->val[k] / (*sgn * ap);
    }
    long supp = 0;
    for (int k = 0; k < n; k++) supp += (a[k] != 0);
    if ((long)S->nnz != supp * (supp + 1) / 2) return 0;
    /* every stored entry must lie in the support of a (together with the count this
     * makes the pattern exactly supp x supp) */
    for (int k = 0; k < S->nnz; k++)
        if (a[S->row[k]] == 0 || a[S->col[k]] == 0) return 0;
    double tol = 1e-12 * amax;
    for (int k = 0; k < S->nnz; k++) {
        double v = *sgn * a[S->row[k]] * a[S->col[k]];
        if (fabs(v - S->val[k]) > tol) return 0;
    }
    return 1;
}

static int samesign_diag(const SpSym *S, double *sgn) {
    if (S->nnz == 0) return 0;
    *sgn = S->val[0] > 0 ? 1.0 : -1.0;
    for (int k = 0; k < S->nnz; k++)
        if (S->row[k] != S->col[k] || S->val[k] * *sgn <= 0) return 0;
    return 1;
}

/* delete index set (del[r] != 0) from every matrix of block B */
static Problem *g_ps_P = NULL;    /* problem being presolved (postsolve hooks) */
static void block_delete_indices(Block *B, const char *del) {
    if (g_ps_P) ps_hook_delete(g_ps_P, B, del);
    int n = B->n;
    int *map = malloc(sizeof(int) * n);
    int nn = 0;
    for (int r = 0; r < n; r++) map[r] = del[r] ? -1 : nn++;
    int is_lp = B->type == BLK_LP;
    for (int t = -1; t < B->ncon; t++) {
        SpSym *S = t < 0 ? &B->C : &B->A[t];
        int touched = 0;
        for (int k = 0; k < S->nnz && !touched; k++) if (map[S->row[k]] < 0 || map[S->col[k]] < 0) touched = 1;
        if (!touched) {
            /* 4.22: the map is monotone, so an untouched matrix keeps its order: renumber in
             * place (the rebuild cleared an n-byte marker per matrix: 80 s of facial
             * reduction on case6468_rte, 60,000 constraints x 57 reductions) */
            for (int k = 0; k < S->nnz; k++) { S->row[k] = map[S->row[k]]; S->col[k] = map[S->col[k]]; }
            for (int k = 0; k < S->ef; k++) { S->fr[k] = map[S->fr[k]]; S->fc[k] = map[S->fc[k]]; }
            for (int k = 0; k < S->nr; k++) S->rows[k] = map[S->rows[k]];
            continue;
        }
        Ent *E = malloc(sizeof(Ent) * (S->nnz + 1));
        size_t ne = 0;
        for (int k = 0; k < S->nnz; k++) {
            int r = map[S->row[k]], c = map[S->col[k]];
            if (r >= 0 && c >= 0) { E[ne].r = r; E[ne].c = c; E[ne].v = S->val[k]; ne++; }
        }
        sp_from_entries(S, E, ne, nn, is_lp);
        free(E);
    }
    B->n = nn;
    free(map);
}

/* X = V W V', V = [e_k + w_k e_p]_{k != p};  M' = V' M V for all matrices */
static void block_rank1_reduce(Block *B, const double *a, int p) {
    const int n = B->n;
    double *w = calloc(n, sizeof(double));
    int *ws = malloc(sizeof(int) * n), nw = 0;
    for (int r = 0; r < n; r++)
        if (r != p && a[r] != 0) { w[r] = -a[r] / a[p]; ws[nw++] = r; }
    if (g_ps_P) ps_hook_rank1(g_ps_P, B, w, p);
    int *map = malloc(sizeof(int) * n);
    for (int r = 0, c = 0; r < n; r++) map[r] = (r == p) ? -1 : c++;
    double *mp = calloc(n, sizeof(double));
    int *ms = malloc(sizeof(int) * n);
    for (int t = -1; t < B->ncon; t++) {
        SpSym *S = t < 0 ? &B->C : &B->A[t];
        {   /* 2 November 2026: a matrix that does not touch p is renumbered in place (the map is monotone: the same
             * order, no duplicates, no zeros; as block_delete_indices does); the sequence of ISSUES 745 reduces
             * tiger_texture's block 600 times, each over its 1,803 matrices */
            int touches = 0;
            for (int k = 0; k < S->nnz && !touches; k++) if (S->row[k] == p || S->col[k] == p) touches = 1;
            if (!touches) {
                for (int k = 0; k < S->nnz; k++) { S->row[k] = map[S->row[k]]; S->col[k] = map[S->col[k]]; }
                for (int k = 0; k < S->ef; k++) { S->fr[k] = map[S->fr[k]]; S->fc[k] = map[S->fc[k]]; }
                for (int k = 0; k < S->nr; k++) S->rows[k] = map[S->rows[k]];
                continue;
            }
        }
        int nm = 0;
        double mpp = 0;
        for (int k = 0; k < S->nnz; k++) {
            int r = S->row[k], c = S->col[k];
            if (r == p && c == p) mpp = S->val[k];
            else if (r == p) { mp[c] = S->val[k]; ms[nm++] = c; }
            else if (c == p) { mp[r] = S->val[k]; ms[nm++] = r; }
        }
        size_t cap = (size_t)S->nnz + (size_t)nw * nm * 2 + (mpp != 0 ? (size_t)nw * (nw + 1) / 2 : 0) + 1;
        Ent *E = malloc(sizeof(Ent) * cap);
        size_t ne = 0;
        for (int k = 0; k < S->nnz; k++) {
            int r = S->row[k], c = S->col[k];
            if (r == p || c == p) continue;
            E[ne].r = map[r]; E[ne].c = map[c]; E[ne].v = S->val[k]; ne++;
        }
        /* w*mp' + mp*w' : off-diagonal pairs appear once per ordered term, diagonal twice */
        for (int x = 0; x < nw; x++) {
            int r = ws[x];
            for (int y = 0; y < nm; y++) {
                int c = ms[y];
                double v = w[r] * mp[c];
                E[ne].r = map[r]; E[ne].c = map[c]; E[ne].v = (r == c) ? 2 * v : v; ne++;
            }
        }
        if (mpp != 0)
            for (int x = 0; x < nw; x++)
                for (int y = x; y < nw; y++) {
                    E[ne].r = map[ws[x]]; E[ne].c = map[ws[y]]; E[ne].v = mpp * w[ws[x]] * w[ws[y]]; ne++;
                }
        sp_from_entries(S, E, ne, n - 1, 0);
        free(E);
        for (int y = 0; y < nm; y++) mp[ms[y]] = 0;
    }
    B->n = n - 1;
    free(w); free(ws); free(map); free(mp); free(ms);
}


/* ---------------------------------------------------------------------- */
/* Higher-rank case: the part of a constraint on its support S is +-PSD with
 * rank r. N (s x k, k = s - r, orthonormal) spans its null space. Substitute
 * X = V W V' with V = identity on the rows outside S and N on S. The block
 * becomes: rows outside S (renumbered, same order), then k new coordinates. */
/* O(nnz) necessary conditions for +-semidefiniteness: one-signed diagonal, a zero
 * diagonal entry implies a zero row, and |a_ij|^2 <= a_ii a_jj.                 */
static int semidef_quick(const SpSym *A, int n) {
    double *dg = calloc(n, sizeof(double));
    int pos = 0, neg = 0, ok = 1;
    for (int q = 0; q < A->nnz; q++)
        if (A->row[q] == A->col[q]) { dg[A->row[q]] = A->val[q]; if (A->val[q] > 0) pos = 1; else neg = 1; }
    if (pos && neg) ok = 0;
    for (int q = 0; q < A->nnz && ok; q++) {
        int r = A->row[q], c = A->col[q];
        if (r == c) continue;
        double v = A->val[q];
        if (dg[r] == 0 || dg[c] == 0 || v * v > dg[r] * dg[c] * (1 + 1e-12)) ok = 0;
    }
    free(dg);
    return ok;
}

/* zrel: an eigenvalue within zrel of the largest is zero. gapf > 0 (the faces of clique_faces, a numerical test): the
 * smallest nonzero eigenvalue must exceed gapf times the largest zero one (and 1e-16 of the largest), so that the rank
 * is read across a clear gap (swissroll: zeros up to 1.2e-15, the smallest nonzero 1.3e-11 of the largest) */
static int psd_nullspace_tol(const SpSym *A, int n, int smax, double zrel, double gapf, double *sgn, int **Sout,
                             int *sout, double **Nout, int *kout, double **Uout) {
    int s = A->nr;
    if (s == 0 || s > smax) return 0;
    if (!semidef_quick(A, n)) return 0;
    const int *S = A->rows;
    int *pos = malloc(sizeof(int) * n);
    for (int i = 0; i < n; i++) pos[i] = -1;
    for (int i = 0; i < s; i++) pos[S[i]] = i;
    double *T = calloc((size_t)s * s, sizeof(double));
    double amax = 0;
    for (int q = 0; q < A->ef; q++) {
        T[pos[A->fr[q]] + (size_t)pos[A->fc[q]] * s] = A->fv[q];
        amax = fmax(amax, fabs(A->fv[q]));
    }
    free(pos);
    double *ev = malloc(sizeof(double) * s), wq;
    double *T2 = malloc(sizeof(double) * (size_t)s * s);
    memcpy(T2, T, sizeof(double) * (size_t)s * s);
    int lwork = -1, info;
    BL(dsyev_)("V", "U", &s, T, &s, ev, &wq, &lwork, &info);
    lwork = (int)wq + 1;
    double *work = malloc(sizeof(double) * lwork);
    BL(dsyev_)("N", "U", &s, T2, &s, ev, work, &lwork, &info);     /* eigenvalues only */
    int semidef = (info == 0);
    if (semidef) {
        double em = fmax(fabs(ev[0]), fabs(ev[s - 1]));
        semidef = em > 0 && !(ev[0] < -1e-10 * em && ev[s - 1] > 1e-10 * em);
    }
    if (semidef) BL(dsyev_)("V", "U", &s, T, &s, ev, work, &lwork, &info);
    free(work); free(T2);
    if (!semidef || info != 0) { free(T); free(ev); return 0; }
    double emax = fmax(fabs(ev[0]), fabs(ev[s - 1]));
    double ztol = zrel * emax;
    int neg = 0, pos_ = 0, k = 0;
    double nzmin = HUGE_VAL, zmax = 0;
    for (int i = 0; i < s; i++) {
        if (ev[i] < -ztol) neg++;
        else if (ev[i] > ztol) pos_++;
        else k++;
        if (fabs(ev[i]) > ztol) nzmin = fmin(nzmin, fabs(ev[i])); else zmax = fmax(zmax, fabs(ev[i]));
    }
    if (gapf > 0 && getenv("BRISK_CFDBG")) {
        printf("   [clique face eigen test: order %d, emax %.3e, zero %d, neg %d, pos %d, smallest nonzero %.3e;", s, emax, k, neg, pos_, nzmin / emax);
        for (int i = 0; i < s && i < 12; i++) printf(" %.2e", ev[i] / emax);
        printf(" ...]\n");
    }
    if (emax == 0 || (neg && pos_) || (gapf > 0 && nzmin < gapf * fmax(zmax, 1e-16 * emax))) { free(T); free(ev); return 0; }
    *sgn = neg ? -1.0 : 1.0;
    double *N = malloc(sizeof(double) * (size_t)s * (k ? k : 1));
    double *U = malloc(sizeof(double) * (size_t)s * (s - k ? s - k : 1));
    int c = 0, cu = 0;
    for (int i = 0; i < s; i++) {
        if (fabs(ev[i]) <= ztol) memcpy(N + (size_t)(c++) * s, T + (size_t)i * s, sizeof(double) * s);
        else memcpy(U + (size_t)(cu++) * s, T + (size_t)i * s, sizeof(double) * s);
    }
    if (Uout) *Uout = U; else free(U);
    free(T); free(ev);
    int *Sc = malloc(sizeof(int) * s);
    memcpy(Sc, S, sizeof(int) * s);
    *Sout = Sc; *sout = s; *Nout = N; *kout = k;
    return 1;
}
static int psd_nullspace(const SpSym *A, int n, int smax, double *sgn, int **Sout, int *sout,
                         double **Nout, int *kout, double **Uout) {
    return psd_nullspace_tol(A, n, smax, 1e-10, 0.0, sgn, Sout, sout, Nout, kout, Uout);
}

static int g_bfr_sparse = 0;   /* (the entry-by-entry compression: the faces of clique_faces only, so that the others keep their bits) */
static void block_face_reduce(Block *B, const int *S, int s, const double *N, int k) {
    const int n = B->n;
    char *inS = calloc(n, 1);
    int *spos = malloc(sizeof(int) * n);
    for (int i = 0; i < s; i++) { inS[S[i]] = 1; spos[S[i]] = i; }
    int *map = malloc(sizeof(int) * n);
    int nr = 0;
    for (int i = 0; i < n; i++) map[i] = inS[i] ? -1 : nr++;
    const int nn = nr + k;
    if (g_ps_P) ps_hook_face(g_ps_P, B, S, s, N, k, inS);
    double *MSS = malloc(sizeof(double) * (size_t)s * s);
    double *tmp = malloc(sizeof(double) * (size_t)s * (k ? k : 1));
    double *red = malloc(sizeof(double) * (size_t)(k ? k : 1) * (k ? k : 1));
    double *rowS = calloc((size_t)n * s, sizeof(double));      /* M[i, S] for i outside S */
    char *rowtouch = calloc(n, 1);
    double *Nt = NULL;                                          /* N by rows (the sparse path) */
    const double d1 = 1.0, d0 = 0.0;
    for (int t = -1; t < B->ncon; t++) {
        SpSym *M = t < 0 ? &B->C : &B->A[t];
        int touches = 0;
        for (int q = 0; q < M->ef; q++) if (inS[M->fr[q]] || inS[M->fc[q]]) { touches = 1; break; }
        size_t cap = (size_t)M->nnz + 1;
        if (touches) cap += (size_t)k * (k + 1) / 2 + (size_t)n * k;
        Ent *E = malloc(sizeof(Ent) * cap);
        size_t ne = 0;
        if (!touches) {
            for (int q = 0; q < M->nnz; q++) {
                E[ne].r = map[M->row[q]]; E[ne].c = map[M->col[q]]; E[ne].v = M->val[q]; ne++;
            }
        } else {
            /* 3 November 2026 (ISSUES 745): a matrix with few entries on S (swissroll's distance constraints: 3 to 5 entries
             * on a support of 799) is compressed entry by entry, sum_e v_e N(r_e,:)' N(c_e,:): ess k^2 work instead of
             * the products' 2 s^2 k and the s^2 clearing */
            int ess = 0;
            for (int q = 0; q < M->ef; q++) if (inS[M->fr[q]] && inS[M->fc[q]]) ess++;
            const int sparse_ss = g_bfr_sparse && k > 0 && (double)ess * k < 2.0 * s * s;
            if (!sparse_ss) memset(MSS, 0, sizeof(double) * (size_t)s * s);
            else {
                if (!Nt) {
                    Nt = malloc(sizeof(double) * (size_t)s * k);
                    for (int a = 0; a < k; a++) for (int j = 0; j < s; j++) Nt[(size_t)j * k + a] = N[j + (size_t)a * s];
                }
                memset(red, 0, sizeof(double) * (size_t)k * k);
            }
            int nrows = 0;
            int *rows = malloc(sizeof(int) * n);
            for (int q = 0; q < M->ef; q++) {
                int r = M->fr[q], c = M->fc[q];
                if (inS[r] && inS[c]) {
                    if (!sparse_ss) MSS[spos[r] + (size_t)spos[c] * s] = M->fv[q];
                    else {
                        const double v = M->fv[q], *nr_ = Nt + (size_t)spos[r] * k, *nc_ = Nt + (size_t)spos[c] * k;
                        for (int b2 = 0; b2 < k; b2++) {
                            const double w = v * nc_[b2];
                            double *col = red + (size_t)b2 * k;
                            for (int a = 0; a < k; a++) col[a] += nr_[a] * w;
                        }
                    }
                }
                else if (!inS[r] && inS[c]) {
                    if (!rowtouch[r]) { rowtouch[r] = 1; rows[nrows++] = r; }
                    rowS[(size_t)r * s + spos[c]] = M->fv[q];
                } else if (!inS[r] && !inS[c] && r <= c) {
                    E[ne].r = map[r]; E[ne].c = map[c]; E[ne].v = M->fv[q]; ne++;
                }
            }
            if (k > 0) {
                /* N' M_SS N */
                if (!sparse_ss) {
                    BL(dgemm_)("N", "N", &s, &k, &s, &d1, MSS, &s, N, &s, &d0, tmp, &s);
                    BL(dgemm_)("T", "N", &k, &k, &s, &d1, N, &s, tmp, &s, &d0, red, &k);
                }
                for (int b2 = 0; b2 < k; b2++)          /* (by column, then row: the order of sp_from_entries) */
                    for (int a = 0; a <= b2; a++) {
                        double v = 0.5 * (red[a + (size_t)b2 * k] + red[b2 + (size_t)a * k]);
                        if (fabs(v) > 1e-15) { E[ne].r = nr + a; E[ne].c = nr + b2; E[ne].v = v; ne++; }
                    }
                /* M[i,S] N for rows outside S */
                for (int q = 0; q < nrows; q++) {
                    int r = rows[q];
                    const double *ms = rowS + (size_t)r * s;
                    for (int a = 0; a < k; a++) {
                        double v = 0;
                        const double *na = N + (size_t)a * s;
                        for (int j = 0; j < s; j++) v += ms[j] * na[j];
                        if (fabs(v) > 1e-15) { E[ne].r = map[r]; E[ne].c = nr + a; E[ne].v = v; ne++; }
                    }
                }
            }
            for (int q = 0; q < nrows; q++) {
                rowtouch[rows[q]] = 0;
                memset(rowS + (size_t)rows[q] * s, 0, sizeof(double) * s);
            }
            free(rows);
        }
        sp_from_entries(M, E, ne, nn, 0);
        free(E);
    }
    B->n = nn;
    free(inS); free(spos); free(map); free(MSS); free(tmp); free(red); free(rowS); free(rowtouch); free(Nt);
}

/* ---------------------------------------------------------------------- */
/* 2 November 2026 (ISSUES 745): a constraint <A, X> = 0 whose matrix is +-semidefinite of any rank, proven in exact
 * arithmetic as a sum of rank-one terms A = sum_t c_t c_t' / d_t (d_t > 0) by a sparse symmetric elimination with diagonal
 * pivots (the least connected row first). Every product, quotient and difference of the elimination is checked to be
 * exact in double (fma for the product and the quotient, the error term of the difference); one rounding anywhere and
 * nothing is claimed. Then every feasible X has c_t' X c_t = 0 for every t, so X = V W V' with V spanning the vectors
 * orthogonal to all c_t, and the block is reduced by the rank-one substitutions of the terms in their elimination order
 * (the term t is zero at the pivots eliminated before it, so each substitution leaves the later terms as they are): a
 * sparse reduction, where the eigenvector basis of psd_nullspace (dense, support up to 400) would make every matrix of
 * the block dense. tiger_texture (Mittelmann, SDP): the sum over 600 pixels of (x_a + x_b + x_c - 1)^2 lifted, rank 600
 * on a support of 1,801. Work bounded by `budget` updates; support at most 3,000 (the postsolve keeps the support's
 * null-space and range bases densely). BRISK_FRTERMS=0: off. */
typedef struct {
    int s, r;                /* support size, number of terms (the rank) */
    int *sup;                /* support (block indices at the test) */
    int *piv;                /* per term: its pivot (local index into sup) */
    int *cp, *ci; double *cv;/* per term: its column c_t (local indices, values; the pivot included) */
    double sgn;
} PsdTerms;
static void psd_terms_free(PsdTerms *T) { free(T->sup); free(T->piv); free(T->cp); free(T->ci); free(T->cv); memset(T, 0, sizeof(*T)); }
/* old - vi vj / d exactly in double, or 0 */
static int ex_update(double old, double vi, double vj, double d, double *out) {
    const double q = vi * vj;
    if (fma(vi, vj, -q) != 0.0) return 0;
    const double r = q / d;
    if (fma(r, d, -q) != 0.0) return 0;
    const double s = old - r, bb = s - old;
    if ((old - (s - bb)) + (-r - bb) != 0.0) return 0;
    *out = s;
    return 1;
}
static int psd_terms(const SpSym *A, int n, int smax, double *sgn, PsdTerms *out) {
    const int s = A->nr;
    static int off = -1; if (off < 0) { const char *e = getenv("BRISK_FRTERMS"); off = e && atoi(e) == 0; }
    if (s < 2 || s > smax || off) return 0;
    if (!semidef_quick(A, n)) return 0;
    int *pos = malloc(sizeof(int) * n);
    for (int i = 0; i < n; i++) pos[i] = -1;
    for (int i = 0; i < s; i++) pos[A->rows[i]] = i;
    double sg = 0;
    for (int q = 0; q < A->nnz; q++) if (A->row[q] == A->col[q] && A->val[q] != 0) { sg = A->val[q] > 0 ? 1.0 : -1.0; break; }
    if (sg == 0) { free(pos); return 0; }
    /* rows: off-diagonal lists (local index, value) and the diagonal */
    int *len = calloc(s, sizeof(int)), *cap = malloc(sizeof(int) * s), **rc = malloc(sizeof(int *) * s);
    double *dg = calloc(s, sizeof(double)), **rv = malloc(sizeof(double *) * s);
    char *alive = malloc(s);
    for (int i = 0; i < s; i++) { cap[i] = 4; rc[i] = malloc(sizeof(int) * 4); rv[i] = malloc(sizeof(double) * 4); alive[i] = 1; }
    #define PT_PUSH(i_, j_, v_) do { if (len[i_] == cap[i_]) { cap[i_] *= 2; rc[i_] = realloc(rc[i_], sizeof(int) * cap[i_]); rv[i_] = realloc(rv[i_], sizeof(double) * cap[i_]); } rc[i_][len[i_]] = (j_); rv[i_][len[i_]] = (v_); len[i_]++; } while (0)
    for (int q = 0; q < A->ef; q++) {
        const int i = pos[A->fr[q]], j = pos[A->fc[q]];
        const double v = sg * A->fv[q];
        if (i == j) dg[i] = v; else PT_PUSH(i, j, v);
    }
    free(pos);
    const long budget = 50L * A->nnz + 2000000L;
    long work = 0;
    int ok = 1, r = 0, tcap = 64, ccap = 256, nc = 0;
    int *piv = malloc(sizeof(int) * tcap), *cp = malloc(sizeof(int) * (tcap + 1)), *ci = malloc(sizeof(int) * ccap);
    double *cv = malloc(sizeof(double) * ccap);
    cp[0] = 0;
    for (;;) {
        int p = -1;
        for (int i = 0; i < s && ok; i++) {
            if (!alive[i]) continue;
            if (dg[i] < 0 || (dg[i] == 0 && len[i] > 0)) ok = 0;          /* not semidefinite */
            else if (dg[i] == 0) alive[i] = 0;                              /* a zero row: a free coordinate */
            else if (p < 0 || len[i] < len[p] || (len[i] == len[p] && dg[i] > dg[p])) p = i;
        }
        if (!ok || p < 0) break;
        if (work > budget) { ok = 0; break; }
        /* the term: c = row p with its diagonal */
        if (r == tcap) { tcap *= 2; piv = realloc(piv, sizeof(int) * tcap); cp = realloc(cp, sizeof(int) * (tcap + 1)); }
        while (nc + len[p] + 1 > ccap) { ccap *= 2; ci = realloc(ci, sizeof(int) * ccap); cv = realloc(cv, sizeof(double) * ccap); }
        piv[r] = p;
        ci[nc] = p; cv[nc] = dg[p]; nc++;
        for (int e = 0; e < len[p]; e++) { ci[nc] = rc[p][e]; cv[nc] = rv[p][e]; nc++; }
        cp[++r] = nc;
        const double d = dg[p];
        alive[p] = 0;
        work += (long)len[p] * (len[p] + 1);
        if (work > budget) { ok = 0; break; }
        /* the Schur update over the neighbours of p, and p out of their rows */
        for (int x = 0; x < len[p] && ok; x++) {
            const int i = rc[p][x]; const double vi = rv[p][x];
            for (int e = 0; e < len[i]; e++) if (rc[i][e] == p) { rc[i][e] = rc[i][len[i] - 1]; rv[i][e] = rv[i][len[i] - 1]; len[i]--; break; }
            double nv;
            if (!ex_update(dg[i], vi, vi, d, &nv)) { ok = 0; break; }
            dg[i] = nv;
            for (int y = 0; y < len[p] && ok; y++) {
                const int j = rc[p][y];
                if (j == i) continue;
                const double vj = rv[p][y];
                int f = -1;
                work += len[i];
                for (int e = 0; e < len[i]; e++) if (rc[i][e] == j) { f = e; break; }
                if (!ex_update(f >= 0 ? rv[i][f] : 0.0, vi, vj, d, &nv)) { ok = 0; break; }
                if (f >= 0) {
                    if (nv == 0.0) { rc[i][f] = rc[i][len[i] - 1]; rv[i][f] = rv[i][len[i] - 1]; len[i]--; }
                    else rv[i][f] = nv;
                } else if (nv != 0.0) PT_PUSH(i, j, nv);
            }
        }
        len[p] = 0;
    }
    #undef PT_PUSH
    for (int i = 0; i < s; i++) { free(rc[i]); free(rv[i]); }
    free(rc); free(rv); free(len); free(cap); free(dg); free(alive);
    if (!ok || r == 0 || r == s) { free(piv); free(cp); free(ci); free(cv); return 0; }   /* (rank s: X = 0 on the support, the diagonal cases' work) */
    *sgn = sg;
    if (out) {
        out->s = s; out->r = r; out->sgn = sg;
        out->sup = malloc(sizeof(int) * s); memcpy(out->sup, A->rows, sizeof(int) * s);
        out->piv = piv; out->cp = cp; out->ci = ci; out->cv = cv;
    } else { free(piv); free(cp); free(ci); free(cv); }
    return r;
}
/* the null space (s x k, k = s - r) and range (s x r) bases of the terms on the support, dense, for the postsolve: a
 * non-pivot coordinate u gives e_u with the pivots solved in reverse elimination order (c_t' v = 0 for every t) */
static void psd_terms_bases(const PsdTerms *T, double **Nout, double **Uout, int *kout) {
    const int s = T->s, r = T->r, k = s - r;
    char *isp = calloc(s, 1);
    for (int t = 0; t < r; t++) isp[T->piv[t]] = 1;
    double *N = calloc((size_t)s * (k ? k : 1), sizeof(double)), *U = calloc((size_t)s * (r ? r : 1), sizeof(double));
    for (int t = 0; t < r; t++) for (int e = T->cp[t]; e < T->cp[t + 1]; e++) U[T->ci[e] + (size_t)t * s] = T->cv[e];
    for (int u = 0, a = 0; u < s; u++) {
        if (isp[u]) continue;
        double *v = N + (size_t)a * s;
        v[u] = 1.0;
        for (int t = r - 1; t >= 0; t--) {
            double sm = 0, vp = 0;
            const int p = T->piv[t];
            for (int e = T->cp[t]; e < T->cp[t + 1]; e++) { if (T->ci[e] == p) vp = T->cv[e]; else sm += T->cv[e] * v[T->ci[e]]; }
            v[p] = -sm / vp;
        }
        a++;
    }
    free(isp);
    *Nout = N; *Uout = U; *kout = k;
}

/* Remove rows/columns of SDP blocks (variables of LP blocks) that appear in no
 * constraint and not in C: they can be fixed to zero, and keeping them leaves
 * the dual slack Z singular (no strictly feasible dual point).               */
static int remove_unused_indices(Problem *P) {
    int removed = 0;
    for (int k = 0; k < P->nblk; k++) {
        Block *B = &P->blk[k];
        char *used = calloc(B->n, 1);
        for (int t = -1; t < B->ncon; t++) {
            const SpSym *M = t < 0 ? &B->C : &B->A[t];
            for (int q = 0; q < M->nnz; q++) { used[M->row[q]] = 1; used[M->col[q]] = 1; }
        }
        char *del = calloc(B->n, 1);
        int cnt = 0;
        for (int i = 0; i < B->n; i++) if (!used[i]) { del[i] = 1; cnt++; }
        if (cnt > 0 && cnt < B->n) block_delete_indices(B, del);
        else if (cnt == B->n) { if (g_ps_P) ps_hook_vanish(g_ps_P, B); B->n = 0; }
        removed += cnt;
        free(used); free(del);
    }
    return removed;
}

static int find_part(const Block *B, int i) {
    int lo = 0, hi = B->ncon - 1;
    while (lo <= hi) {
        int mid = (lo + hi) / 2;
        if (B->con[mid] == i) return mid;
        if (B->con[mid] < i) lo = mid + 1; else hi = mid - 1;
    }
    return -1;
}

/* 5.18: the parts of constraint i in the blocks blks[0..nb) taken out, and the parts those
 * blocks' reduction left empty dropped: the constraint numbering is kept until the end of the
 * sweep (compact_removed). Until 5.17 every reduction removed the constraint from every block
 * and pruned the empty parts of all blocks, walking every part of every block and shifted b, b0, orig: (reductions x parts) work, 2.8 s
 * of a 4 s run on a chained-Wood SOS relaxation with 1,000 variables (16,972 reductions). */
static void remove_parts_of(Problem *P, int i, const int *parts, int np) {
    for (int q = 0; q < np; q++) {
        Block *B = &P->blk[parts[2 * q]];
        int w = 0;
        for (int t = 0; t < B->ncon; t++) {
            if (B->con[t] == i || B->A[t].nnz == 0) { sp_release(&B->A[t]); continue; }
            B->A[w] = B->A[t]; B->con[w] = B->con[t]; w++;
        }
        B->ncon = w;
    }
}
/* the constraints marked dead[] removed at once: the others renumbered in order (the con lists
 * stay sorted), b, b0, orig and the levels clev compacted */
static void compact_removed(Problem *P, const char *dead, int *clev) {
    int *nw = malloc(sizeof(int) * (P->m + 1)), w = 0;
    for (int i = 0; i < P->m; i++) {
        nw[i] = dead[i] ? -1 : w;
        if (!dead[i]) { P->b[w] = P->b[i]; P->b0[w] = P->b0[i]; P->orig[w] = P->orig[i]; clev[w] = clev[i]; w++; }
    }
    for (int k = 0; k < P->nblk; k++) {
        Block *B = &P->blk[k];
        int u = 0;
        for (int t = 0; t < B->ncon; t++) {
            if (nw[B->con[t]] < 0) { sp_release(&B->A[t]); continue; }
            B->A[u] = B->A[t]; B->con[u] = nw[B->con[t]]; u++;
        }
        B->ncon = u;
    }
    P->m = w;
    free(nw);
}

/* drop constraint parts that became empty */
static void prune_empty_parts(Problem *P) {
    for (int k = 0; k < P->nblk; k++) {
        Block *B = &P->blk[k];
        int w = 0;
        for (int t = 0; t < B->ncon; t++) {
            if (B->A[t].nnz == 0) { sp_release(&B->A[t]); continue; }
            B->A[w] = B->A[t]; B->con[w] = B->con[t]; w++;
        }
        B->ncon = w;
    }
}


/* FR depth (4.20): a reduction has level 1 + the highest level of the reductions that
 * changed its constraint's matrix. Level >= 2 is how a singularity degree >= 2 shows up:
 * the dual optimum of the original problem is then typically not attained and the
 * recovery of the removed duals diverges (rose13, roa_acrobot, taha1a).              */
static void fr_touch(const Block *B, const unsigned char *mask, int L, int *clev, int skip) {
    for (int u = 0; u < B->ncon; u++) {
        if (u == skip) continue;
        const SpSym *S = &B->A[u];
        for (int e = 0; e < S->nnz; e++)
            if (mask[S->row[e]] || mask[S->col[e]]) { if (clev[B->con[u]] < L) clev[B->con[u]] = L; break; }
    }
}

/* 5.17 (ISSUES 93): does the data couple the directions a reduction removes to the face it keeps?
 * The recovery of the removed dual through the certificate needs the coupling g (the slack between
 * the removed directions W and the face) in the range of the reduced slack Z_n; with no data entry
 * between W and the face, g = 0 and the recovery always succeeds; with such entries it succeeds
 * only when the reduced slack happens to be nonsingular on them (a moment matrix of finitely
 * many atoms leaves Z_n singular, and the recovery fails). The test is on the block as it is at
 * the reduction (its current coordinates), matrix by matrix, the removed constraint left out. */
static int fr_coupled(const Block *B, int skip, int kind, const char *del, const double *a,
                      const int *Sx, int sx, const double *Nx, int kx, const double *Ux) {
    const int n = B->n;
    if (kind == 2) return 0;                   /* an LP variable: its dual is free of the others */
    double *v = NULL, *T = NULL, *Y = NULL; int *pos = NULL;
    double amax = 0, aa = 0;
    if (kind == 0) { v = malloc(sizeof(double) * n); for (int r = 0; r < n; r++) { amax = fmax(amax, fabs(a[r])); aa += a[r] * a[r]; } }
    if (kind == 3) {
        pos = malloc(sizeof(int) * n);
        for (int i = 0; i < n; i++) pos[i] = -1;
        for (int i = 0; i < sx; i++) pos[Sx[i]] = i;
        T = malloc(sizeof(double) * ((size_t)sx * sx + 1)); Y = malloc(sizeof(double) * ((size_t)sx * sx + 1));
    }
    int coupled = 0;
    for (int u = -1; u < B->ncon && !coupled; u++) {
        if (u == skip) continue;
        const SpSym *M = u < 0 ? &B->C : &B->A[u];
        if (M->nnz == 0) continue;
        if (kind == 1) {
            for (int e = 0; e < M->nnz; e++) if (del[M->row[e]] != del[M->col[e]]) { coupled = 1; break; }
        } else if (kind == 0) {
            /* M a must be parallel to a (the removed direction is a; the face is its complement) */
            double mmax = 0;
            for (int r = 0; r < n; r++) v[r] = 0;
            for (int q = 0; q < M->ef; q++) { v[M->fr[q]] += M->fv[q] * a[M->fc[q]]; mmax = fmax(mmax, fabs(M->fv[q])); }
            double av = 0;
            for (int r = 0; r < n; r++) av += a[r] * v[r];
            const double c = aa > 0 ? av / aa : 0, tol = 1e-10 * mmax * amax * n;
            for (int r = 0; r < n; r++) if (fabs(v[r] - c * a[r]) > tol) { coupled = 1; break; }
        } else {
            /* U' M V with V = [N on the support S; the identity off S]: an entry between S and its
             * complement couples (U has full row rank on S); on S, Y = U' M_SS N must vanish */
            double mmax = 0; int inS = 0;
            for (size_t q = 0; q < (size_t)sx * sx; q++) T[q] = 0;
            for (int e = 0; e < M->nnz; e++) {
                const int r = M->row[e], cc = M->col[e];
                const int pr = pos[r], pc = pos[cc];
                mmax = fmax(mmax, fabs(M->val[e]));
                if ((pr >= 0) != (pc >= 0)) { coupled = 1; break; }
                if (pr >= 0) { T[pr + (size_t)pc * sx] += M->val[e]; if (pr != pc) T[pc + (size_t)pr * sx] += M->val[e]; inS = 1; }
            }
            if (coupled || !inS) continue;
            const int kw = sx - kx;
            if (kw <= 0 || kx <= 0) continue;
            /* Y = (U' T) N */
            for (int b = 0; b < kx; b++) for (int aw = 0; aw < kw; aw++) {
                double s = 0;
                for (int i = 0; i < sx; i++) { double ti = 0; for (int j = 0; j < sx; j++) ti += T[i + (size_t)j * sx] * Nx[j + (size_t)b * sx]; s += Ux[i + (size_t)aw * sx] * ti; }
                if (fabs(s) > 1e-10 * mmax * sx) { coupled = 1; break; }
            }
        }
    }
    free(v); free(T); free(Y); free(pos);
    return coupled;
}

/* 5.18: can the facial reduction remove anything? A reduction needs b_i = 0 and every part of constraint i
 * semidefinite of one sign (an LP part: coefficients of one sign; semidef_quick is necessary for that), and a
 * constraint that fails it can only become a candidate after some other reduction; so when no constraint passes,
 * the facial reduction records nothing (unused indices aside) and the copy of the problem kept for undoing it
 * (main.c) is not needed: 1 s on an A-optimal design file of 8 million entries. */
int fr_has_candidate(const Problem *P) {
    char *bad = calloc(P->m + 1, 1), *seen = calloc(P->m + 1, 1);
    for (int i = 0; i < P->m; i++) if (P->b[i] != 0.0) bad[i] = 1;
    for (int k = 0; k < P->nblk; k++) {
        const Block *B = &P->blk[k];
        for (int t = 0; t < B->ncon; t++) {
            const int i = B->con[t];
            if (bad[i]) continue;
            const SpSym *S = &B->A[t];
            seen[i] = 1;
            if (S->nnz == 0) continue;
            if (B->type == BLK_LP) { double sg; if (!samesign_diag(S, &sg)) bad[i] = 1; }
            else if (!semidef_quick(S, B->n)) bad[i] = 1;
        }
    }
    int any = 0;
    for (int i = 0; i < P->m && !any; i++) if (!bad[i]) any = 1;     /* (an empty constraint with b = 0 counts) */
    free(bad); free(seen);
    return any;
}

/* 23 October 2026: faces of lifted relaxations exposed by their linear constraints. In a block Y = [g x'; x X] whose
 * diagonal entry Y_hh = g > 0 is fixed by a constraint of its own, a constraint on row h alone reads a'x = beta (x the
 * rest of row h, a_j = 2 A_hj, beta = b - A_hh g) and gives v = (-beta/g at h, a elsewhere) with v'Yv = a'Xa - beta^2/g,
 * which is >= 0 for every Y >= 0 (X >= x x'/g). When v v' is a combination sum d_i A_i of the constraints with b'd = 0,
 * every feasible Y has <v v', Y> = b'd = 0, hence Y v = 0: a face of the cone that only that combination exposes, so the
 * single-constraint facial reduction below does not see it (the lifted QAP relaxation: its row and column sums with the
 * diagonal and the products of two entries of one row or column, the face of Zhao, Karisch, Rendl and Wolkowicz, 1998).
 * The test is exact linear algebra on the constraints of the block whose entries lie in supp(v) x supp(v): the normal
 * equations of the least-squares fit, two refinement steps, the residual against |v|^2 and b'd against the sizes of its
 * terms. Each exact v becomes a constraint <v v', Y> = 0 (b = 0, rank one) that the facial reduction then removes like
 * any other (sparse: a unit vector plus one entry per index); its orig is -2 - j for combination j of the postsolve's
 * table, through which the postsolve spreads its dual over the file's rows. Returns the number of constraints added.
 * BRISK_LIFTFACE=0: off. */
/* 31 October 2026 (the owner's benchmark, the audit's A1): a face of a lifted relaxation is removed only when it is
 * proven in exact arithmetic on the data, never on a tolerance. The input numbers are doubles, hence exact rationals; the
 * combination d found by least squares is normalized (largest entry 1) and rounded to rationals of a common denominator Q
 * (continued fractions, denominators up to 4096, Q up to 2^24), and with the integers P_a = Q d_a the matrix
 * W = sum P_a A_a and the number sum P_a b_a are formed exactly as expansions (sums of nonoverlapping doubles, Shewchuk's
 * arithmetic: two_sum and the exact product by fma). The face is accepted when sum P_a b_a = 0 exactly and W is exactly
 * of rank one with W_hh > 0 (W_rc W_hh = W_hr W_hc for every pair), so that W = w w' / W_hh is positive semidefinite and
 * <W, X> = 0 on every feasible X: then X w = 0. A combination that passes the floating-point test but not this one is a
 * near-face: nothing is removed (the 16x16 reproducer of the audit and the weak-Slater family: a positive b'd below the
 * old test's floor read as zero, and feasible problems were declared infeasible). */
typedef struct { double *v; int n, cap; } LfExp;
static void lfx_push(LfExp *E, double x) { if (E->n == E->cap) { E->cap = E->cap ? 2 * E->cap : 4; E->v = realloc(E->v, sizeof(double) * E->cap); } E->v[E->n++] = x; }
/* E += b, exactly (grow-expansion with zero elimination; E nonoverlapping, increasing magnitude) */
static void lfx_add(LfExp *E, double b) {
    if (b == 0) return;
    double q = b; int w = 0;
    for (int i = 0; i < E->n; i++) {
        const double x = q + E->v[i], bv = x - q, av = x - bv, br = E->v[i] - bv, ar = q - av, hh = ar + br;
        q = x;
        if (hh != 0) E->v[w++] = hh;
    }
    E->n = w;
    if (q != 0) lfx_push(E, q);
}
/* E += a * b, exactly (the product as two doubles by fma) */
static void lfx_addprod(LfExp *E, double a, double b) { const double p = a * b, e = fma(a, b, -p); lfx_add(E, e); lfx_add(E, p); }
/* R = X * Y exactly (every product of components) */
static void lfx_mul(LfExp *R, const LfExp *X, const LfExp *Y) { R->n = 0; for (int i = 0; i < X->n; i++) for (int j = 0; j < Y->n; j++) lfx_addprod(R, X->v[i], Y->v[j]); }
static int lfx_sign(const LfExp *E) { for (int i = E->n - 1; i >= 0; i--) if (E->v[i] != 0) return E->v[i] > 0 ? 1 : -1; return 0; }
/* the best rational p/q of x with q <= qmax (continued fractions); returns q, 0 when none is within tol */
static long lf_rat(double x, long qmax, double tol, long *p) {
    long h0 = 0, h1 = 1, k0 = 1, k1 = 0; double r = x;
    for (int it = 0; it < 40; it++) {
        const double a = floor(r); if (fabs(a) > 1e12) break;
        const long ai = (long)a, h2 = ai * h1 + h0, k2 = ai * k1 + k0;
        if (k2 > qmax) break;
        h0 = h1; h1 = h2; k0 = k1; k1 = k2;
        if (fabs(x - (double)h1 / (double)k1) <= tol) { *p = h1; return k1; }
        if (r - a < 1e-15) break;
        r = 1.0 / (r - a);
    }
    return 0;
}
static long lf_gcd(long a, long b) { if (a < 0) a = -a; if (b < 0) b = -b; while (b) { const long t = a % b; a = b; b = t; } return a; }
/* the exact proof; on success the weights P_a / Q replace d */
static int lf_exact_face(const Problem *P, const Block *B, const int *rt, int nr, const int *pos, const int *sup, int s, int h, double *d) {
    /* each weight's contribution to W (|d_a| times its constraint's largest entry): a weight whose contribution is below
     * 1e-12 of the largest is 0 (least squares leaves rounding there); the others are rationalized after their binary
     * exponent is split off (x = f 2^e, f in [1/2, 1) rationalized), so that rows scaled by powers of two - which keep
     * the data's identities exact - keep their faces (the coefficients 2^-20 of a row scaled by 2^20) */
    double *ct = malloc(sizeof(double) * (size_t)nr), cm = 0;
    for (int a = 0; a < nr; a++) {
        const SpSym *A = &B->A[rt[a]]; double am = 0;
        for (int e = 0; e < A->nnz; e++) if (fabs(A->val[e]) > am) am = fabs(A->val[e]);
        ct[a] = fabs(d[a]) * am; if (ct[a] > cm) cm = ct[a];
    }
    if (!(cm > 0)) { free(ct); return 0; }
    long *pn = malloc(sizeof(long) * (size_t)nr), *qd = malloc(sizeof(long) * (size_t)nr), Q = 1;
    int *ex = malloc(sizeof(int) * (size_t)nr);
    int ok = 1;
    for (int a = 0; a < nr && ok; a++) {
        if (ct[a] < 1e-12 * cm || d[a] == 0) { pn[a] = 0; qd[a] = 1; ex[a] = 0; continue; }
        const double fr = frexp(d[a], &ex[a]);           /* d_a = fr 2^ex, |fr| in [1/2, 1) */
        qd[a] = lf_rat(fr, 4096, 1e-9, &pn[a]);
        if (!qd[a]) { ok = 0; if (getenv("BRISK_LFDBG")) printf("   [lifted face: d_%d = %.17g has no small rational]\n", a, d[a]); break; }
        const long g = lf_gcd(Q, qd[a]); Q = Q / g * qd[a];
        if (Q > (1L << 24)) { ok = 0; if (getenv("BRISK_LFDBG")) printf("   [lifted face: common denominator above 2^24]\n"); }
    }
    free(ct);
    if (ok) {
        /* Pa = Q d_a = (p_a Q / q_a) 2^ex_a: an integer below 2^53 times a power of two, exact in a double */
        double *Pa = malloc(sizeof(double) * (size_t)nr);
        for (int a = 0; a < nr && ok; a++) {
            const long iv = pn[a] * (Q / qd[a]);
            if (labs(iv) >= (1L << 53)) ok = 0;
            Pa[a] = ldexp((double)iv, ex[a]);
        }
        LfExp bd = { 0 }; for (int a = 0; a < nr; a++) lfx_addprod(&bd, Pa[a], P->b[B->con[rt[a]]]);
        if (lfx_sign(&bd) != 0) { ok = 0; if (getenv("BRISK_LFDBG")) printf("   [lifted face: b'd = %.3e exactly nonzero (Q %ld)]\n", bd.n ? bd.v[bd.n - 1] : 0.0, Q); }
        free(bd.v);
        const long npos = (long)s * (s + 1) / 2;
        LfExp *W = ok ? calloc((size_t)npos, sizeof(LfExp)) : NULL;
        if (W) {
            #define LF_Q(a, b) ((a) <= (b) ? (long)(b) * ((b) + 1) / 2 + (a) : (long)(a) * ((a) + 1) / 2 + (b))
            for (int a = 0; a < nr; a++) if (Pa[a] != 0) {
                const SpSym *A = &B->A[rt[a]];
                for (int e = 0; e < A->nnz; e++) lfx_addprod(&W[LF_Q(pos[A->row[e]], pos[A->col[e]])], Pa[a], A->val[e]);
            }
            /* (the pivot: the largest diagonal entry of W; v_h itself can be 0 - rose15, Mittelmann, SDP) */
            int ph = pos[h]; double best = -1;
            for (int q = 0; q < s; q++) { const LfExp *Wq = &W[LF_Q(q, q)]; const double x = Wq->n ? Wq->v[Wq->n - 1] : 0.0; if (x > best) { best = x; ph = q; } }
            const LfExp *Whh = &W[LF_Q(ph, ph)];
            if (lfx_sign(Whh) <= 0) { ok = 0; if (getenv("BRISK_LFDBG")) printf("   [lifted face: W_hh not positive]\n"); }
            LfExp L = { 0 }, R = { 0 };
            for (int c = 0; c < s && ok; c++) for (int r = 0; r <= c && ok; r++) {
                lfx_mul(&L, &W[LF_Q(r, c)], Whh);
                lfx_mul(&R, &W[LF_Q(ph, r)], &W[LF_Q(ph, c)]);
                for (int i = 0; i < R.n; i++) lfx_add(&L, -R.v[i]);
                if (lfx_sign(&L) != 0) { ok = 0; if (getenv("BRISK_LFDBG")) printf("   [lifted face: W not of rank one at (%d,%d): %.3e]\n", r, c, L.n ? L.v[L.n - 1] : 0.0); }
            }
            #undef LF_Q
            free(L.v); free(R.v);
            for (long q = 0; q < npos; q++) free(W[q].v);
            free(W);
        }
        if (ok) for (int a = 0; a < nr; a++) d[a] = Pa[a] / (double)Q;
        free(Pa);
    }
    (void)sup; (void)h;
    free(pn); free(qd); free(ex);
    return ok;
}

double g_lf_time = 0;     /* (the detection's time, for BRISK_LFSCAN) */
int g_lf_near = 0;        /* (the candidates that passed the floating-point test but not the exact proof: near-faces, not removed) */
int lifted_faces(Problem *P, int verbose) {
    static int off = -1; if (off < 0) { const char *e = getenv("BRISK_LIFTFACE"); off = e && atoi(e) == 0; }
    if (off || !P->ps || !P->orig || P->m < 3) return 0;
    const double tlf0 = wtime();
    const int m0 = P->m;
    int *nb = calloc((size_t)m0 + 1, sizeof(int));          /* blocks in which a constraint has entries */
    for (int k = 0; k < P->nblk; k++) for (int t = 0; t < P->blk[k].ncon; t++) if (P->blk[k].A[t].nnz) nb[P->blk[k].con[t]]++;
    int nf = 0, fcap = 0, *fblk = NULL, *fns = NULL, **fsup = NULL, *fnr = NULL, **fri = NULL;
    double **fval = NULL, **frw = NULL;
    double work = 0;
    for (int k = 0; k < P->nblk && work < 2e9; k++) {
        const Block *B = &P->blk[k];
        if (B->type != BLK_SDP || B->n < 3) continue;
        const int n = B->n;
        double *g = calloc((size_t)n, sizeof(double));      /* Y_hh = g[h] > 0 fixed by a constraint of its own */
        int nfix = 0;
        for (int t = 0; t < B->ncon; t++) {
            const SpSym *S = &B->A[t]; const int i = B->con[t];
            if (nb[i] != 1 || S->nnz != 1 || S->row[0] != S->col[0] || P->b[i] == 0 || S->val[0] == 0) continue;
            const double gh = P->b[i] / S->val[0];
            if (gh > 0 && g[S->row[0]] == 0) { g[S->row[0]] = gh; nfix++; }
        }
        if (!nfix) { free(g); continue; }
        int *pos = malloc(sizeof(int) * (size_t)n); for (int j = 0; j < n; j++) pos[j] = -1;
        double *vd = calloc((size_t)n, sizeof(double));
        int *sup = malloc(sizeof(int) * (size_t)n);
        /* an orthonormal basis of the face vectors accepted in this block (dense, n <= 4000): a candidate in their span
         * adds nothing (Y v = 0 follows from the others), so at most n - 1 faces a block reach the facial reduction (a
         * Mittelmann problem with one block of order 569 has 9,900 candidates) */
        double *Qb = n <= 4000 ? malloc(sizeof(double) * (size_t)n * 64) : NULL, *wq = Qb ? malloc(sizeof(double) * (size_t)n) : NULL;
        int nq = 0, qcap = Qb ? 64 : 0;
        /* the constraints of this block alone by index (each listed under the indices of its entries), so that a
         * candidate visits only the constraints that meet its support (fap25: a scan of all 2,118 constraints for each
         * of its candidates took 1.9 s) */
        int *ip = calloc((size_t)n + 1, sizeof(int)), *il = NULL, *stamp = malloc(sizeof(int) * ((size_t)B->ncon + 1));
        for (int t = 0; t < B->ncon; t++) {
            stamp[t] = -1;
            const SpSym *A = &B->A[t];
            if (nb[B->con[t]] != 1) continue;
            for (int e = 0; e < A->nnz; e++) { ip[A->row[e] + 1]++; if (A->col[e] != A->row[e]) ip[A->col[e] + 1]++; }
        }
        for (int j = 0; j < n; j++) ip[j + 1] += ip[j];
        il = malloc(sizeof(int) * ((size_t)ip[n] + 1));
        {
            int *fp = malloc(sizeof(int) * ((size_t)n + 1)); memcpy(fp, ip, sizeof(int) * (size_t)n);
            for (int t = 0; t < B->ncon; t++) {
                const SpSym *A = &B->A[t];
                if (nb[B->con[t]] != 1) continue;
                for (int e = 0; e < A->nnz; e++) { il[fp[A->row[e]]++] = t; if (A->col[e] != A->row[e]) il[fp[A->col[e]]++] = t; }
            }
            free(fp);
        }
        work += (double)ip[n];
        for (int t0 = 0; t0 < B->ncon && work < 2e9 && !(Qb && nq >= n - 1); t0++) {
            const SpSym *S = &B->A[t0]; const int i0 = B->con[t0];
            if (nb[i0] != 1 || S->nnz < 1) continue;
            int h = -1;                                      /* every entry on row h of a fixed diagonal, one off it at least */
            for (int side = 0; side < 2 && h < 0; side++) {
                int hh = -1;
                for (int e = 0; e < S->nnz; e++) if (S->row[e] != S->col[e]) { hh = side ? S->col[e] : S->row[e]; break; }
                if (hh < 0 || g[hh] == 0) continue;
                int ok = 1;
                for (int e = 0; e < S->nnz && ok; e++) if (S->row[e] != hh && S->col[e] != hh) ok = 0;
                if (ok) h = hh;
            }
            if (h < 0) continue;
            double ahh = 0; int s = 0;
            sup[s] = h; pos[h] = s++;
            for (int e = 0; e < S->nnz; e++) {
                if (S->row[e] == S->col[e]) { ahh += S->val[e]; continue; }
                const int j = S->row[e] == h ? S->col[e] : S->row[e];
                if (pos[j] < 0) { sup[s] = j; pos[j] = s++; }
                vd[j] += 2.0 * S->val[e];
            }
            vd[h] = -(P->b[i0] - ahh * g[h]) / g[h];
            if (Qb) {                                        /* the part of v outside the span of the accepted faces */
                double vn = 0; for (int q = 0; q < s; q++) vn += vd[sup[q]] * vd[sup[q]];
                /* first |w|^2 = |v|^2 - |Q'v|^2 from the support of v alone (most candidates lie in the span: nq s work
                 * instead of 2 nq n); only a candidate that passes is projected densely */
                double s2 = 0;
                for (int c = 0; c < nq; c++) { const double *qc = Qb + (size_t)c * n; double dt = 0; for (int q = 0; q < s; q++) dt += qc[sup[q]] * vd[sup[q]]; s2 += dt * dt; }
                work += (double)nq * s;
                if (!(vn - s2 > 1e-12 * vn)) { for (int q = 0; q < s; q++) { vd[sup[q]] = 0; pos[sup[q]] = -1; } continue; }
                memset(wq, 0, sizeof(double) * (size_t)n);
                for (int q = 0; q < s; q++) wq[sup[q]] = vd[sup[q]];
                for (int pass = 0; pass < 2; pass++)
                    for (int c = 0; c < nq; c++) {
                        const double *qc = Qb + (size_t)c * n; double dt = 0;
                        for (int j = 0; j < n; j++) dt += qc[j] * wq[j];
                        for (int j = 0; j < n; j++) wq[j] -= dt * qc[j];
                    }
                double wn = 0; for (int j = 0; j < n; j++) wn += wq[j] * wq[j];
                work += 2.0 * nq * n;
                if (!(wn > 1e-18 * vn)) { for (int q = 0; q < s; q++) { vd[sup[q]] = 0; pos[sup[q]] = -1; } continue; }
            }
            const long npos = (long)s * (s + 1) / 2;
            /* the constraints of the block with every entry in supp(v) x supp(v) (and no entries in other blocks), from
             * the index; and whether they cover every nonzero of v v' (if one is left out, no combination exists) */
            int nr = 0, *rt = malloc(sizeof(int) * ((size_t)B->ncon + 1));
            double visits = 0;
            for (int q = 0; q < s; q++) {
                const int j = sup[q];
                visits += ip[j + 1] - ip[j];
                for (int u = ip[j]; u < ip[j + 1]; u++) {
                    const int t = il[u];
                    if (stamp[t] == t0) continue;
                    stamp[t] = t0;
                    const SpSym *A = &B->A[t];
                    int in = A->nnz > 0;
                    for (int e = 0; e < A->nnz && in; e++) if (pos[A->row[e]] < 0 || pos[A->col[e]] < 0) in = 0;
                    visits += A->nnz;
                    if (in) rt[nr++] = t;
                }
            }
            int covered = nr > 0;
            if (covered) {
                char *cov = calloc((size_t)npos, 1);
                for (int a = 0; a < nr; a++) { const SpSym *A = &B->A[rt[a]]; for (int e = 0; e < A->nnz; e++) { const int r = pos[A->row[e]], c = pos[A->col[e]]; cov[r <= c ? (long)c * (c + 1) / 2 + r : (long)r * (r + 1) / 2 + c] = 1; } }
                for (int c = 0; c < s && covered; c++) for (int r = 0; r <= c && covered; r++)
                    if (vd[sup[r]] * vd[sup[c]] != 0 && !cov[(long)c * (c + 1) / 2 + r]) covered = 0;
                free(cov);
            }
            work += visits + (covered ? (double)nr * npos : (double)npos);
            int exact = 0;
            double *Ah = NULL, *G = NULL, *d = NULL, *rhs = NULL, *res = NULL, *tv = NULL;
            if (covered && nr <= 4000 && (double)nr * npos <= 2e7) {
                /* A^ (nr x npos): <A_a, M> = A^_a . vec(M) with vec(M)_p = M_rc (r == c), sqrt2 M_rc (r < c), p over the
                 * pairs of supp(v) in the order of their positions */
                Ah = calloc((size_t)nr * npos, sizeof(double));
                #define LF_P(a, b) ((a) <= (b) ? (long)(b) * ((b) + 1) / 2 + (a) : (long)(a) * ((a) + 1) / 2 + (b))
                for (int a = 0; a < nr; a++) {
                    const SpSym *A = &B->A[rt[a]];
                    for (int e = 0; e < A->nnz; e++) {
                        const int r = pos[A->row[e]], c = pos[A->col[e]];
                        Ah[a + (size_t)nr * LF_P(r, c)] += r == c ? A->val[e] : M_SQRT2 * A->val[e];
                    }
                }
                tv = malloc(sizeof(double) * (size_t)npos);
                for (int c = 0; c < s; c++) for (int r = 0; r <= c; r++)
                    tv[LF_P(r, c)] = r == c ? vd[sup[r]] * vd[sup[c]] : M_SQRT2 * vd[sup[r]] * vd[sup[c]];
                G = malloc(sizeof(double) * (size_t)nr * nr);
                const double one = 1.0, zero = 0.0, mone = -1.0; const int inc = 1, np_ = (int)npos;
                BL(dsyrk_)("L", "N", &nr, &np_, &one, Ah, &nr, &zero, G, &nr);
                int info = 0;
                BL(dpotrf_)("L", &nr, G, &nr, &info);
                if (info == 0) {
                    d = calloc((size_t)nr, sizeof(double)); rhs = malloc(sizeof(double) * (size_t)nr); res = malloc(sizeof(double) * (size_t)npos);
                    double vv = 0; for (int q = 0; q < s; q++) vv += vd[sup[q]] * vd[sup[q]];
                    double rn = 0;
                    for (int it = 0; it < 3; it++) {
                        memcpy(res, tv, sizeof(double) * (size_t)npos);            /* res = vec(v v') - A^' d */
                        BL(dgemv_)("T", &nr, &np_, &mone, Ah, &nr, d, &inc, &one, res, &inc);
                        rn = 0; for (long p = 0; p < npos; p++) rn += res[p] * res[p];
                        rn = sqrt(rn);
                        if (it == 2) break;
                        BL(dgemv_)("N", &nr, &np_, &one, Ah, &nr, res, &inc, &zero, rhs, &inc);
                        BL(dpotrs_)("L", &nr, &inc, G, &nr, rhs, &nr, &info);
                        for (int a = 0; a < nr; a++) d[a] += rhs[a];
                    }
                    double bd = 0, bs = 0;
                    for (int a = 0; a < nr; a++) { const double x = P->b[B->con[rt[a]]] * d[a]; bd += x; bs += fabs(x); }
                    int origok = 1;
                    for (int a = 0; a < nr; a++) if (P->orig[B->con[rt[a]]] < 0) origok = 0;
                    exact = origok && vv > 0 && rn <= 1e-11 * vv && fabs(bd) <= 1e-11 * fmax(bs, vv);
                    /* (the floating-point test above only selects candidates; the face is accepted on its exact proof) */
                    const int near = exact;
                    if (exact && !(getenv("BRISK_LFNOPROOF") && atoi(getenv("BRISK_LFNOPROOF")))) exact = lf_exact_face(P, B, rt, nr, pos, sup, s, h, d);
                    if (near && !exact) g_lf_near++;
                    if (verbose >= 2) printf("   lifted face candidate: block %d, row %d, |supp| %d, %d constraints inside: residual %.1e of |v|^2, b'd %.1e of %.1e%s\n",
                                             k, h, s, nr, rn / vv, fabs(bd), fmax(bs, vv), exact ? " (exact)" : "");
                }
            }
            if (exact) {
                if (nf == fcap) {
                    fcap = fcap ? 2 * fcap : 16;
                    fblk = realloc(fblk, sizeof(int) * fcap); fns = realloc(fns, sizeof(int) * fcap); fnr = realloc(fnr, sizeof(int) * fcap);
                    fsup = realloc(fsup, sizeof(int *) * fcap); fval = realloc(fval, sizeof(double *) * fcap);
                    fri = realloc(fri, sizeof(int *) * fcap); frw = realloc(frw, sizeof(double *) * fcap);
                }
                fblk[nf] = k; fns[nf] = s; fnr[nf] = nr;
                fsup[nf] = malloc(sizeof(int) * s); fval[nf] = malloc(sizeof(double) * s);
                for (int q = 0; q < s; q++) { fsup[nf][q] = sup[q]; fval[nf][q] = vd[sup[q]]; }
                fri[nf] = malloc(sizeof(int) * nr); frw[nf] = malloc(sizeof(double) * nr);
                for (int a = 0; a < nr; a++) { fri[nf][a] = B->con[rt[a]]; frw[nf][a] = d[a]; }
                nf++;
                if (Qb) {                                    /* (wq: v without its part in the span, from the test above) */
                    if (nq == qcap) { qcap *= 2; Qb = realloc(Qb, sizeof(double) * (size_t)n * qcap); }
                    double wn = 0; for (int j = 0; j < n; j++) wn += wq[j] * wq[j];
                    wn = sqrt(wn);
                    for (int j = 0; j < n; j++) Qb[(size_t)nq * n + j] = wq[j] / wn;
                    nq++;
                }
            }
            free(Ah); free(G); free(d); free(rhs); free(res); free(tv); free(rt);
            for (int q = 0; q < s; q++) { vd[sup[q]] = 0; pos[sup[q]] = -1; }
        }
        free(g); free(pos); free(vd); free(sup); free(Qb); free(wq); free(ip); free(il); free(stamp);
    }
    free(nb);
    g_lf_time = wtime() - tlf0;
    if (nf == 0) return 0;
    /* the constraints <v v', Y> = 0, appended; their combinations in the postsolve's table (the file's rows) */
    Postsolve *ps = P->ps;
    const int m1 = m0 + nf, c0 = ps->ncomb;
    P->b = realloc(P->b, sizeof(double) * m1); P->b0 = realloc(P->b0, sizeof(double) * m1); P->orig = realloc(P->orig, sizeof(int) * m1);
    int tot = ps->ncomb ? ps->comb_ptr[ps->ncomb] : 0;
    for (int f = 0; f < nf; f++) tot += fnr[f];
    ps->comb_ptr = realloc(ps->comb_ptr, sizeof(int) * (c0 + nf + 1)); if (c0 == 0) ps->comb_ptr[0] = 0;
    ps->comb_i = realloc(ps->comb_i, sizeof(int) * (tot + 1)); ps->comb_w = realloc(ps->comb_w, sizeof(double) * (tot + 1));
    for (int f = 0; f < nf; f++) {
        const int i = m0 + f, j = c0 + f, s = fns[f];
        P->b[i] = 0; P->b0[i] = 0; P->orig[i] = -2 - j;
        int w = ps->comb_ptr[j];
        for (int a = 0; a < fnr[f]; a++) { ps->comb_i[w] = P->orig[fri[f][a]]; ps->comb_w[w] = frw[f][a]; w++; }
        ps->comb_ptr[j + 1] = w;
        Block *B = &P->blk[fblk[f]];
        B->con = realloc(B->con, sizeof(int) * (B->ncon + 1)); B->A = realloc(B->A, sizeof(SpSym) * (B->ncon + 1));
        memset(&B->A[B->ncon], 0, sizeof(SpSym));
        Ent *E = malloc(sizeof(Ent) * ((size_t)s * (s + 1) / 2 + 1)); size_t ne = 0;
        for (int a = 0; a < s; a++) for (int b2 = a; b2 < s; b2++) { E[ne].r = fsup[f][a]; E[ne].c = fsup[f][b2]; E[ne].v = fval[f][a] * fval[f][b2]; ne++; }
        sp_from_entries(&B->A[B->ncon], E, ne, B->n, 0);
        free(E);
        B->con[B->ncon++] = i;
        free(fsup[f]); free(fval[f]); free(fri[f]); free(frw[f]);
    }
    ps->ncomb = c0 + nf;
    P->m = m1;
    free(fblk); free(fns); free(fnr); free(fsup); free(fval); free(fri); free(frw);
    if (verbose >= 0) printf("presolve: %d face%s of a lifted relaxation found from its linear constraints: <v v', X> = 0 added as exact combinations of the constraints, for the facial reduction\n", nf, nf > 1 ? "s" : "");
    return nf;
}

/* ---------------------------------------------------------------------- */
/* 3 November 2026 (ISSUES 745, the owner: "face detecting only numerically is fine for that problem class (but stay with
 * exact arithmetic for other problem classes)"): faces fixed by small index sets. For a set S of indices of a semidefinite
 * block, take the constraints (of this block alone) whose entries lie in S x S. If their matrices span all of
 * { Q W Q' : W symmetric } for the orthonormal basis Q (k x t) of their joint column space (the test: the rank of the
 * constraints' vectors is t(t+1)/2), the data fix G = Q' X_S Q for every feasible X; a null vector n of G gives v = Q n
 * with v' X v = 0, hence X v = 0. Squared distances on a clique of k >= d + 2 points in R^d are one instance (their span
 * is { Y : Y e = 0 }, G the centred Gram matrix, of rank d): the facial reduction of sensor network localization and
 * maximum variance unfolding (Krislock and Wolkowicz, 2010), found clique by clique. The sets are the maximal cliques
 * (Bron-Kerbosch with a pivot, at most CF_KMAX indices) of the graph whose edges join the indices of a constraint with at
 * most CF_KMAX indices. The test is numerical: the ranks across a gap (1e-10 relative), and n a null vector when its
 * eigenvalue is below CF_TOL of G's largest (data rounded to 16 digits leave swissroll's G singular only to 5e-14), so the
 * face is proven up to that tolerance, not exactly (the other faces of this file are). Each v v' is a combination of the
 * set's constraints (the least-norm weights; b'd = n'G n, set to 0 within the tolerance); their sum E = sum v v' becomes
 * one constraint <E, X> = 0, a combination of the file's constraints (the postsolve's table, as lifted_faces), which the
 * facial reduction removes by its numerical eigen test (kind 5: rank across a gap) and the dense face basis, the only
 * constraint that test is applied to. Blocks of order 4 to 3,000. Off by default while it is judged (BRISK_CLIQUEFACE=1: on);
 * BRISK_CFTOL: the tolerance. */
#define CF_KMAX 10
typedef struct {
    const uint64_t *adj; int words;
    int kmax; long work, budget;
    int *R, nR;
    int ncl, clcap, *clp, *cli;      /* the cliques found (CSR) */
} CfBK;
static inline int cf_adj(const CfBK *b, int u, int v) { return (int)((b->adj[(size_t)u * b->words + (v >> 6)] >> (v & 63)) & 1); }
static void cf_emit(CfBK *b) {
    if (b->nR < 3) return;
    if (b->ncl + 1 >= b->clcap) { b->clcap = 2 * b->clcap + 64; b->clp = realloc(b->clp, sizeof(int) * (b->clcap + 1)); }
    const int base = b->clp[b->ncl];
    b->cli = realloc(b->cli, sizeof(int) * (base + b->nR + 1));
    for (int q = 0; q < b->nR; q++) b->cli[base + q] = b->R[q];
    b->clp[++b->ncl] = base + b->nR;
}
static void cf_bk(CfBK *b, const int *Pv, int np, const int *Xv, int nx) {
    if (b->work > b->budget) return;
    if (np == 0) { if (nx == 0) cf_emit(b); return; }
    if (b->nR == b->kmax) { cf_emit(b); return; }        /* (a clique of the largest size kept; not extended) */
    int u = -1, best = -1;                                /* the pivot: the most neighbours among the candidates */
    for (int q = 0; q < np + nx; q++) {
        const int w = q < np ? Pv[q] : Xv[q - np];
        int c = 0;
        for (int r = 0; r < np; r++) c += cf_adj(b, w, Pv[r]);
        b->work += np;
        if (c > best) { best = c; u = w; }
    }
    int *P2 = malloc(sizeof(int) * (np + 1)), *X2 = malloc(sizeof(int) * (np + nx + 1));
    int *Pc = malloc(sizeof(int) * (np + 1)), *Xc = malloc(sizeof(int) * (np + nx + 1));
    memcpy(Pc, Pv, sizeof(int) * np); memcpy(Xc, Xv, sizeof(int) * nx);
    int npc = np, nxc = nx;
    for (int q = 0; q < np; q++) {
        const int v = Pv[q];
        if (cf_adj(b, u, v)) continue;
        int n2 = 0, m2 = 0;
        for (int r = 0; r < npc; r++) if (Pc[r] != v && cf_adj(b, v, Pc[r])) P2[n2++] = Pc[r];
        for (int r = 0; r < nxc; r++) if (cf_adj(b, v, Xc[r])) X2[m2++] = Xc[r];
        b->work += npc + nxc;
        b->R[b->nR++] = v;
        cf_bk(b, P2, n2, X2, m2);
        b->nR--;
        for (int r = 0; r < npc; r++) if (Pc[r] == v) { Pc[r] = Pc[--npc]; break; }
        Xc[nxc++] = v;
        if (b->work > b->budget) break;
    }
    free(P2); free(X2); free(Pc); free(Xc);
}
/* eigen-decomposition of the symmetric k x k matrix A (column-major, overwritten by the eigenvectors), ascending */
static int cf_eig(int k, double *A, double *w) {
    int lwork = -1, info = 0; double wq;
    BL(dsyev_)("V", "U", &k, A, &k, w, &wq, &lwork, &info);
    lwork = (int)wq + 1;
    double *work = malloc(sizeof(double) * lwork);
    BL(dsyev_)("V", "U", &k, A, &k, w, work, &lwork, &info);
    free(work);
    return info == 0;
}
int g_cf_vectors = 0, g_cf_sets = 0;   /* (the last detection's face vectors and fixed sets, for the log) */
int clique_faces(Problem *P, int verbose) {
    static int off = -1; if (off < 0) { const char *e = getenv("BRISK_CLIQUEFACE"); off = !(e && atoi(e) != 0); }   /* (off by default until judged on the sets; ISSUES 745) */
    static double tol = -1; if (tol < 0) tol = getenv("BRISK_CFTOL") ? atof(getenv("BRISK_CFTOL")) : 1e-9;
    g_cf_vectors = g_cf_sets = 0;
    if (off || !P->ps || !P->orig || P->m < 3) return 0;
    const double t0 = wtime();
    int *nb = calloc((size_t)P->m + 1, sizeof(int));
    for (int k = 0; k < P->nblk; k++) for (int t = 0; t < P->blk[k].ncon; t++) if (P->blk[k].A[t].nnz) nb[P->blk[k].con[t]]++;
    int nadd = 0;
    for (int k = 0; k < P->nblk; k++) {
        const Block *B = &P->blk[k];
        const int n = B->n;
        if (B->type != BLK_SDP || n < 4 || n > 3000) continue;
        /* the small constraints of this block alone, listed under their indices */
        int *ip = calloc((size_t)n + 1, sizeof(int)), nsmall = 0;
        for (int t = 0; t < B->ncon; t++) {
            const SpSym *A = &B->A[t]; const int i = B->con[t];
            if (nb[i] != 1 || A->nnz == 0 || A->nr > CF_KMAX || P->orig[i] < 0) continue;
            for (int q = 0; q < A->nr; q++) ip[A->rows[q] + 1]++;
            nsmall++;
        }
        if (nsmall < 3) { free(ip); continue; }
        for (int j = 0; j < n; j++) ip[j + 1] += ip[j];
        int *il = malloc(sizeof(int) * ((size_t)ip[n] + 1));
        {
            int *fp = malloc(sizeof(int) * ((size_t)n + 1)); memcpy(fp, ip, sizeof(int) * (size_t)n);
            for (int t = 0; t < B->ncon; t++) {
                const SpSym *A = &B->A[t]; const int i = B->con[t];
                if (nb[i] != 1 || A->nnz == 0 || A->nr > CF_KMAX || P->orig[i] < 0) continue;
                for (int q = 0; q < A->nr; q++) il[fp[A->rows[q]]++] = t;
            }
            free(fp);
        }
        /* the graph: the indices of each small constraint pairwise adjacent (a bit matrix) */
        const int words = (n + 63) / 64;
        uint64_t *adj = calloc((size_t)n * words, sizeof(uint64_t));
        long nedge = 0;
        for (int t = 0; t < B->ncon; t++) {
            const SpSym *A = &B->A[t]; const int i = B->con[t];
            if (nb[i] != 1 || A->nnz == 0 || A->nr > CF_KMAX || P->orig[i] < 0) continue;
            for (int a = 0; a < A->nr; a++) for (int c = 0; c < A->nr; c++) if (a != c) {
                const int u = A->rows[a], v = A->rows[c];
                uint64_t *wd = &adj[(size_t)u * words + (v >> 6)];
                if (!((*wd >> (v & 63)) & 1)) { *wd |= 1ULL << (v & 63); nedge++; }
            }
        }
        if (nedge == 0) { free(adj); free(ip); free(il); continue; }
        /* the maximal cliques, each vertex with its later neighbours as candidates and the earlier as excluded */
        CfBK bk; memset(&bk, 0, sizeof(bk));
        bk.adj = adj; bk.words = words; bk.kmax = CF_KMAX; bk.budget = 200L * ip[n] + 4000000L;
        bk.R = malloc(sizeof(int) * (CF_KMAX + 1)); bk.clp = malloc(sizeof(int) * 65); bk.clcap = 64; bk.clp[0] = 0;
        int *Pv = malloc(sizeof(int) * n), *Xv = malloc(sizeof(int) * n);
        for (int v = 0; v < n && bk.work <= bk.budget; v++) {
            int np = 0, nx = 0;
            for (int w = 0; w < n; w++) if (w != v && cf_adj(&bk, v, w)) { if (w > v) Pv[np++] = w; else Xv[nx++] = w; }
            bk.work += words;
            bk.R[0] = v; bk.nR = 1;
            cf_bk(&bk, Pv, np, Xv, nx);
        }
        free(Pv); free(Xv); free(adj);
        if (bk.work > bk.budget) {
            if (verbose >= 2) printf("presolve: faces fixed by small index sets: the clique search's budget ran out (block of order %d)\n", n);
            free(bk.R); free(bk.clp); free(bk.cli); free(ip); free(il); continue;
        }
        /* each clique: the fixed compression test, its null vectors, their combinations */
        int *pos = malloc(sizeof(int) * n); for (int j = 0; j < n; j++) pos[j] = -1;
        int *stamp = malloc(sizeof(int) * ((size_t)B->ncon + 1)); for (int t = 0; t < B->ncon; t++) stamp[t] = -1;
        int *ct = malloc(sizeof(int) * ((size_t)B->ncon + 1));
        double *E = calloc((size_t)n * n, sizeof(double));     /* sum v v' (upper triangle used) */
        double *wsum = calloc((size_t)B->ncon + 1, sizeof(double));
        double *Qb = malloc(sizeof(double) * (size_t)n * 64), *wq = malloc(sizeof(double) * n);
        int nq = 0, qcap = 64, nvec = 0, nfix = 0;
        double bdmax = 0;
        const int NP = CF_KMAX * (CF_KMAX + 1) / 2;
        double *Mm = malloc(sizeof(double) * (size_t)NP * 64), *MtM = malloc(sizeof(double) * NP * NP), *lam = malloc(sizeof(double) * NP);
        int mcap = 64;
        double *H = malloc(sizeof(double) * CF_KMAX * CF_KMAX), *hw = malloc(sizeof(double) * CF_KMAX);
        double *Ad = malloc(sizeof(double) * CF_KMAX * CF_KMAX), *G = malloc(sizeof(double) * CF_KMAX * CF_KMAX), *gw = malloc(sizeof(double) * CF_KMAX);
        double *x = malloc(sizeof(double) * NP), *f = malloc(sizeof(double) * NP), *z = malloc(sizeof(double) * NP), *bb = malloc(sizeof(double) * 64);
        double *XS = malloc(sizeof(double) * CF_KMAX * CF_KMAX), *QX = malloc(sizeof(double) * CF_KMAX * CF_KMAX), *v = malloc(sizeof(double) * CF_KMAX), *d = malloc(sizeof(double) * 64);
        for (int c = 0; c < bk.ncl && nq < n - 1; c++) {
            const int *S = bk.cli + bk.clp[c], ks = bk.clp[c + 1] - bk.clp[c];
            for (int q = 0; q < ks; q++) pos[S[q]] = q;
            int nc = 0;
            for (int q = 0; q < ks; q++) for (int u = ip[S[q]]; u < ip[S[q] + 1]; u++) {
                const int t = il[u];
                if (stamp[t] == c) continue;
                stamp[t] = c;
                const SpSym *A = &B->A[t];
                int in = 1;
                for (int e = 0; e < A->nr && in; e++) if (pos[A->rows[e]] < 0) in = 0;
                if (in) ct[nc++] = t;
            }
            const int np_ = ks * (ks + 1) / 2;
            #define CF_P(r_, c_) ((r_) <= (c_) ? (c_) * ((c_) + 1) / 2 + (r_) : (r_) * ((r_) + 1) / 2 + (c_))
            if (nc < 3) goto next;
            if (nc > mcap) { mcap = nc; Mm = realloc(Mm, sizeof(double) * (size_t)NP * mcap); bb = realloc(bb, sizeof(double) * mcap); d = realloc(d, sizeof(double) * mcap); }
            /* M (nc x np_, row-major by constraint): <A, X> = M_a . vec(X), vec(X)_p = X_rr or sqrt2 X_rc */
            memset(Mm, 0, sizeof(double) * (size_t)nc * np_);
            memset(H, 0, sizeof(double) * ks * ks);
            for (int a = 0; a < nc; a++) {
                const SpSym *A = &B->A[ct[a]];
                memset(Ad, 0, sizeof(double) * ks * ks);
                for (int e = 0; e < A->nnz; e++) {
                    const int r = pos[A->row[e]], cc = pos[A->col[e]];
                    Mm[(size_t)a * np_ + CF_P(r, cc)] += r == cc ? A->val[e] : M_SQRT2 * A->val[e];
                    Ad[r + cc * ks] = A->val[e]; Ad[cc + r * ks] = A->val[e];
                }
                for (int i = 0; i < ks; i++) for (int j = 0; j < ks; j++) {      /* H += A A (the joint column space) */
                    double sm = 0; for (int l = 0; l < ks; l++) sm += Ad[i + l * ks] * Ad[l + j * ks];
                    H[i + j * ks] += sm;
                }
                bb[a] = P->b[B->con[ct[a]]];
            }
            for (int p = 0; p < np_; p++) for (int q = 0; q < np_; q++) {
                double sm = 0; for (int a = 0; a < nc; a++) sm += Mm[(size_t)a * np_ + p] * Mm[(size_t)a * np_ + q];
                MtM[p + q * np_] = sm;
            }
            if (!cf_eig(np_, MtM, lam) || !cf_eig(ks, H, hw)) goto next;
            {
                const double lmax = lam[np_ - 1], hmax = hw[ks - 1];
                if (!(lmax > 0) || !(hmax > 0)) goto next;
                int rM = 0, tq = 0;
                for (int p = 0; p < np_; p++) if (lam[p] > 1e-10 * lmax) rM++;
                for (int p = 0; p < ks; p++) if (hw[p] > 1e-10 * hmax) tq++;
                if (rM != tq * (tq + 1) / 2) goto next;
                /* the gaps: no eigenvalue between 1e-14 and 1e-10 of the largest (a rank read across a gap) */
                for (int p = 0; p < np_; p++) if (lam[p] > 1e-14 * lmax && lam[p] <= 1e-10 * lmax) goto next;
                for (int p = 0; p < ks; p++) if (hw[p] > 1e-14 * hmax && hw[p] <= 1e-10 * hmax) goto next;
                const double *Q = H + (size_t)(ks - tq) * ks;                  /* the top tq eigenvectors */
                /* x = M^+ b (least norm, through M'M = V L V' on its range), one refinement */
                #define CF_PINV(out_, in_) do { for (int p_ = 0; p_ < np_; p_++) { out_[p_] = 0; } \
                    for (int e_ = np_ - rM; e_ < np_; e_++) { const double *ve = MtM + (size_t)e_ * np_; double s_ = 0; \
                        for (int p_ = 0; p_ < np_; p_++) s_ += ve[p_] * in_[p_]; s_ /= lam[e_]; \
                        for (int p_ = 0; p_ < np_; p_++) out_[p_] += s_ * ve[p_]; } } while (0)
                for (int p = 0; p < np_; p++) { double sm = 0; for (int a = 0; a < nc; a++) sm += Mm[(size_t)a * np_ + p] * bb[a]; f[p] = sm; }
                CF_PINV(x, f);
                double rn = 0, bn = 0;
                for (int it = 0; it < 2; it++) {
                    rn = 0; bn = 0;
                    for (int p = 0; p < np_; p++) f[p] = 0;
                    for (int a = 0; a < nc; a++) {
                        double sm = 0; for (int p = 0; p < np_; p++) sm += Mm[(size_t)a * np_ + p] * x[p];
                        const double r = bb[a] - sm; rn += r * r; bn += bb[a] * bb[a];
                        for (int p = 0; p < np_; p++) f[p] += Mm[(size_t)a * np_ + p] * r;
                    }
                    if (it == 1) break;
                    CF_PINV(z, f);
                    for (int p = 0; p < np_; p++) x[p] += z[p];
                }
                if (rn > 1e-18 * fmax(bn, 1e-300)) goto next;                  /* the set's constraints are not consistent */
                for (int i = 0; i < ks; i++) for (int j = i; j < ks; j++) {
                    const double xv = i == j ? x[CF_P(i, j)] : x[CF_P(i, j)] / M_SQRT2;
                    XS[i + j * ks] = xv; XS[j + i * ks] = xv;
                }
                /* G = Q' X_S Q */
                for (int i = 0; i < ks; i++) for (int j = 0; j < tq; j++) { double sm = 0; for (int l = 0; l < ks; l++) sm += XS[i + l * ks] * Q[l + j * ks]; QX[i + j * ks] = sm; }
                for (int i = 0; i < tq; i++) for (int j = 0; j < tq; j++) { double sm = 0; for (int l = 0; l < ks; l++) sm += Q[l + i * ks] * QX[l + j * ks]; G[i + j * tq] = sm; }
                if (!cf_eig(tq, G, gw)) goto next;
                const double gs = fmax(fabs(gw[0]), fabs(gw[tq - 1]));
                if (!(gs > 0) || gw[0] < -1e-6 * gs) goto next;                /* (not semidefinite beyond rounding: no claim) */
                nfix++;
                for (int p = 0; p < tq && gw[p] <= tol * gs; p++) {
                    for (int i = 0; i < ks; i++) { double sm = 0; for (int j = 0; j < tq; j++) sm += Q[i + j * ks] * G[j + p * tq]; v[i] = sm; }
                    /* the weights: sum d_a A_a = v v' (least norm, d = M z with M'M z = vec(v v')) */
                    for (int i = 0; i < ks; i++) for (int j = i; j < ks; j++) f[CF_P(i, j)] = i == j ? v[i] * v[j] : M_SQRT2 * v[i] * v[j];
                    CF_PINV(z, f);
                    double res2 = 0, bd = 0, bs = 0;
                    for (int a = 0; a < nc; a++) { double sm = 0; for (int p2 = 0; p2 < np_; p2++) sm += Mm[(size_t)a * np_ + p2] * z[p2]; d[a] = sm; bd += bb[a] * sm; bs += fabs(bb[a] * sm); }
                    for (int p2 = 0; p2 < np_; p2++) { double sm = 0; for (int a = 0; a < nc; a++) sm += Mm[(size_t)a * np_ + p2] * d[a]; res2 += (sm - f[p2]) * (sm - f[p2]); }
                    if (res2 > 1e-20) continue;                                 /* v v' is not their combination */
                    /* every certified vector goes into E (the dual certificate: all of them together are far better
                     * conditioned than the subset that spans, swissroll 1.5e-7 against 1.3e-11 of the largest eigenvalue) */
                    bdmax = fmax(bdmax, fabs(bd) / fmax(bs, 1e-300));
                    for (int a = 0; a < nc; a++) wsum[ct[a]] += d[a];
                    for (int i = 0; i < ks; i++) for (int j = 0; j < ks; j++) {
                        const int r = S[i], cc = S[j];
                        if (r <= cc) E[r + (size_t)cc * n] += v[i] * v[j];
                    }
                    nvec++;
                    /* the face's basis: the part of v outside the span of the vectors accepted (Gram-Schmidt twice); a
                     * new direction when it is above 1e-6 (|v| = 1; swissroll: at 1e-10 a 637th direction of rounding
                     * size cuts the feasible set, the dropped constraints' right-hand sides then off by 4.9e-2) */
                    memset(wq, 0, sizeof(double) * n);
                    for (int i = 0; i < ks; i++) wq[S[i]] = v[i];
                    for (int pass = 0; pass < 2; pass++)
                        for (int cq = 0; cq < nq; cq++) {
                            const double *qc = Qb + (size_t)cq * n; double dt = 0;
                            for (int j = 0; j < n; j++) dt += qc[j] * wq[j];
                            for (int j = 0; j < n; j++) wq[j] -= dt * qc[j];
                        }
                    double wn = 0; for (int j = 0; j < n; j++) wn += wq[j] * wq[j];
                    if (!(wn > 1e-12) || nq >= n - 1) continue;
                    if (nq == qcap) { qcap *= 2; Qb = realloc(Qb, sizeof(double) * (size_t)n * qcap); }
                    wn = sqrt(wn);
                    for (int j = 0; j < n; j++) Qb[(size_t)nq * n + j] = wq[j] / wn;
                    nq++;
                }
                #undef CF_PINV
            }
        next:
            #undef CF_P
            for (int q = 0; q < ks; q++) pos[S[q]] = -1;
        }
        free(Mm); free(MtM); free(lam); free(H); free(hw); free(Ad); free(G); free(gw); free(x); free(f); free(z); free(bb);
        free(XS); free(QX); free(v); free(d); free(wq); free(pos); free(stamp); free(ct);
        free(bk.R); free(bk.clp); free(bk.cli); free(ip); free(il);
        if (nvec == 0) { free(E); free(wsum); free(Qb); continue; }
        /* the constraint <E, X> = 0, a combination of the file's constraints: appended, with its row in the table */
        Postsolve *ps = P->ps;
        const int i = P->m, jc = ps->ncomb;
        int nw = 0; for (int t = 0; t < B->ncon; t++) if (wsum[t] != 0) nw++;
        P->b = realloc(P->b, sizeof(double) * (P->m + 1)); P->b0 = realloc(P->b0, sizeof(double) * (P->m + 1)); P->orig = realloc(P->orig, sizeof(int) * (P->m + 1));
        P->b[i] = 0; P->b0[i] = 0; P->orig[i] = -2 - jc;
        int tot = jc ? ps->comb_ptr[jc] : 0;
        ps->comb_ptr = realloc(ps->comb_ptr, sizeof(int) * (jc + 2)); if (jc == 0) ps->comb_ptr[0] = 0;
        ps->comb_i = realloc(ps->comb_i, sizeof(int) * (tot + nw + 1)); ps->comb_w = realloc(ps->comb_w, sizeof(double) * (tot + nw + 1));
        for (int t = 0; t < B->ncon; t++) if (wsum[t] != 0) { ps->comb_i[tot] = P->orig[B->con[t]]; ps->comb_w[tot] = wsum[t]; tot++; }
        ps->comb_ptr[jc + 1] = tot;
        ps->comb_num = realloc(ps->comb_num, jc + 1);
        ps->comb_fn = realloc(ps->comb_fn, sizeof(int) * (jc + 1)); ps->comb_fk = realloc(ps->comb_fk, sizeof(int) * (jc + 1));
        ps->comb_fN = realloc(ps->comb_fN, sizeof(double *) * (jc + 1)); ps->comb_fU = realloc(ps->comb_fU, sizeof(double *) * (jc + 1));
        for (int q = ps->ncomb_num; q < jc; q++) { ps->comb_num[q] = 0; ps->comb_fn[q] = ps->comb_fk[q] = 0; ps->comb_fN[q] = ps->comb_fU[q] = NULL; }
        ps->comb_num[jc] = 1; ps->ncomb_num = jc + 1;
        {   /* the face: the complement of the span of the accepted vectors (Qb, orthonormal by Gram-Schmidt twice), the
             * eigenvectors of I - Qb Qb' for the eigenvalue 1 (eigenvalues 0 and 1: accurate to rounding) */
            double *Pc = malloc(sizeof(double) * (size_t)n * n), *ev = malloc(sizeof(double) * n);
            const double mone = -1.0, one = 1.0;
            for (size_t q = 0; q < (size_t)n * n; q++) Pc[q] = 0;
            for (int j = 0; j < n; j++) Pc[j + (size_t)j * n] = 1.0;
            BL(dgemm_)("N", "T", &n, &n, &nq, &mone, Qb, &n, Qb, &n, &one, Pc, &n);
            int kf = 0;
            if (cf_eig(n, Pc, ev)) for (int j = 0; j < n; j++) if (ev[j] > 0.5) kf++;
            if (kf == n - nq) {
                ps->comb_fn[jc] = n; ps->comb_fk[jc] = kf;
                ps->comb_fN[jc] = malloc(sizeof(double) * (size_t)n * (kf ? kf : 1));
                memcpy(ps->comb_fN[jc], Pc + (size_t)(n - kf) * n, sizeof(double) * (size_t)n * kf);
                ps->comb_fU[jc] = malloc(sizeof(double) * (size_t)n * nq);
                memcpy(ps->comb_fU[jc], Qb, sizeof(double) * (size_t)n * nq);
            } else { ps->comb_fn[jc] = ps->comb_fk[jc] = 0; ps->comb_fN[jc] = ps->comb_fU[jc] = NULL; }
            free(Pc); free(ev);
        }
        ps->ncomb = jc + 1;
        Block *Bw = &P->blk[k];
        Bw->con = realloc(Bw->con, sizeof(int) * (Bw->ncon + 1)); Bw->A = realloc(Bw->A, sizeof(SpSym) * (Bw->ncon + 1));
        memset(&Bw->A[Bw->ncon], 0, sizeof(SpSym));
        size_t ne = 0;
        for (size_t q = 0; q < (size_t)n * n; q++) if (E[q] != 0) ne++;
        Ent *En = malloc(sizeof(Ent) * (ne + 1)); ne = 0;
        for (int cc = 0; cc < n; cc++) for (int r = 0; r <= cc; r++) if (E[r + (size_t)cc * n] != 0) { En[ne].r = r; En[ne].c = cc; En[ne].v = E[r + (size_t)cc * n]; ne++; }
        sp_from_entries(&Bw->A[Bw->ncon], En, ne, n, 0);
        free(En);
        Bw->con[Bw->ncon++] = i;
        P->m++;
        nadd++;
        g_cf_vectors += nvec; g_cf_sets += nfix;
        if (verbose >= 1) printf("presolve: %d face vector%s (spanning %d dimensions) from %d index sets whose constraints fix their compressed block (block of order %d; null vectors below %.0e of the largest eigenvalue, a numerical test; largest |b'd| %.1e of its terms): <E, X> = 0 added as a combination of the constraints, for the facial reduction (%.2fs)\n",
                                 nvec, nvec > 1 ? "s" : "", nq, nfix, n, tol, bdmax, wtime() - t0);
        free(E); free(wsum); free(Qb);
    }
    free(nb);
    return nadd;
}

/* 3 November 2026 (ISSUES 745): on the face X = N W N' of a whole block (N: n x k, orthonormal), the block's other
 * constraints <N'A N, W> = b can depend on each other (swissroll: 642 of 3,379 independent on the face of order 163), and
 * once compressed their matrices are dense (k(k+1)/2 entries each), which the general check cannot afford. Their Gram
 * matrix on the face comes from the sparse matrices as read and the projector P = N N': <N'A N, N'B N> = tr(A P B P) =
 * sum over the entries (i,j) of A and (k,l) of B of A_ij B_kl P_jk P_li. A pivoted Cholesky (the largest remaining
 * diagonal relative to its own first: the squared sine of the angle to the span of those taken) stops when every
 * remaining relative pivot is below 1e-10 (swissroll's pivots fall without a gap, 1.6e-8, 1.6e-10, 6.1e-11, ... 4e-13: the
 * constraints on the face are ill-conditioned, not only dependent); the right-hand sides of the rest must follow the same
 * combinations (b_a - L_a L_K^{-1} b_K within 1e-8 of the scale of b), or the face is not applied (-1). The dependent constraints are listed in *drop (block parts), to be removed with zero duals. Only
 * when every constraint of the block has no part in another block (cbp); returns 0 otherwise. */
static int face_dependent_rows(const Problem *P, const Block *B, int tskip, const int *S, int s, const double *N, int k,
                               const int *cbp, const char *dead, int **drop, int verbose) {
    *drop = NULL;
    const int n = B->n;
    if (s != n || k <= 0) return 0;
    for (int q = 0; q < s; q++) if (S[q] != q) return 0;
    int *ti = malloc(sizeof(int) * ((size_t)B->ncon + 1)), m = 0;
    for (int t = 0; t < B->ncon; t++) {
        if (t == tskip) continue;
        const int i = B->con[t];
        if (dead[i]) continue;
        if (cbp[i + 1] - cbp[i] != 1) { free(ti); return 0; }
        if (B->A[t].ef) ti[m++] = t;
    }
    if (m < 2) { free(ti); return 0; }
    const double t0 = wtime();
    double *Pm = malloc(sizeof(double) * (size_t)n * n);
    { const double one = 1.0, zero = 0.0; BL(dgemm_)("N", "T", &n, &n, &k, &one, N, &n, N, &n, &zero, Pm, &n); }
    /* a constraint with many entries (those an earlier rank-one substitution filled: swissroll's rows on the pivot of
     * sum(X) = 0, 640,000 entries) gets Y = P A P once (n^3), and its products read sum_(k,l) B_kl Y_lk */
    int *yid = malloc(sizeof(int) * m), ny = 0;
    for (int a = 0; a < m; a++) yid[a] = B->A[ti[a]].ef > 4 * n ? ny++ : -1;
    if (ny > 64) { free(yid); free(Pm); free(ti); return 0; }
    double *Y = ny ? malloc(sizeof(double) * (size_t)ny * n * n) : NULL, *T = ny ? malloc(sizeof(double) * (size_t)n * n) : NULL;
    for (int a = 0; a < m; a++) if (yid[a] >= 0) {
        const SpSym *A = &B->A[ti[a]];
        memset(T, 0, sizeof(double) * (size_t)n * n);                 /* T = A P (column-major: T[i + c n]) */
        for (int e = 0; e < A->ef; e++) { const int i = A->fr[e], j = A->fc[e]; const double v = A->fv[e]; for (int c = 0; c < n; c++) T[i + (size_t)c * n] += v * Pm[j + (size_t)c * n]; }
        const double one = 1.0, zero = 0.0;
        BL(dgemm_)("N", "N", &n, &n, &n, &one, Pm, &n, T, &n, &zero, Y + (size_t)yid[a] * n * n, &n);
    }
    free(T);
    double *G = malloc(sizeof(double) * (size_t)m * m);
    #pragma omp parallel for schedule(dynamic, 16)
    for (int a = 0; a < m; a++) {
        const SpSym *A = &B->A[ti[a]];
        for (int b2 = a; b2 < m; b2++) {
            const SpSym *C = &B->A[ti[b2]];
            double g = 0;
            if (yid[a] >= 0 || yid[b2] >= 0) {
                const SpSym *R = yid[a] >= 0 ? C : A;
                const double *Yq = Y + (size_t)(yid[a] >= 0 ? yid[a] : yid[b2]) * n * n;
                for (int f = 0; f < R->ef; f++) g += R->fv[f] * Yq[R->fc[f] + (size_t)R->fr[f] * n];
            } else
                for (int e = 0; e < A->ef; e++) {
                    const int i = A->fr[e], j = A->fc[e]; const double av = A->fv[e];
                    const double *Pj = Pm + (size_t)j * n, *Pi = Pm + (size_t)i * n;
                    for (int f = 0; f < C->ef; f++) g += av * C->fv[f] * Pj[C->fr[f]] * Pi[C->fc[f]];
                }
            G[a + (size_t)b2 * m] = g; G[b2 + (size_t)a * m] = g;
        }
    }
    free(Y); free(yid);
    double *d0 = malloc(sizeof(double) * m), *dr = malloc(sizeof(double) * m);
    for (int a = 0; a < m; a++) { d0[a] = G[a + (size_t)a * m]; dr[a] = d0[a]; }
    int rcap = 64, r = 0, *piv = malloc(sizeof(int) * m);
    double *L = malloc(sizeof(double) * (size_t)m * rcap);
    char *isp = calloc(m, 1);
    double last = 1.0, stopmax = 0;
    int ok = 1;
    #define FDR_STEP(p_) do { \
        if (r == rcap) { rcap *= 2; L = realloc(L, sizeof(double) * (size_t)m * rcap); } \
        double *Lc = L + (size_t)r * m; const double sq = sqrt(dr[p_]); \
        for (int a = 0; a < m; a++) { double v = G[a + (size_t)(p_) * m]; \
            for (int q = 0; q < r; q++) v -= L[a + (size_t)q * m] * L[(p_) + (size_t)q * m]; Lc[a] = v / sq; } \
        for (int a = 0; a < m; a++) if (!isp[a]) dr[a] -= Lc[a] * Lc[a]; \
        isp[p_] = 1; piv[r++] = (p_); } while (0)
    for (;;) {
        int p = -1; double best = -1;
        for (int a = 0; a < m; a++) if (!isp[a] && d0[a] > 0 && dr[a] / d0[a] > best) { best = dr[a] / d0[a]; p = a; }
        if (p < 0) break;
        static double prel = -1; if (prel < 0) prel = getenv("BRISK_CFPIV") ? atof(getenv("BRISK_CFPIV")) : 1e-10;
        if (best < prel) { stopmax = best; break; }
        FDR_STEP(p);
        last = best;
    }
    for (int a = 0; a < m && ok; a++) if (!(d0[a] > 0) && P->b[B->con[ti[a]]] != 0) ok = 0;   /* (a constraint zero on the face, b != 0) */
    int nd = 0, nadd = 0;
    double rmax = 0;
    if (ok) {
        /* a constraint is dropped only when its right-hand side follows the others too; one that does not becomes a
         * pivot (kept) while its own relative pivot is above 1e-14 (swissroll: one at 1.3e-8 of the scale) */
        double bsc = 0;
        for (int a = 0; a < m; a++) if (d0[a] > 0) bsc = fmax(bsc, fabs(P->b[B->con[ti[a]]]) / sqrt(d0[a]));
        static double crel = -1; if (crel < 0) crel = getenv("BRISK_CFCTOL") ? atof(getenv("BRISK_CFCTOL")) : 1e-8;
        const double ctol = crel * (bsc > 0 ? bsc : 1.0);
        double *z = malloc(sizeof(double) * (m + 1));
        for (;;) {
            for (int j = 0; j < r; j++) {
                double v = P->b[B->con[ti[piv[j]]]];
                for (int q = 0; q < j; q++) v -= L[piv[j] + (size_t)q * m] * z[q];
                z[j] = v / L[piv[j] + (size_t)j * m];
            }
            int worst = -1; double wrel = 0; rmax = 0;
            for (int a = 0; a < m; a++) {
                if (isp[a]) continue;
                double v = P->b[B->con[ti[a]]];
                for (int q = 0; q < r; q++) v -= L[a + (size_t)q * m] * z[q];
                const double rel = d0[a] > 0 ? fabs(v) / sqrt(d0[a]) : fabs(v);
                if (rel > ctol && rel > wrel) { wrel = rel; worst = a; }
                else if (rel <= ctol) rmax = fmax(rmax, rel / (bsc > 0 ? bsc : 1.0));
            }
            if (worst < 0) break;
            if (!(d0[worst] > 0) || dr[worst] / d0[worst] <= 1e-14) { ok = 0; rmax = wrel / (bsc > 0 ? bsc : 1.0); break; }
            FDR_STEP(worst);
            nadd++;
        }
        free(z);
        if (ok) {
            *drop = malloc(sizeof(int) * (m - r + 1));
            for (int a = 0; a < m; a++) if (!isp[a]) (*drop)[nd++] = ti[a];
        }
    }
    #undef FDR_STEP
    free(G);
    if (verbose >= 1) {
        if (ok) printf("presolve: facial reduction: on the face, %d of %d constraints independent (relative pivots down to %.1e, the rest below %.1e; %d kept for their right-hand sides), %d dependent removed (right-hand sides consistent to %.1e; %.2fs)\n", r, m, last, stopmax, nadd, nd, rmax, wtime() - t0);
        else printf("presolve: facial reduction: on the face of the small index sets the constraints' dependences are not clear or not consistent (%d of %d independent, relative pivots %.1e then %.1e; right-hand sides to %.1e): the face is not applied\n", r, m, last, stopmax, rmax);
    }
    free(Pm); free(d0); free(dr); free(piv); free(L); free(isp); free(ti);
    return ok ? nd : -1;
}

int facial_reduction(Problem *P, int verbose) {
    int removed = 0, reduced_dims = 0;
    if (!P->orig) {                        /* read_sdpa normally sets the identity map */
        P->morig = P->m;
        P->orig = malloc(sizeof(int) * (P->m ? P->m : 1));
        for (int i = 0; i < P->m; i++) P->orig[i] = i;
    }
    g_ps_P = P->ps ? P : NULL;
    if (P->ps) P->ps->blk0 = P->blk;
    int maxn = 1;
    for (int k = 0; k < P->nblk; k++) if (P->blk[k].n > maxn) maxn = P->blk[k].n;
    double *a = malloc(sizeof(double) * maxn);
    int *parts = malloc(sizeof(int) * 2 * P->nblk);
    int unused = remove_unused_indices(P);
    /* Sweeps over the constraints: a reduction is applied as soon as it is found and the
     * sweep continues with the next constraint; sweeps repeat until one finds nothing
     * (a reduction can make earlier constraints reducible). Restarting from the first
     * constraint after every reduction, as before, costs (reductions x m) tests.     */
    int progress = 1, maxlev = 0, ncoupled = 0;
    int *clev = calloc(P->m + 1, sizeof(int));
    /* 23 October 2026: the constraints lifted_faces added are each an exact certificate on the data as read (a
     * combination of its constraints): their level stays 0 until their own reduction, however the others change
     * their coordinates (the 2n - 1 faces of a QAP relaxation share the corner index) */
    const int has_comb = P->ps && P->ps->ncomb > 0;
    unsigned char *mask = calloc(maxn + 1, 1);
    prune_empty_parts(P);
    while (progress) {
        progress = 0;
        /* 4.22: the blocks of each constraint, once per sweep (the incidence only shrinks
         * during a sweep); scanning every block per constraint was 1e8 binary searches on
         * the TSSOS models (5218 blocks: 2.3 s of case2869's 20 s) */
        int *cbp = calloc(P->m + 2, sizeof(int)), *cbk;
        for (int k = 0; k < P->nblk; k++) for (int t = 0; t < P->blk[k].ncon; t++) cbp[P->blk[k].con[t] + 1]++;
        for (int i = 0; i < P->m; i++) cbp[i + 1] += cbp[i];
        cbk = malloc(sizeof(int) * (cbp[P->m] + 1));
        {
            int *fp = malloc(sizeof(int) * (P->m + 1));
            memcpy(fp, cbp, sizeof(int) * (P->m + 1));
            for (int k = 0; k < P->nblk; k++) for (int t = 0; t < P->blk[k].ncon; t++) cbk[fp[P->blk[k].con[t]]++] = k;
            free(fp);
        }
        /* 5.18: the removed constraints keep their numbers until the end of the sweep */
        char *dead = calloc(P->m + 1, 1);
        int ndead = 0;
        /* (3 November 2026: the constraints of clique_faces first, while their blocks are as the faces were found) */
        int *ord = malloc(sizeof(int) * (P->m + 1)), no = 0;
        for (int pass = 0; pass < 2; pass++)
            for (int i = 0; i < P->m; i++) {
                const int jq = P->orig[i] <= -2 ? -2 - P->orig[i] : -1;
                const int nq_ = jq >= 0 && P->ps && P->ps->comb_num && jq < P->ps->ncomb_num && P->ps->comb_num[jq];
                if (nq_ == (pass == 0)) ord[no++] = i;
            }
        for (int ii = 0; ii < no; ii++) {
            const int i = ord[ii];
            if (P->b[i] != 0.0) continue;
            int np = 0, ok = 1;
            double sgn0 = 0;
            const int i0 = i;
            int *cfS = NULL, cfs = 0, cfk = 0, *cfdrop = NULL, ncfdrop = 0; double *cfN = NULL, *cfU = NULL;   /* (kind 5) */
            const int nk = cbp[i0 + 1] - cbp[i0];
            for (int kk = 0; kk < nk && ok; kk++) {
                const int k = cbk[cbp[i0] + kk];
                int t = find_part(&P->blk[k], i);
                if (t < 0) continue;
                const Block *B = &P->blk[k];
                double sg;
                int kind;
                /* (clique_faces' constraints: the numerical test, and only for them) */
                const int jn = P->orig[i] <= -2 ? -2 - P->orig[i] : -1;
                const int numc = jn >= 0 && P->ps && P->ps->comb_num && jn < P->ps->ncomb_num && P->ps->comb_num[jn];
                if (numc) {
                    free(cfS); free(cfN); free(cfU); free(cfdrop); cfS = NULL; cfN = cfU = NULL; cfdrop = NULL; ncfdrop = 0;
                    const Postsolve *ps_ = P->ps;
                    if (B->type != BLK_LP && ps_->comb_fN && ps_->comb_fN[jn] && ps_->comb_fn[jn] == B->n) {
                        /* the face from the vectors themselves (the block as it was when they were found) */
                        const int nn_ = B->n, kk_ = ps_->comb_fk[jn];
                        cfs = nn_; cfk = kk_; sg = 1.0;
                        cfS = malloc(sizeof(int) * nn_); for (int q2 = 0; q2 < nn_; q2++) cfS[q2] = q2;
                        cfN = malloc(sizeof(double) * (size_t)nn_ * (kk_ ? kk_ : 1)); memcpy(cfN, ps_->comb_fN[jn], sizeof(double) * (size_t)nn_ * kk_);
                        cfU = malloc(sizeof(double) * (size_t)nn_ * (nn_ - kk_ ? nn_ - kk_ : 1)); memcpy(cfU, ps_->comb_fU[jn], sizeof(double) * (size_t)nn_ * (nn_ - kk_));
                        kind = 5;
                    } else
                        kind = B->type != BLK_LP && psd_nullspace_tol(&B->A[t], B->n, 3000, 1e-12, 1e3, &sg, &cfS, &cfs, &cfN, &cfk, &cfU) ? 5 : -1;
                    if (kind < 0) { ok = 0; if (verbose >= 1) printf("presolve: faces fixed by small index sets: the combined constraint's rank is not clear (no gap): not applied\n"); }
                    else {
                        ncfdrop = getenv("BRISK_CFNODEP") ? 0 : face_dependent_rows(P, B, t, cfS, cfs, cfN, cfk, cbp, dead, &cfdrop, verbose);
                        if (ncfdrop < 0) { ok = 0; kind = -1; ncfdrop = 0; }
                    }
                }
                else if (B->type == BLK_LP) { if (!samesign_diag(&B->A[t], &sg)) ok = 0; kind = 2; }
                else if (samesign_diag(&B->A[t], &sg)) kind = 1;
                else if (rank1_test(&B->A[t], B->n, a, &sg)) kind = 0;
                else {
                    /* eigen test on the support; bounded work (support^3 plus fill) */
                    int *Sx = NULL, sx = 0, kx = 0; double *Nx = NULL;
                    kind = -1;
                    if (B->A[t].nr <= 400 && psd_nullspace(&B->A[t], B->n, 400, &sg, &Sx, &sx, &Nx, &kx, NULL)) kind = 3;
                    free(Sx); free(Nx);
                    /* (ISSUES 745: any rank, as exact rank-one terms) */
                    if (kind < 0 && B->type != BLK_LP) {
                        const int rk = psd_terms(&B->A[t], B->n, 3000, &sg, NULL);
                        if (rk > 0) {
                            /* the substitutions scan the block's matrices once per term: at most 2e8 entry visits */
                            double bnz = B->C.nnz;
                            for (int u = 0; u < B->ncon; u++) bnz += B->A[u].nnz;
                            if ((double)rk * bnz <= 2e8) kind = 4;
                        }
                    }
                    if (kind < 0) ok = 0;
                }
                if (!ok) break;
                if (sgn0 != 0 && sg != sgn0) { ok = 0; break; }
                sgn0 = sg;
                parts[2 * np] = k; parts[2 * np + 1] = kind; np++;
            }
            if (!ok) { free(cfS); free(cfN); free(cfU); free(cfdrop); continue; }
            {   /* (4 November 2026, the owner) -frdepth k: the chain cut at depth k (the reductions that rest on k earlier ones
                 * are not made); a shallow chain's removed duals can be rebuilt where a deep one's cannot (a user's
                 * Navier-Stokes moment relaxation: depth 16, the moments lost; depth 1, every moment rebuilt) */
                extern int g_fr_maxdepth;
                if (g_fr_maxdepth > 0 && clev[i] + 1 > g_fr_maxdepth) { free(cfS); free(cfN); free(cfU); free(cfdrop); continue; }
            }
            /* apply (and log the reduction for postsolve) */
            PSRecord *rec = g_ps_P ? ps_record_begin(P, P->orig[i], sgn0 != 0 ? sgn0 : 1.0) : NULL;
            const int L = clev[i] + 1;
            if (L > maxlev) maxlev = L;
            int coupled = 0;
            for (int q = 0; q < np; q++) {
                Block *B = &P->blk[parts[2 * q]];
                int t = find_part(B, i);
                {   /* indices whose coordinates this reduction changes */
                    memset(mask, 0, B->n);
                    const SpSym *S = &B->A[t];
                    for (int e = 0; e < S->nnz; e++) { mask[S->row[e]] = 1; mask[S->col[e]] = 1; }
                    fr_touch(B, mask, L, clev, t);
                }
                if (parts[2 * q + 1] == 0) {
                    double sg;
                    rank1_test(&B->A[t], B->n, a, &sg);
                    /* pivot: support index whose row is touched least often */
                    int *cnt = calloc(B->n, sizeof(int));
                    for (int u = -1; u < B->ncon; u++) {
                        const SpSym *S = u < 0 ? &B->C : &B->A[u];
                        if (u == t) continue;
                        for (int e = 0; e < S->nnz; e++) { cnt[S->row[e]]++; if (S->row[e] != S->col[e]) cnt[S->col[e]]++; }
                    }
                    int p = -1;
                    for (int r = 0; r < B->n; r++)
                        if (a[r] != 0 && (p < 0 || cnt[r] < cnt[p] ||
                                          (cnt[r] == cnt[p] && fabs(a[r]) > fabs(a[p])))) p = r;
                    free(cnt);
                    if (rec) {
                        double *wv = calloc(B->n, sizeof(double));
                        for (int rr = 0; rr < B->n; rr++) if (rr != p && a[rr] != 0) wv[rr] = -a[rr] / a[p];
                        ps_record_part(P, rec, B, PS_RANK1, NULL, wv, p, NULL, 0, NULL, 0, NULL);
                        free(wv);
                    }
                    coupled |= fr_coupled(B, t, 0, NULL, a, NULL, 0, NULL, 0, NULL);
                    block_rank1_reduce(B, a, p);
                    reduced_dims++;
                } else if (parts[2 * q + 1] == 5) {
                    /* (clique_faces: the face of the combined constraint, its rank read across a gap; the dense
                     * coupling test is not run, s^3 a matrix) */
                    if (cfN) {
                        /* the constraints dependent on the face first (zero duals), then the compression of the rest */
                        for (int q2 = 0; q2 < ncfdrop; q2++) { const int td = cfdrop[q2]; if (!dead[B->con[td]]) { dead[B->con[td]] = 1; ndead++; } sp_release(&B->A[td]); }
                        if (rec) ps_record_part(P, rec, B, PS_FACE, NULL, NULL, 0, cfS, cfs, cfN, cfk, cfU);
                        coupled = 1;
                        const double tf = wtime();
                        g_bfr_sparse = 1; block_face_reduce(B, cfS, cfs, cfN, cfk); g_bfr_sparse = 0;
                        reduced_dims += cfs - cfk;
                        if (verbose >= 1) printf("presolve: facial reduction: the faces fixed by small index sets: %d indices -> %d (%.2fs)\n", cfs, cfk, wtime() - tf);
                    }
                } else if (parts[2 * q + 1] == 3) {
                    int *Sx = NULL, sx = 0, kx = 0; double *Nx = NULL, *Ux = NULL, sg;
                    if (psd_nullspace(&B->A[t], B->n, 400, &sg, &Sx, &sx, &Nx, &kx, &Ux)) {
                        if (rec) ps_record_part(P, rec, B, PS_FACE, NULL, NULL, 0, Sx, sx, Nx, kx, Ux);
                        coupled |= fr_coupled(B, t, 3, NULL, NULL, Sx, sx, Nx, kx, Ux);
                        block_face_reduce(B, Sx, sx, Nx, kx);
                        reduced_dims += sx - kx;
                    }
                    free(Sx); free(Nx); free(Ux);
                } else if (parts[2 * q + 1] == 4) {
                    /* (ISSUES 745) the terms' rank-one substitutions in their elimination order; one face record for
                     * the postsolve (its dual recovery is the same for any basis of the face) */
                    PsdTerms PT; double sg;
                    memset(&PT, 0, sizeof(PT));
                    if (psd_terms(&B->A[t], B->n, 3000, &sg, &PT)) {
                        if (rec) {
                            double *Nx = NULL, *Ux = NULL; int kx = 0;
                            psd_terms_bases(&PT, &Nx, &Ux, &kx);
                            ps_record_part(P, rec, B, PS_FACE, NULL, NULL, 0, PT.sup, PT.s, Nx, kx, Ux);
                            free(Nx); free(Ux);
                        }
                        coupled = 1;    /* (not tested: the dense test of kind 3 costs s^3 per matrix; it only informs) */
                        int *cur = malloc(sizeof(int) * PT.s);
                        for (int u = 0; u < PT.s; u++) cur[u] = PT.sup[u];
                        double *av = malloc(sizeof(double) * (B->n + 1));
                        for (int tt = 0; tt < PT.r; tt++) {
                            memset(av, 0, sizeof(double) * B->n);
                            for (int e = PT.cp[tt]; e < PT.cp[tt + 1]; e++) av[cur[PT.ci[e]]] = PT.cv[e];
                            const int pc = cur[PT.piv[tt]];
                            block_rank1_reduce(B, av, pc);
                            for (int u = 0; u < PT.s; u++) if (cur[u] > pc) cur[u]--;
                            cur[PT.piv[tt]] = -1;
                        }
                        free(av); free(cur);
                        reduced_dims += PT.r;
                        if (verbose) printf("presolve: facial reduction: a semidefinite constraint of rank %d on %d indices (as %d exact rank-one terms)\n", PT.r, PT.s, PT.r);
                    }
                    psd_terms_free(&PT);
                } else {
                    char *del = calloc(B->n, 1);
                    const SpSym *S = &B->A[t];
                    for (int e = 0; e < S->nnz; e++) del[S->row[e]] = 1;
                    reduced_dims += S->nnz;
                    if (rec) ps_record_part(P, rec, B, B->type == BLK_LP ? PS_LP : PS_DIAG, del, NULL, 0, NULL, 0, NULL, 0, NULL);
                    coupled |= fr_coupled(B, t, B->type == BLK_LP ? 2 : 1, del, NULL, NULL, 0, NULL, 0, NULL);
                    block_delete_indices(B, del);
                    free(del);
                }
            }
            free(cfS); free(cfN); free(cfU); free(cfdrop);
            if (rec) rec->coupled = coupled;
            if (has_comb) for (int r = 0; r < P->m; r++) if (P->orig[r] <= -2) clev[r] = 0;
            ncoupled += coupled;
            remove_parts_of(P, i, parts, np);
            dead[i] = 1; ndead++;
            removed++;
            progress = 1;
        }
        free(ord);
        if (ndead) compact_removed(P, dead, clev);
        free(dead);
        free(cbp); free(cbk);
    }
    free(a); free(parts); free(clev); free(mask);
    P->fr_removed = removed;
    P->fr_depth = maxlev;
    { extern int g_fr_depth_used; if (maxlev > g_fr_depth_used) g_fr_depth_used = maxlev; }
    P->fr_coupled = ncoupled;
    unused += remove_unused_indices(P);
    /* drop blocks that vanished (after the second unused-index pass, which can empty
     * a block too; a size-0 block must never reach LAPACK) */
    int w = 0;
    for (int k = 0; k < P->nblk; k++) {
        Block *B = &P->blk[k];
        if (B->n > 0) {
            if (P->ps) P->ps->cur2orig[w] = P->ps->cur2orig[k];
            P->blk[w++] = *B;
            continue;
        }
        sp_release(&B->C);
        for (int t = 0; t < B->ncon; t++) sp_release(&B->A[t]);
        free(B->A); free(B->con);
    }
    P->nblk = w;
    if (P->ps) { P->ps->nblk_after = w; P->ps->blk0 = NULL; }
    g_ps_P = NULL;
    if (verbose && (removed || unused)) {
        printf("presolve: facial reduction removed %d constraint(s), %d cone dimension(s), depth %d; %d unused index(es)\n",
               removed, reduced_dims, maxlev, unused);
        if (removed && verbose >= 2) printf("presolve: facial reduction: %d of the %d removed constraint(s) coupled to the kept indices by the data (their duals are rebuilt only if the reduced slack allows), %d not\n",
                            ncoupled, removed, removed - ncoupled);
    }
    return removed;
}

/* 20 October 2026: linearly dependent equality constraints. The Gram matrix G_ij = <A_i, A_j> (the matrices and
 * the LP and cone parts as one vector each) is factored by the sparse Cholesky with the tiny-pivot rule: the pivot
 * of constraint i over its diagonal G_ii is the squared sine of the angle between A_i and the span of the constraints
 * eliminated before it, so a pivot below tol of its diagonal marks A_i as a combination of those (SDPT3's check of the
 * constraints, checkdepconstr, factors A A' the same way). The right-hand side must follow the same combination:
 * x = G^+ b by the factor (the marked components 0) gives r = b - A A' x, which is 0 on the kept constraints and, on a
 * marked one, b_i minus the combination of the kept b_j. When every marked r_i is negligible the marked constraints
 * are removed (their dual values are 0 in the postsolve: y_i A_i adds nothing that the kept constraints cannot); when
 * one is not, the constraints are inconsistent, the problem has no feasible point and nothing is removed (*ninc set).
 * The work is bounded (forming G, its fill and its factorization; the routine returns 0 above the budget): on a
 * problem whose interior-point solve factors its Schur complement, the pattern of G lies inside that of the Schur
 * complement and the check costs at most about one iteration. Returns the number of constraints removed. */
typedef struct { uint64_t key; int con; double v; } DepEnt;
typedef struct { uint64_t h; int i; } DepPH;
static int cmp_depph(const void *a, const void *b) {
    const DepPH *x = a, *y = b;
    if (x->h != y->h) return x->h < y->h ? -1 : 1;
    return x->i - y->i;
}
static int cmp_depent(const void *a, const void *b) {
    const DepEnt *x = a, *y = b;
    if (x->key != y->key) return x->key < y->key ? -1 : 1;
    return x->con - y->con;
}
/* The entries in cmp_depent's order (by position, then constraint) by a least-significant-digit radix sort: stable
 * passes of 17 bits over the constraint, then over the position, as many as their largest values need (the
 * constraint's passes are skipped when con_sorted: the entries of one position then lie in one block, in the order of
 * its constraints, which ascend). Where the
 * positions are too many for a counting sort (a block of order 100,001 has 5e9 positions; r2_reglmi_N100k, SDP with
 * SOCP: 10 million entries, 1.6 s in qsort) it takes a few passes over the entries. Falls back to qsort without memory. */
static void dep_sort(DepEnt **pE, size_t ne, int con_sorted) {
    DepEnt *a = *pE, *b = malloc(sizeof(DepEnt) * (ne + 1));
    enum { DB = 17, DN = 1 << DB };
    size_t *cnt = malloc(sizeof(size_t) * (DN + 1));
    if (!b || !cnt) { free(b); free(cnt); qsort(a, ne, sizeof(DepEnt), cmp_depent); return; }
    uint64_t kmax = 0; unsigned cmax = 0;
    for (size_t q = 0; q < ne; q++) { if (a[q].key > kmax) kmax = a[q].key; if ((unsigned)a[q].con > cmax) cmax = (unsigned)a[q].con; }
    for (int pass = con_sorted ? 1 : 0; pass < 2; pass++) {   /* the constraint first (the minor key), then the position */
        const uint64_t top = pass == 0 ? cmax : kmax;
        for (int sh = 0; sh < 64 && (top >> sh) != 0; sh += DB) {
            memset(cnt, 0, sizeof(size_t) * (DN + 1));
            for (size_t q = 0; q < ne; q++) {
                const uint64_t v = pass == 0 ? (uint64_t)(unsigned)a[q].con : a[q].key;
                cnt[((v >> sh) & (DN - 1)) + 1]++;
            }
            for (int d = 0; d < DN; d++) cnt[d + 1] += cnt[d];
            for (size_t q = 0; q < ne; q++) {
                const uint64_t v = pass == 0 ? (uint64_t)(unsigned)a[q].con : a[q].key;
                b[cnt[(v >> sh) & (DN - 1)]++] = a[q];
            }
            DepEnt *t = a; a = b; b = t;
        }
    }
    free(cnt); free(b);
    *pE = a;
}
/* 31 October 2026 (the owner's external SDP benchmark, many-blocks case 18-09: 16 blocks of order 84, 420 rows, 24 million
 * entries, 5.3 s of the check for nothing): when the rows are dense over the positions, a screen first. G = A A' from the
 * dense m x positions matrix built straight from the blocks (no list of entries, no sort), scaled to unit diagonal; if
 * G_s - 10 tol I factors, lambda_min(G_s) > 10 tol, so every pivot of every elimination order is above tol times its
 * diagonal and the check below would find nothing. Returns 1 when that is proved, 0 otherwise (the check then runs). */
static int dep_dense_screen(const Problem *P, double tol, size_t ne) {
    const int m = P->m;
    size_t R = 0;
    for (int k = 0; k < P->nblk; k++) { const size_t n = (size_t)P->blk[k].n; R += P->blk[k].type == BLK_SDP ? n * (n + 1) / 2 : n; }
    if (R == 0 || (double)ne < 0.25 * (double)m * (double)R || 8.0 * (double)m * ((double)R + m) > 0.1 * brisk_mem_limit()) return 0;
    double *Ad = calloc((size_t)m * R + 1, sizeof(double)), *G = calloc((size_t)m * m + 1, sizeof(double));
    size_t *off = malloc(sizeof(size_t) * ((size_t)P->nblk + 1));
    if (!Ad || !G || !off) { free(Ad); free(G); free(off); return 0; }   /* (0: not run) */
    { size_t o = 0; for (int k = 0; k < P->nblk; k++) { off[k] = o; const size_t n = (size_t)P->blk[k].n; o += P->blk[k].type == BLK_SDP ? n * (n + 1) / 2 : n; } }
    #pragma omp parallel for schedule(dynamic, 1)
    for (int k = 0; k < P->nblk; k++) {          /* (each block writes its own positions) */
        const Block *B = &P->blk[k];
        const int sdp = B->type == BLK_SDP;
        for (int t = 0; t < B->ncon; t++) {
            const SpSym *S = &B->A[t];
            const size_t row = (size_t)B->con[t];
            for (int e = 0; e < S->nnz; e++) {
                const size_t r = (size_t)S->row[e], c = (size_t)S->col[e];
                const size_t pos = off[k] + (sdp ? (r <= c ? c * (c + 1) / 2 + r : r * (r + 1) / 2 + c) : r);
                Ad[row + pos * (size_t)m] += sdp && r != c ? M_SQRT2 * S->val[e] : S->val[e];
            }
        }
    }
    free(off);
    const int Ri = (int)R; const double one = 1.0, zero = 0.0;
    if ((double)R > 2147483647.0) { free(Ad); free(G); return 0; }
    BL(dsyrk_)("L", "N", &m, &Ri, &one, Ad, &m, &zero, G, &m);
    free(Ad);
    int ok = 1;
    double *dg = malloc(sizeof(double) * (size_t)m);
    for (int i = 0; i < m && ok; i++) { dg[i] = G[(size_t)i * m + i]; if (!(dg[i] > 0)) ok = 0; else dg[i] = 1.0 / sqrt(dg[i]); }
    if (ok) {
        for (int j = 0; j < m; j++) for (int i = j; i < m; i++) G[i + (size_t)j * m] *= dg[i] * dg[j];
        for (int i = 0; i < m; i++) G[i + (size_t)i * m] -= 10.0 * tol;
        int info = 0;
        BL(dpotrf_)("L", &m, G, &m, &info);
        ok = info == 0;
    }
    free(dg); free(G);
    return ok ? 1 : -1;   /* (-1: the screen ran and proved nothing; the check follows) */
}
int remove_dependent_rows(Problem *P, int verbose, double tol, int *ninc) {
    const int mfull = P->m;
    int m = mfull;
    *ninc = 0;
    if (m < 2 || tol <= 0) return 0;
    const double t0 = wtime();
    if (!P->orig) { P->morig = m; P->orig = malloc(sizeof(int) * m); for (int i = 0; i < m; i++) P->orig[i] = i; }
    int *cmap = NULL; double *bb = NULL;
    size_t ne = 0;
    for (int k = 0; k < P->nblk; k++) for (int t = 0; t < P->blk[k].ncon; t++) ne += (size_t)P->blk[k].A[t].nnz;
    {   const int sc = getenv("BRISK_DEPNOSCREEN") ? 0 : dep_dense_screen(P, tol, ne);
        if (sc == 1) {
            if (verbose >= 2) printf("presolve: no dependent constraints (the Gram matrix of the dense rows, scaled to unit diagonal, has lambda_min > %.0e; %.2fs)\n", 10.0 * tol, wtime() - t0);
            return 0;
        }
        if (sc < 0 && verbose >= 2) printf("presolve: dependent constraints: the dense rows' Gram matrix may be singular (%.2fs); the full check follows\n", wtime() - t0);
    }
    if (ne == 0) return 0;
    DepEnt *E = malloc(sizeof(DepEnt) * ne);
    size_t w = 0;
    for (int k = 0; k < P->nblk; k++) {
        const Block *B = &P->blk[k];
        for (int t = 0; t < B->ncon; t++) {
            const SpSym *S = &B->A[t];
            for (int e = 0; e < S->nnz; e++) {
                if (S->val[e] == 0.0) continue;
                const uint64_t r = (uint64_t)S->row[e], c = (uint64_t)S->col[e];
                const int sdp = B->type == BLK_SDP;
                E[w].key = ((uint64_t)k << 44) | (sdp ? c * (c + 1) / 2 + r : r);
                E[w].con = B->con[t];
                E[w].v = sdp && r != c ? M_SQRT2 * S->val[e] : S->val[e];   /* <A, B> counts an off-diagonal pair twice */
                w++;
            }
        }
    }
    ne = w;
    {   /* by position, then constraint: a counting sort over the positions when there are not many more of them than
         * entries (stable, and the constraints come in ascending order within a block), else a radix sort (dep_sort) */
        size_t R = 0;
        size_t *off = malloc(sizeof(size_t) * ((size_t)P->nblk + 1));
        for (int k = 0; k < P->nblk; k++) { off[k] = R; const size_t n = (size_t)P->blk[k].n; R += P->blk[k].type == BLK_SDP ? n * (n + 1) / 2 : n; }
        if (R <= 4 * ne + ((size_t)1 << 20)) {
            size_t *cnt = calloc(R + 1, sizeof(size_t));
            for (size_t q = 0; q < ne; q++) { const uint64_t d = off[E[q].key >> 44] + (E[q].key & (((uint64_t)1 << 44) - 1)); E[q].key = d; cnt[d + 1]++; }
            for (size_t d = 0; d < R; d++) cnt[d + 1] += cnt[d];
            DepEnt *E2 = malloc(sizeof(DepEnt) * (ne + 1));
            for (size_t q = 0; q < ne; q++) E2[cnt[E[q].key]++] = E[q];
            free(cnt); free(E); E = E2;
        } else {   /* (the positions as one index, as above: 33 bits for a block of order 100,001) */
            int con_sorted = 1;
            for (int k = 0; k < P->nblk && con_sorted; k++) for (int t = 1; t < P->blk[k].ncon; t++) if (P->blk[k].con[t] <= P->blk[k].con[t - 1]) { con_sorted = 0; break; }
            for (size_t q = 0; q < ne; q++) E[q].key = off[E[q].key >> 44] + (E[q].key & (((uint64_t)1 << 44) - 1));
            dep_sort(&E, ne, con_sorted);
        }
        free(off);
    }
    w = 0;                                     /* (one entry per constraint and position; no zeros: a zero is no entry) */
    for (size_t q = 0; q < ne; q++) {
        if (w > 0 && E[w - 1].key == E[q].key && E[w - 1].con == E[q].con) E[w - 1].v += E[q].v; else E[w++] = E[q];
    }
    { size_t z = 0; for (size_t q = 0; q < w; q++) if (E[q].v != 0.0) E[z++] = E[q]; w = z; }
    ne = w;
    /* A constraint that is alone at some position among the remaining ones takes no part in a dependence (its
     * coefficient in a vanishing combination of them is 0 there): such constraints are set aside one after another,
     * and only the rest (the core) goes to the Gram matrix; a dependence of the whole set lies in the core. In a moment
     * relaxation the highest moments are alone at their positions, then the next ones, and so on. */
    {
        size_t ngp = 0;
        int *gq = malloc(sizeof(int) * (ne + 1)), *cnt = malloc(sizeof(int) * (ne + 1));
        size_t *gp = malloc(sizeof(size_t) * (ne + 1));
        for (size_t q = 0; q < ne; ) { size_t q1 = q; while (q1 < ne && E[q1].key == E[q].key) q1++; gp[ngp] = q; cnt[ngp] = (int)(q1 - q); for (size_t t = q; t < q1; t++) gq[t] = (int)ngp; ngp++; q = q1; }
        gp[ngp] = ne;
        int *crp = calloc((size_t)m + 1, sizeof(int)), *cri = malloc(sizeof(int) * (ne + 1));
        for (size_t q = 0; q < ne; q++) crp[E[q].con + 1]++;
        for (int i = 0; i < m; i++) crp[i + 1] += crp[i];
        { int *fill = malloc(sizeof(int) * (m + 1)); memcpy(fill, crp, sizeof(int) * m); for (size_t q = 0; q < ne; q++) cri[fill[E[q].con]++] = (int)q; free(fill); }
        char *aside = calloc((size_t)m + 1, 1);
        int *stk = malloc(sizeof(int) * (ngp + 1)); size_t ns = 0;
        for (size_t g = 0; g < ngp; g++) if (cnt[g] == 1) stk[ns++] = (int)g;
        while (ns > 0) {
            const int g = stk[--ns];
            if (cnt[g] != 1) continue;
            int r = -1;
            for (size_t q = gp[g]; q < gp[g + 1]; q++) if (!aside[E[q].con]) { r = E[q].con; break; }
            if (r < 0) continue;
            aside[r] = 1;
            for (int t = crp[r]; t < crp[r + 1]; t++) { const int g2 = gq[cri[t]]; if (--cnt[g2] == 1) stk[ns++] = g2; }
        }
        int mc = 0;
        for (int i = 0; i < m; i++) if (!aside[i]) mc++;
        free(gq); free(cnt); free(gp); free(crp); free(cri); free(stk);
        if (getenv("BRISK_DEPDBG")) printf("   [dependent constraints: %d of %d set aside (alone at a position), core %d; %.2fs]\n", m - mc, m, mc, wtime() - t0);
        if (mc == 0) { free(aside); free(E); if (verbose >= 2) printf("presolve: no dependent constraints (each constraint alone at some position once the others are set aside)\n"); return 0; }
        int *nw = malloc(sizeof(int) * (m + 1)); cmap = malloc(sizeof(int) * (mc + 1)); bb = malloc(sizeof(double) * (mc + 1));
        int c = 0;
        for (int i = 0; i < m; i++) { nw[i] = aside[i] ? -1 : c; if (!aside[i]) { cmap[c] = i; bb[c] = P->b[i]; c++; } }
        w = 0;
        for (size_t q = 0; q < ne; q++) if (nw[E[q].con] >= 0) { E[w] = E[q]; E[w].con = nw[E[q].con]; w++; }
        ne = w; m = mc;
        free(nw); free(aside);
    }
    /* Multiples of another constraint (duplicated constraints, the moments a relaxation repeats: 1,350 of the 2,610
     * dependences of a user's relaxation) are found first, exactly, by a hash of the pattern and of the values over
     * the first one, and a comparison within each class; only the rest goes to the Gram matrix. */
    char *pdead = calloc((size_t)mfull + 1, 1);
    int npar = 0, nparbad = 0; double rpar = 0, bsc0 = 0;
    {
        int *prp = calloc((size_t)m + 1, sizeof(int)), *pe = malloc(sizeof(int) * (ne + 1));
        for (size_t q = 0; q < ne; q++) prp[E[q].con + 1]++;
        for (int i = 0; i < m; i++) prp[i + 1] += prp[i];
        { int *fill = malloc(sizeof(int) * ((size_t)m + 1)); for (int i = 0; i < m; i++) fill[i] = prp[i]; for (size_t q = 0; q < ne; q++) pe[fill[E[q].con]++] = (int)q; free(fill); }
        double *nrm = calloc((size_t)m + 1, sizeof(double));
        DepPH *ph = malloc(sizeof(DepPH) * ((size_t)m + 1));
        for (int i = 0; i < m; i++) {
            double ss = 0; uint64_t h = 1469598103934665603ULL ^ (uint64_t)(prp[i + 1] - prp[i]);
            const double v0 = prp[i + 1] > prp[i] ? E[pe[prp[i]]].v : 1.0;
            for (int p = prp[i]; p < prp[i + 1]; p++) {
                const DepEnt *e = &E[pe[p]]; ss += e->v * e->v;
                int ex; const double mt = frexp(e->v / v0, &ex);
                const int64_t mq = llrint(mt * 1099511627776.0);   /* (40 bits of the value over the first one) */
                h = (h ^ e->key) * 1099511628211ULL; h = (h ^ (uint64_t)mq) * 1099511628211ULL; h = (h ^ (uint64_t)(ex + 2048)) * 1099511628211ULL;
            }
            nrm[i] = sqrt(ss); ph[i].h = h; ph[i].i = i;
            if (nrm[i] > 0) bsc0 = fmax(bsc0, fabs(bb[i]) / nrm[i]);
        }
        /* (sorted by the hash, then the index: the representative of a class is its first constraint) */
        qsort(ph, (size_t)m, sizeof(DepPH), cmp_depph);
        const double ctol0 = 1e-8 * (bsc0 > 0 ? bsc0 : 1.0);
        char *cd = calloc((size_t)m + 1, 1);
        for (int a = 0; a < m; ) {
            int z = a + 1; while (z < m && ph[z].h == ph[a].h) z++;
            for (int x = a; x < z; x++) {
                const int i = ph[x].i; if (cd[i] || !(nrm[i] > 0)) continue;
                const double vi = E[pe[prp[i]]].v;
                for (int y = x + 1; y < z; y++) {
                    const int j = ph[y].i; if (cd[j] || prp[j + 1] - prp[j] != prp[i + 1] - prp[i]) continue;
                    const double vj = E[pe[prp[j]]].v, r = vj / vi;
                    int same = 1;
                    for (int t = 0; t < prp[i + 1] - prp[i] && same; t++) {
                        const DepEnt *ei = &E[pe[prp[i] + t]], *ej = &E[pe[prp[j] + t]];
                        if (ei->key != ej->key || fabs(ej->v - r * ei->v) > 1e-14 * fabs(ej->v)) same = 0;
                    }
                    if (!same) continue;
                    const double rel = fabs(bb[j] - r * bb[i]) / nrm[j];
                    if (rel > ctol0) { nparbad++; continue; }
                    cd[j] = 1; pdead[cmap[j]] = 1; npar++; if (rel > rpar) rpar = rel;
                }
            }
            a = z;
        }
        free(ph); free(prp); free(pe);
        if (getenv("BRISK_DEPDBG")) printf("   [dependent constraints: %d multiples of another, %d inconsistent; %.2fs]\n", npar, nparbad, wtime() - t0);
        if (nparbad > 0) {
            *ninc = nparbad;
            if (verbose >= 0) printf("presolve: %d equality constraint%s multiples of others with different right-hand sides: the constraints are inconsistent and the problem has no feasible point\n", nparbad, nparbad > 1 ? "s are" : " is");
            free(cd); free(nrm); free(pdead); free(E); free(cmap); free(bb);
            return 0;
        }
        if (npar > 0) {                   /* the core without them */
            int *nw = malloc(sizeof(int) * ((size_t)m + 1)), c = 0;
            for (int i = 0; i < m; i++) { nw[i] = cd[i] ? -1 : c; if (!cd[i]) { cmap[c] = cmap[i]; bb[c] = bb[i]; c++; } }
            w = 0;
            for (size_t q = 0; q < ne; q++) if (nw[E[q].con] >= 0) { E[w] = E[q]; E[w].con = nw[E[q].con]; w++; }
            ne = w; m = c;
            free(nw);
        }
        free(cd); free(nrm);
    }
    if (m < 2 || ne == 0) {               /* (the multiples were all) */
        if (npar > 0) { int *clev = calloc((size_t)mfull + 1, sizeof(int)); compact_removed(P, pdead, clev); free(clev); }
        if (npar > 0 && verbose >= 1) printf("presolve: %d linearly dependent equality constraints removed (multiples of another; right-hand sides consistent to %.1e)\n", npar, rpar / (bsc0 > 0 ? bsc0 : 1));
        free(pdead); free(E); free(cmap); free(bb);
        return npar;
    }
    /* the positions (columns of A) and the work of forming G */
    size_t ng = 0; double work = 0;
    int *colof = malloc(sizeof(int) * (ne + 1));
    size_t *gs = malloc(sizeof(size_t) * (ne + 1));
    for (size_t q = 0; q < ne; ) {
        size_t q1 = q; while (q1 < ne && E[q1].key == E[q].key) q1++;
        gs[ng] = q; for (size_t p = q; p < q1; p++) colof[p] = (int)ng;
        work += (double)(q1 - q) * (double)(q1 - q);
        ng++; q = q1;
    }
    gs[ng] = ne;
    const double budget = getenv("BRISK_DEPWORK") ? atof(getenv("BRISK_DEPWORK")) : 2e9;
    int removed = 0;
    int *rp = NULL, *ri = NULL, *deg = NULL, **nbr = NULL, *pool = NULL, *mark = NULL;
    double *acc = NULL, *d0 = NULL, *piv = NULL, *x = NULL, *u = NULL, *wk = NULL, *Gacc = NULL;
    char *dead = NULL;
    SChol *S = NULL;
    /* dense data (every constraint at most positions): G = A A' by one BLAS-3 product of the dense m x positions matrix,
     * when that costs less than a tenth of the scalar pairs (a random problem with 300 dense 80 x 80 constraints: 1.2 s
     * in the pairs) */
    const double mm = (double)m;
    const int dense = mm * mm * (double)ng * 0.5 <= 10.0 * work && 8.0 * mm * ((double)ng + mm) <= 0.1 * brisk_mem_limit();
    if (dense ? mm * mm * (double)ng * 0.5 > 25.0 * budget : work > budget) { if (verbose >= 2) printf("presolve: dependent constraints not checked (forming the Gram matrix: %.2g > %.2g)\n", work, budget); goto out; }
    /* the entries by constraint */
    rp = calloc((size_t)m + 1, sizeof(int)); ri = malloc(sizeof(int) * (ne + 1));
    for (size_t q = 0; q < ne; q++) rp[E[q].con + 1]++;
    for (int i = 0; i < m; i++) rp[i + 1] += rp[i];
    { int *fill = malloc(sizeof(int) * ((size_t)m + 1)); for (int i = 0; i < m; i++) fill[i] = rp[i]; for (size_t q = 0; q < ne; q++) ri[fill[E[q].con]++] = (int)q; free(fill); }
    /* the pattern of G (adjacency without the diagonal) */
    mark = malloc(sizeof(int) * ((size_t)m + 1)); for (int i = 0; i < m; i++) mark[i] = -1;
    deg = calloc((size_t)m + 1, sizeof(int));
    size_t tot = 0;
    /* G accumulated as a dense array in one pass when it fits and its pattern can be dense (the work of forming it at
     * least a quarter of its entries); the pattern is then read from the values instead of two more passes over the
     * pairs of entries (a random SDP with 1,500 constraints in three blocks of order 150, forming work 2.9e8: the pattern
     * 0.91 s and the values 0.46 s, on one thread, for a pattern 99 % full). The columns on the threads, each written by
     * one thread and summed in the same order as before (the same values at any thread count). */
    if (!dense && (double)work >= 0.25 * mm * (mm - 1.0) && 8.0 * mm * mm <= 0.1 * brisk_mem_limit() && !getenv("BRISK_DEPNOACC")) {
        /* (zeroed on one thread before the parallel loop: the threads' first touches of a fresh array fault its pages
         * at the same time, and with little work a column that cost 15 ns a product, against 1 ns; poly_slack_n32,
         * a control LMI with 2,016 core constraints) */
        Gacc = malloc(sizeof(double) * (size_t)m * m);
        if (!Gacc) goto out;
        memset(Gacc, 0, sizeof(double) * (size_t)m * m);
        const double ta0 = wtime();
        #pragma omp parallel for schedule(dynamic, 8)
        for (int i = 0; i < m; i++) {
            double *col = Gacc + (size_t)i * m;
            for (int p = rp[i]; p < rp[i + 1]; p++) { const DepEnt *ei = &E[ri[p]]; const int g = colof[ri[p]];
                for (size_t q = gs[g]; q < gs[g + 1]; q++) if (E[q].con >= i) col[E[q].con] += ei->v * E[q].v; }
        }
        const double ta1 = wtime();
        /* the upper triangle from the lower by tiles (each column then holds the whole row: the lists below are read
         * from one column each, contiguously; a list filled from both triangles costs a cache miss an entry, 0.1 s for
         * a full pattern of a million entries) */
        { const int TB = 64;
          #pragma omp parallel for schedule(dynamic, 1)
          for (int jb = 0; jb < m; jb += TB)
              for (int ib = jb; ib < m; ib += TB)
                  for (int j = jb; j < jb + TB && j < m; j++)
                      for (int i = ib > j + 1 ? ib : j + 1; i < ib + TB && i < m; i++) Gacc[(size_t)i * m + j] = Gacc[(size_t)j * m + i]; }
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < m; i++) { const double *col = Gacc + (size_t)i * m; int c = 0;
            for (int j = 0; j < m; j++) if (j != i && col[j] != 0) c++;
            deg[i] = c; }
        for (int i = 0; i < m; i++) tot += (size_t)deg[i];
        if (getenv("BRISK_DEPDBG")) printf("   [... dense accumulation %.3fs, transpose and pattern %.3fs]\n", ta1 - ta0, wtime() - ta1);
    }
    else if (dense) { for (int i = 0; i < m; i++) deg[i] = m - 1; tot = (size_t)m * (size_t)(m - 1); }
    else for (int i = 0; i < m; i++) {
        mark[i] = i;
        for (int p = rp[i]; p < rp[i + 1]; p++) { const int g = colof[ri[p]];
            for (size_t q = gs[g]; q < gs[g + 1]; q++) { const int j = E[q].con; if (mark[j] != i) { mark[j] = i; deg[i]++; } } }
        tot += (size_t)deg[i];
    }
    const size_t fillcap = (size_t)(0.1 * brisk_mem_limit() / 8.0);
    if (tot > fillcap) goto out;
    const double fcap = getenv("BRISK_DEPFLOPS") ? atof(getenv("BRISK_DEPFLOPS")) : 2e10;
    {   /* 24 October 2026: the factor holds at least the pattern, nnz(L) >= tot/2 + m in any order, and its flops (the sum
         * of the squared column counts) are at least nnz(L)^2 / m (Cauchy-Schwarz): over the flop budget the ordering is
         * not computed (pglib_opf_case13659_pegase, SDP: a core of 17,467 constraints, the bound 1.5e11; the analysis took
         * 10.5 s to find 3.5e11 and the check was then not made) */
        const double nl = 0.5 * (double)tot + mm, lbf = nl * nl / mm;
        if (lbf > fcap) { if (verbose >= 2) printf("presolve: dependent constraints not checked (factoring the Gram matrix: at least %.2g flops > %.2g)\n", lbf, fcap); goto out; }
    }
    pool = malloc(sizeof(int) * (tot + 1)); nbr = malloc(sizeof(int *) * (m + 1));
    for (int i = 0; i < m; i++) mark[i] = -1;
    if (Gacc) {   /* (each list from its own column, in increasing order) */
        size_t o = 0; for (int i = 0; i < m; i++) { nbr[i] = pool + o; o += (size_t)deg[i]; }
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < m; i++) { const double *col = Gacc + (size_t)i * m; int c = 0;
            for (int j = 0; j < m; j++) if (j != i && col[j] != 0) nbr[i][c++] = j; }
    }
    else { size_t o = 0;
      for (int i = 0; i < m; i++) {
          nbr[i] = pool + o; int c = 0; mark[i] = i;
          if (dense) { for (int j = 0; j < m; j++) if (j != i) nbr[i][c++] = j; o += (size_t)c; continue; }
          for (int p = rp[i]; p < rp[i + 1]; p++) { const int g = colof[ri[p]];
              for (size_t q = gs[g]; q < gs[g + 1]; q++) { const int j = E[q].con; if (mark[j] != i) { mark[j] = i; nbr[i][c++] = j; } } }
          o += (size_t)c;
      } }
    const int dbg = getenv("BRISK_DEPDBG") != NULL;
    if (dbg) printf("   [dependent constraints: %zu entries, %zu positions, forming work %.3g, pattern %zu%s; %.2fs]\n", ne, ng, work, tot, Gacc ? " (from the values accumulated as dense)" : "", wtime() - t0);
    /* (AMD alone, as for the LP's normal equations: one factorization does not repay the search for a better ordering;
     * a pattern of AC-OPF's moment relaxation, 4,517 constraints and 40 % dense: 0.63 -> 0.2 s) */
    /* (a pattern at least a quarter full is factored as dense: the identity order, the values gathered from a dense
     * array; the state-feedback LMI with 96 states, 2,304 core constraints and a full pattern: 0.65 s in the ordering
     * and the scattered additions) */
    const int dpat = !dense && (double)tot >= 0.25 * mm * (mm - 1.0) && 8.0 * mm * mm <= 0.1 * brisk_mem_limit();
    int *idp = NULL;
    if (dpat) { idp = malloc(sizeof(int) * ((size_t)m + 1)); for (int i = 0; i < m; i++) idp[i] = i; schol_set_perm(idp); }
    const int oamd = schol_set_amd(getenv("BRISK_DEPAMD") ? atoi(getenv("BRISK_DEPAMD")) : 2);
    S = schol_analyze_adj(m, deg, nbr, fillcap);
    schol_set_amd(oamd);
    if (dpat) { schol_set_perm(NULL); free(idp); }
    if (dbg && S) printf("   [... analysis: factor %zu entries, %.3g flops; %.2fs]\n", schol_nnz(S), schol_flops(S), wtime() - t0);
    if (!S) { if (verbose >= 2) printf("presolve: dependent constraints not checked (the factor of the Gram matrix exceeds its memory budget)\n"); goto out; }
    if (schol_flops(S) > fcap) { if (verbose >= 2) printf("presolve: dependent constraints not checked (factoring the Gram matrix: %.2g flops > %.2g)\n", schol_flops(S), fcap); goto out; }
    /* the values */
    schol_zero(S);
    acc = calloc((size_t)m, sizeof(double)); d0 = malloc(sizeof(double) * m);
    if (dense) {
        double *Ad = calloc((size_t)m * ng, sizeof(double)), *Gd = calloc((size_t)m * m, sizeof(double));
        for (size_t q = 0; q < ne; q++) Ad[(size_t)E[q].con + (size_t)colof[q] * m] = E[q].v;
        const int ngi = (int)ng; const double one = 1.0, zero = 0.0;
        BL(dsyrk_)("L", "N", &m, &ngi, &one, Ad, &m, &zero, Gd, &m);
        free(Ad);
        for (int i = 0; i < m; i++) d0[i] = Gd[(size_t)i * m + i];
        schol_gather_dense(S, Gd);
        free(Gd);
    }
    else if (Gacc) {
        for (int i = 0; i < m; i++) d0[i] = Gacc[(size_t)i * m + i];
        if (dpat) schol_gather_dense(S, Gacc);
        else for (int i = 0; i < m; i++) {
            schol_add(S, (size_t)i, (size_t)i, d0[i]);
            for (int c = 0; c < deg[i]; c++) { const int j = nbr[i][c]; if (j > i) schol_add(S, (size_t)j, (size_t)i, Gacc[(size_t)i * m + j]); }
        }
        free(Gacc); Gacc = NULL;
    }
    else if (dpat) {
        double *Gd = calloc((size_t)m * m, sizeof(double));
        for (int i = 0; i < m; i++) {
            double *col = Gd + (size_t)i * m;
            for (int p = rp[i]; p < rp[i + 1]; p++) { const DepEnt *ei = &E[ri[p]]; const int g = colof[ri[p]];
                for (size_t q = gs[g]; q < gs[g + 1]; q++) if (E[q].con >= i) col[E[q].con] += ei->v * E[q].v; }
            d0[i] = col[i];
        }
        schol_gather_dense(S, Gd);
        free(Gd);
    }
    else for (int i = 0; i < m; i++) {
        for (int p = rp[i]; p < rp[i + 1]; p++) { const DepEnt *ei = &E[ri[p]]; const int g = colof[ri[p]];
            for (size_t q = gs[g]; q < gs[g + 1]; q++) acc[E[q].con] += ei->v * E[q].v; }
        d0[i] = acc[i];
        schol_add(S, (size_t)i, (size_t)i, acc[i]); acc[i] = 0;
        for (int c = 0; c < deg[i]; c++) { const int j = nbr[i][c]; if (j > i) schol_add(S, (size_t)j, (size_t)i, acc[j]); acc[j] = 0; }
    }
    if (dbg) printf("   [... values: %.2fs]\n", wtime() - t0);
    schol_set_tinypiv(S, tol);
    if (schol_factor(S, 0.0)) goto out;
    if (dbg) printf("   [... factor: %.2fs]\n", wtime() - t0);
    piv = malloc(sizeof(double) * m);
    schol_pivots(S, piv);
    dead = calloc((size_t)m + 1, 1);
    char *flag = calloc((size_t)m + 1, 1);
    int nf = 0;
    double pmin = INFINITY;                /* (the smallest relative pivot not flagged, for the log) */
    for (int i = 0; i < m; i++) {
        if (piv[i] >= 1e100 || !(d0[i] > 0)) { flag[i] = 1; nf++; }
        else if (piv[i] / d0[i] < pmin) pmin = piv[i] / d0[i];
    }
    if (nf == 0) { free(flag); if (verbose >= 2) printf("presolve: no dependent constraints (smallest pivot of the Gram matrix %.1e of its diagonal)\n", pmin); goto out; }
    /* Each flagged A_i is checked in the space of the data, not through G (which squares the condition): lambda from
     * G_KK lambda = A_K a_i (K: the constraints not flagged), the residual v = a_i - A_K' lambda, one step of refinement
     * (G_KK dl = A_K v; the corrected semi-normal equations), and A_i is dependent when |v| <= stol |a_i|; its right-hand
     * side must then be b_i = lambda' b_K. A flagged constraint that fails the test is kept (it was only left out of the
     * span the others are checked against, which can hide a dependence but not create one). */
    {
        const double vwork = (double)nf * 2.0 * (2.0 * (double)ne + (double)ng + 2.0 * (double)schol_nnz(S));
        const double vbud = getenv("BRISK_DEPVWORK") ? atof(getenv("BRISK_DEPVWORK")) : 1e10;
        if (vwork > vbud) { free(flag); if (verbose >= 2) printf("presolve: %d constraints flagged as dependent, not verified (work %.2g > %.2g); kept\n", nf, vwork, vbud); goto out; }
    }
    static double stol = -1; if (stol < 0) stol = getenv("BRISK_DEPSIN") ? atof(getenv("BRISK_DEPSIN")) : 1e-9;
    u = malloc(sizeof(double) * (ng + 1));
    double bsc = 0;
    for (int i = 0; i < m; i++) if (d0[i] > 0) bsc = fmax(bsc, fabs(bb[i]) / sqrt(d0[i]));
    const double ctol = 1e-8 * (bsc > 0 ? bsc : 1.0);
    int nd = 0, nbad = 0, nkeep = 0;
    double sdep = 0, skeep = INFINITY, rd = 0;
    /* the flagged constraints in chunks, their solves together (schol_solve_many) */
    int *fl = malloc(sizeof(int) * (nf + 1)), nfl = 0;
    for (int i = 0; i < m; i++) if (flag[i]) {
        if (!(d0[i] > 0)) { if (bb[i] == 0) { dead[i] = 1; nd++; } else nbad++; }   /* an empty constraint: consistent only with b_i = 0 */
        else fl[nfl++] = i;
    }
    int ch = (int)((1u << 23) / ((size_t)m + 1)); if (ch < 1) ch = 1; if (ch > nfl) ch = nfl > 0 ? nfl : 1;
    x = malloc(sizeof(double) * (size_t)m * ch); wk = malloc(sizeof(double) * ((size_t)m * ch + m + 1));
    double *g = malloc(sizeof(double) * (size_t)m * ch);
    /* the entries by constraint as contiguous arrays (the products A_K v read them row by row) */
    double *rv = malloc(sizeof(double) * (ne + 1)); int *rcl = malloc(sizeof(int) * (ne + 1));
    for (size_t p = 0; p < ne; p++) { rv[p] = E[ri[p]].v; rcl[p] = colof[ri[p]]; }
    int nth = 1;
#ifdef _OPENMP
    nth = omp_get_max_threads();
#endif
    double *ubuf = malloc(sizeof(double) * ((size_t)ng + 1) * (size_t)nth);
    double *snc = malloc(sizeof(double) * ((size_t)ch + 1)), *relc = malloc(sizeof(double) * ((size_t)ch + 1));
    /* v = a_i - A_K' xc over the positions (xc NULL: v = a_i), into uu */
    #define DEP_RESID(uu, i, xc) do { memset(uu, 0, sizeof(double) * ng); \
        for (int p_ = rp[i]; p_ < rp[(i) + 1]; p_++) uu[rcl[p_]] = rv[p_]; \
        if (xc) for (size_t q_ = 0; q_ < ne; q_++) if (!flag[E[q_].con]) uu[colof[q_]] -= E[q_].v * (xc)[E[q_].con]; } while (0)
    char *ok = malloc((size_t)ch + 1);
    double t_solve = 0;
    for (int c0 = 0; c0 < nfl; c0 += ch) {
        const int nc = nfl - c0 < ch ? nfl - c0 : ch;
        memset(x, 0, sizeof(double) * (size_t)m * nc); memset(ok, 0, (size_t)nc);
        for (int pass = 0; pass < 2; pass++) {
            /* (each candidate on its own, on the threads: the same result at any thread count) */
            #pragma omp parallel num_threads(nth) if (nth > 1 && nc > 1)
            {
                int tid = 0;
#ifdef _OPENMP
                tid = omp_get_thread_num();
#endif
                double *uu = ubuf + (size_t)tid * ((size_t)ng + 1);
                #pragma omp for schedule(dynamic, 1)
                for (int c = 0; c < nc; c++) {
                    const int i = fl[c0 + c]; double *xc = x + (size_t)c * m, *gc = g + (size_t)c * m;
                    if (ok[c]) { memset(gc, 0, sizeof(double) * m); continue; }
                    if (pass) DEP_RESID(uu, i, xc); else DEP_RESID(uu, i, (double *)NULL);
                    for (int j = 0; j < m; j++) {
                        double t = 0;
                        if (!flag[j]) for (int p = rp[j]; p < rp[j + 1]; p++) t += rv[p] * uu[rcl[p]];
                        gc[j] = t;
                    }
                }
            }
            { const double ts_ = wtime(); schol_solve_many_t(S, g, nc, wk); t_solve += wtime() - ts_; }
            #pragma omp parallel num_threads(nth) if (nth > 1 && nc > 1)
            {
                int tid = 0;
#ifdef _OPENMP
                tid = omp_get_thread_num();
#endif
                double *uu = ubuf + (size_t)tid * ((size_t)ng + 1);
                #pragma omp for schedule(dynamic, 1)
                for (int c = 0; c < nc; c++) {
                    if (ok[c]) continue;
                    double *xc = x + (size_t)c * m, *gc = g + (size_t)c * m;
                    for (int j = 0; j < m; j++) if (!flag[j]) xc[j] += gc[j];
                    const int i = fl[c0 + c];
                    DEP_RESID(uu, i, xc);
                    double vv = 0; for (size_t q = 0; q < ng; q++) vv += uu[q] * uu[q];
                    snc[c] = sqrt(vv / d0[i]);
                    /* (the refinement only where the first residual is not already far below stol) */
                    if (pass == 0 && snc[c] <= 1e-3 * stol) ok[c] = 1;
                    if (pass == 1 || ok[c]) {
                        double r = bb[i];
                        for (int j = 0; j < m; j++) if (!flag[j]) r -= xc[j] * bb[j];
                        relc[c] = fabs(r) / sqrt(d0[i]);
                    }
                }
            }
        }
        for (int c = 0; c < nc; c++) {
            const int i = fl[c0 + c]; const double sn = snc[c], rel = relc[c];
            if (sn > stol) { nkeep++; if (sn < skeep) skeep = sn; continue; }
            if (rel > ctol) { nbad++; continue; }
            dead[i] = 1; nd++;
            if (sn > sdep) sdep = sn;
            if (rel > rd) rd = rel;
        }
    }
    free(ok); free(rv); free(rcl); free(ubuf); free(snc); free(relc);
    if (getenv("BRISK_DEPDBG")) printf("   [... verification: solves %.2fs]\n", t_solve);
    #undef DEP_RESID
    free(fl);
    free(g); free(flag);
    if (getenv("BRISK_DEPDBG")) printf("   [... verified: %d of %d flagged dependent (largest sine %.1e), %d kept (smallest sine %.1e), %d inconsistent; %.2fs]\n", nd, nf, sdep, nkeep, skeep, nbad, wtime() - t0);
    if (nbad > 0) {
        *ninc = nbad;
        if (verbose >= 0) printf("presolve: %d equality constraint%s linear combinations of others with different right-hand sides: the constraints are inconsistent and the problem has no feasible point\n", nbad, nbad > 1 ? "s are" : " is");
        goto out;
    }
    if (nd == 0) { if (verbose >= 2 && !npar) printf("presolve: no dependent constraints (%d flagged by the Gram matrix, none confirmed)\n", nf); goto out; }
    { int *clev = calloc((size_t)mfull + 1, sizeof(int));
      for (int i = 0; i < m; i++) if (dead[i]) pdead[cmap[i]] = 1;
      compact_removed(P, pdead, clev); free(clev); }
    removed = nd + npar;
    if (verbose >= 1) printf("presolve: %d linearly dependent equality constraints removed (%d multiples of another, %d more by the factorization; rank %d of %d; distance to the span of the others at most %.1e of their norm, right-hand sides consistent to %.1e)\n",
                             nd + npar, npar, nd, mfull - nd - npar, mfull, sdep, fmax(rd / (bsc > 0 ? bsc : 1), rpar / (bsc0 > 0 ? bsc0 : 1)));
out:
    if (!removed && npar > 0 && *ninc == 0) {   /* (the multiples alone: the rest has no dependence, or was not checked) */
        { int *clev = calloc((size_t)mfull + 1, sizeof(int)); compact_removed(P, pdead, clev); free(clev); }
        removed = npar;
        if (verbose >= 1) printf("presolve: %d linearly dependent equality constraints removed (multiples of another; right-hand sides consistent to %.1e)\n", npar, rpar / (bsc0 > 0 ? bsc0 : 1));
    }
    free(pdead);
    if (S) schol_free(S);
    free(E); free(colof); free(gs); free(rp); free(ri); free(deg); free(nbr); free(pool); free(mark); free(Gacc);
    free(acc); free(d0); free(piv); free(x); free(u); free(wk); free(dead); free(cmap); free(bb);
    return removed;
}
