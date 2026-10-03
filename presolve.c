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
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

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
    for (size_t k = 0; k < ne; k++)
        if (E[k].r > E[k].c) { int t = E[k].r; E[k].r = E[k].c; E[k].c = t; }
    qsort(E, ne, sizeof(Ent), cmp_ent);
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

static int psd_nullspace(const SpSym *A, int n, int smax, double *sgn, int **Sout, int *sout,
                         double **Nout, int *kout, double **Uout) {
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
    double ztol = 1e-10 * emax;
    int neg = 0, pos_ = 0, k = 0;
    for (int i = 0; i < s; i++) {
        if (ev[i] < -ztol) neg++;
        else if (ev[i] > ztol) pos_++;
        else k++;
    }
    if (emax == 0 || (neg && pos_)) { free(T); free(ev); return 0; }
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
            memset(MSS, 0, sizeof(double) * (size_t)s * s);
            int nrows = 0;
            int *rows = malloc(sizeof(int) * n);
            for (int q = 0; q < M->ef; q++) {
                int r = M->fr[q], c = M->fc[q];
                if (inS[r] && inS[c]) MSS[spos[r] + (size_t)spos[c] * s] = M->fv[q];
                else if (!inS[r] && inS[c]) {
                    if (!rowtouch[r]) { rowtouch[r] = 1; rows[nrows++] = r; }
                    rowS[(size_t)r * s + spos[c]] = M->fv[q];
                } else if (!inS[r] && !inS[c] && r <= c) {
                    E[ne].r = map[r]; E[ne].c = map[c]; E[ne].v = M->fv[q]; ne++;
                }
            }
            if (k > 0) {
                /* N' M_SS N */
                BL(dgemm_)("N", "N", &s, &k, &s, &d1, MSS, &s, N, &s, &d0, tmp, &s);
                BL(dgemm_)("T", "N", &k, &k, &s, &d1, N, &s, tmp, &s, &d0, red, &k);
                for (int a = 0; a < k; a++)
                    for (int b2 = a; b2 < k; b2++) {
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
    free(inS); free(spos); free(map); free(MSS); free(tmp); free(red); free(rowS); free(rowtouch);
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

static void remove_constraint(Problem *P, int i) {
    for (int k = 0; k < P->nblk; k++) {
        Block *B = &P->blk[k];
        int w = 0;
        for (int t = 0; t < B->ncon; t++) {
            if (B->con[t] == i) { sp_release(&B->A[t]); continue; }
            B->A[w] = B->A[t];
            B->con[w] = B->con[t] > i ? B->con[t] - 1 : B->con[t];
            w++;
        }
        B->ncon = w;
    }
    memmove(P->b + i, P->b + i + 1, sizeof(double) * (P->m - i - 1));
    memmove(P->b0 + i, P->b0 + i + 1, sizeof(double) * (P->m - i - 1));
    memmove(P->orig + i, P->orig + i + 1, sizeof(int) * (P->m - i - 1));
    P->m--;
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
    int progress = 1, maxlev = 0;
    int *clev = calloc(P->m + 1, sizeof(int));
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
        const int m_sweep = P->m;
        for (int i = 0; i < P->m; i++) {
            if (P->b[i] != 0.0) continue;
            int np = 0, ok = 1;
            double sgn0 = 0;
            /* constraints are removed in increasing order within a sweep, each shifting the
             * later ones down by one: i is sweep-start constraint i + (removed so far) */
            const int i0 = i + (m_sweep - P->m);
            const int nk = cbp[i0 + 1] - cbp[i0];
            for (int kk = 0; kk < nk && ok; kk++) {
                const int k = cbk[cbp[i0] + kk];
                int t = find_part(&P->blk[k], i);
                if (t < 0) continue;
                const Block *B = &P->blk[k];
                double sg;
                int kind;
                if (B->type == BLK_LP) { if (!samesign_diag(&B->A[t], &sg)) ok = 0; kind = 2; }
                else if (samesign_diag(&B->A[t], &sg)) kind = 1;
                else if (rank1_test(&B->A[t], B->n, a, &sg)) kind = 0;
                else {
                    /* eigen test on the support; bounded work (support^3 plus fill) */
                    int *Sx = NULL, sx = 0, kx = 0; double *Nx = NULL;
                    kind = -1;
                    if (B->A[t].nr <= 400 && psd_nullspace(&B->A[t], B->n, 400, &sg, &Sx, &sx, &Nx, &kx, NULL)) kind = 3;
                    free(Sx); free(Nx);
                    if (kind < 0) ok = 0;
                }
                if (!ok) break;
                if (sgn0 != 0 && sg != sgn0) { ok = 0; break; }
                sgn0 = sg;
                parts[2 * np] = k; parts[2 * np + 1] = kind; np++;
            }
            if (!ok) continue;
            /* apply (and log the reduction for postsolve) */
            PSRecord *rec = g_ps_P ? ps_record_begin(P, P->orig[i], sgn0 != 0 ? sgn0 : 1.0) : NULL;
            const int L = clev[i] + 1;
            if (L > maxlev) maxlev = L;
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
                    block_rank1_reduce(B, a, p);
                    reduced_dims++;
                } else if (parts[2 * q + 1] == 3) {
                    int *Sx = NULL, sx = 0, kx = 0; double *Nx = NULL, *Ux = NULL, sg;
                    if (psd_nullspace(&B->A[t], B->n, 400, &sg, &Sx, &sx, &Nx, &kx, &Ux)) {
                        if (rec) ps_record_part(P, rec, B, PS_FACE, NULL, NULL, 0, Sx, sx, Nx, kx, Ux);
                        block_face_reduce(B, Sx, sx, Nx, kx);
                        reduced_dims += sx - kx;
                    }
                    free(Sx); free(Nx); free(Ux);
                } else {
                    char *del = calloc(B->n, 1);
                    const SpSym *S = &B->A[t];
                    for (int e = 0; e < S->nnz; e++) del[S->row[e]] = 1;
                    reduced_dims += S->nnz;
                    if (rec) ps_record_part(P, rec, B, B->type == BLK_LP ? PS_LP : PS_DIAG, del, NULL, 0, NULL, 0, NULL, 0, NULL);
                    block_delete_indices(B, del);
                    free(del);
                }
            }
            remove_constraint(P, i);
            memmove(clev + i, clev + i + 1, sizeof(int) * (P->m - i));
            removed++;
            progress = 1;
            prune_empty_parts(P);
            i--;                       /* constraint i + 1 moved to position i */
        }
        free(cbp); free(cbk);
    }
    free(a); free(parts); free(clev); free(mask);
    P->fr_removed = removed;
    P->fr_depth = maxlev;
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
    if (verbose && (removed || unused))
        printf("presolve: facial reduction removed %d constraint(s), %d cone dimension(s), depth %d; %d unused index(es)\n",
               removed, reduced_dims, maxlev, unused);
    return removed;
}
