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
double g_lf_time = 0;     /* (the detection's time, for BRISK_LFSCAN) */
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
        for (int i = 0; i < P->m; i++) {
            if (P->b[i] != 0.0) continue;
            int np = 0, ok = 1;
            double sgn0 = 0;
            const int i0 = i;
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
                } else if (parts[2 * q + 1] == 3) {
                    int *Sx = NULL, sx = 0, kx = 0; double *Nx = NULL, *Ux = NULL, sg;
                    if (psd_nullspace(&B->A[t], B->n, 400, &sg, &Sx, &sx, &Nx, &kx, &Ux)) {
                        if (rec) ps_record_part(P, rec, B, PS_FACE, NULL, NULL, 0, Sx, sx, Nx, kx, Ux);
                        coupled |= fr_coupled(B, t, 3, NULL, NULL, Sx, sx, Nx, kx, Ux);
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
                    coupled |= fr_coupled(B, t, B->type == BLK_LP ? 2 : 1, del, NULL, NULL, 0, NULL, 0, NULL);
                    block_delete_indices(B, del);
                    free(del);
                }
            }
            if (rec) rec->coupled = coupled;
            if (has_comb) for (int r = 0; r < P->m; r++) if (P->orig[r] <= -2) clev[r] = 0;
            ncoupled += coupled;
            remove_parts_of(P, i, parts, np);
            dead[i] = 1; ndead++;
            removed++;
            progress = 1;
        }
        if (ndead) compact_removed(P, dead, clev);
        free(dead);
        free(cbp); free(cbk);
    }
    free(a); free(parts); free(clev); free(mask);
    P->fr_removed = removed;
    P->fr_depth = maxlev;
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
