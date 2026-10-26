/*
 * Postsolve: map the solution of the presolved problem back to the problem as read,
 * and measure it there.
 *
 * Facial reduction (presolve.c) substitutes X = V W V' block by block and removes
 * constraints <A_r, X> = 0 whose part s A_r is positive semidefinite. Composing the
 * substitutions gives X_orig = T X_red T' with T stored by sparse columns (PSBasis).
 *
 * The removed constraints have b_r = 0, so their multipliers do not change b'y, but
 * with y_r = 0 the dual slack Z = C - A'y is positive semidefinite only on the face.
 * The multipliers are recovered in reverse order of the reductions. For a reduction
 * with previous basis T_p, new coordinates V (the face) and a complement W,
 *     [V W]' (Z + t s A_r) [V W] = [[Zn, G], [G', H + t Aw]],
 * with Zn = V'T_p'ZT_pV (the dual slack of the reduced space, positive definite by
 * induction), G = V'T_p'ZT_pW, H = W'T_p'ZT_pW and Aw = s (T_pW)'A_r(T_pW) > 0.
 * By the Schur complement the matrix is PSD iff t >= lambda_max(Aw^{-1/2}(G'Zn^{-1}G - H)
 * Aw^{-1/2}); y_r = -s t. Deleted LP variables give t >= max(-z_j / (s a_j)).
 */
#include "brisk.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <time.h>

static const int IONE_ = 1;
static const double DONE_ = 1.0, DZERO_ = 0.0;

static void *pmalloc(size_t n) {
    void *p = malloc(n ? n : 1);
    if (!p) { fprintf(stderr, "brisk: out of memory in postsolve (%zu bytes)\n", n); exit(1); }
    return p;
}
static void *pcalloc(size_t n, size_t s) {
    void *p = calloc(n ? n : 1, s);
    if (!p) { fprintf(stderr, "brisk: out of memory in postsolve\n"); exit(1); }
    return p;
}
int g_ps_need_x = 0;      /* set by main: the caller needs X itself (files, crossover) */
int g_ps_fr_correct = 0;  /* set by main (5.17, ISSUES 93): correct the lift of the reduced dual through complementarity when the
                           * certificate passes fail; opt-in since the owner's policy of 16 October 2026 (main.c sets it from BRISK_FRFIX=1),
                           * so that the message that the recovery failed is unambiguous and the default path pays nothing */
static double ps_now(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + 1e-9 * ts.tv_nsec;
}
/* 24 October 2026 (ISSUES 284): the time the run has left when the postsolve starts (main.c sets it before each call;
 * INFINITY: no limit). Past it the optional work is skipped: the recovery of the duals of the constraints the facial
 * reduction removed (each of its passes measures lambda_min of the original blocks: 288 s on pglib_opf_case13659_pegase,
 * SDP, blocks of order 8,185 and 27,317, after a solve stopped at iteration 0 by a 3 s limit) and the projection of X onto
 * A(X) = b. The mapping back and the errors' measure still run. */
static double g_ps_deadline = 0;
void postsolve_set_time_left(double left) { g_ps_deadline = left < INFINITY ? ps_now() + left : 0; }
static int ps_late(void) { return g_ps_deadline > 0 && ps_now() > g_ps_deadline; }

/* ------------------------------------------------------------------ basis */
static void basis_free(PSBasis *T) {
    if (!T->identity && T->idx) {
        for (int c = 0; c < T->n_cur; c++) { free(T->idx[c]); free(T->val[c]); }
    }
    free(T->idx); free(T->val); free(T->nz);
    T->idx = NULL; T->val = NULL; T->nz = NULL;
}
static void basis_materialize(PSBasis *T) {
    if (!T->identity) return;
    int n = T->n_cur;
    T->nz = pmalloc(sizeof(int) * (n + 1));
    T->idx = pmalloc(sizeof(int *) * (n + 1));
    T->val = pmalloc(sizeof(double *) * (n + 1));
    for (int c = 0; c < n; c++) {
        T->nz[c] = 1;
        T->idx[c] = pmalloc(sizeof(int)); T->idx[c][0] = c;
        T->val[c] = pmalloc(sizeof(double)); T->val[c][0] = 1.0;
    }
    T->identity = 0;
}
static size_t basis_nnz(const PSBasis *T) {
    if (T->identity) return 0;
    size_t s = 0;
    for (int c = 0; c < T->n_cur; c++) s += T->nz[c];
    return s;
}
static void basis_copy(PSBasis *D, const PSBasis *S) {
    *D = *S;
    D->idx = NULL; D->val = NULL; D->nz = NULL;
    if (S->identity) return;
    int n = S->n_cur;
    D->nz = pmalloc(sizeof(int) * (n + 1));
    D->idx = pmalloc(sizeof(int *) * (n + 1));
    D->val = pmalloc(sizeof(double *) * (n + 1));
    for (int c = 0; c < n; c++) {
        D->nz[c] = S->nz[c];
        D->idx[c] = pmalloc(sizeof(int) * (S->nz[c] + 1));
        D->val[c] = pmalloc(sizeof(double) * (S->nz[c] + 1));
        memcpy(D->idx[c], S->idx[c], sizeof(int) * S->nz[c]);
        memcpy(D->val[c], S->val[c], sizeof(double) * S->nz[c]);
    }
}
/* column c of T as a dense vector (length n_orig) */
static void basis_col(const PSBasis *T, int c, double *out) {
    memset(out, 0, sizeof(double) * T->n_orig);
    if (T->identity) { out[c] = 1.0; return; }
    for (int e = 0; e < T->nz[c]; e++) out[T->idx[c][e]] = T->val[c][e];
}
/* set column from a dense accumulator (clears it) */
static void col_from_dense(double *acc, int n, int **idx, double **val, int *nz) {
    int cnt = 0;
    for (int i = 0; i < n; i++) if (acc[i] != 0) cnt++;
    *idx = pmalloc(sizeof(int) * (cnt + 1));
    *val = pmalloc(sizeof(double) * (cnt + 1));
    int w = 0;
    for (int i = 0; i < n; i++) if (acc[i] != 0) { (*idx)[w] = i; (*val)[w] = acc[i]; w++; acc[i] = 0; }
    *nz = cnt;
}
static void basis_delete(PSBasis *T, const char *del) {
    int n = T->n_cur, w = 0;
    if (T->identity) basis_materialize(T);
    for (int c = 0; c < n; c++) {
        if (del[c]) { free(T->idx[c]); free(T->val[c]); continue; }
        T->idx[w] = T->idx[c]; T->val[w] = T->val[c]; T->nz[w] = T->nz[c]; w++;
    }
    T->n_cur = w;
}
static void basis_rank1(PSBasis *T, const double *wv, int p) {
    basis_materialize(T);
    int n = T->n_cur, no = T->n_orig;
    double *acc = pcalloc(no, sizeof(double));
    int **ni = pmalloc(sizeof(int *) * n);
    double **nv = pmalloc(sizeof(double *) * n);
    int *nz = pmalloc(sizeof(int) * n);
    int w = 0;
    for (int r = 0; r < n; r++) {
        if (r == p) continue;
        for (int e = 0; e < T->nz[r]; e++) acc[T->idx[r][e]] += T->val[r][e];
        if (wv[r] != 0) for (int e = 0; e < T->nz[p]; e++) acc[T->idx[p][e]] += wv[r] * T->val[p][e];
        col_from_dense(acc, no, &ni[w], &nv[w], &nz[w]);
        w++;
    }
    for (int c = 0; c < n; c++) { free(T->idx[c]); free(T->val[c]); }
    free(T->idx); free(T->val); free(T->nz);
    T->idx = ni; T->val = nv; T->nz = nz; T->n_cur = n - 1;
    free(acc);
}
static void basis_face(PSBasis *T, const int *S, int s, const double *N, int k, const char *inS) {
    basis_materialize(T);
    int n = T->n_cur, no = T->n_orig, nr = 0;
    for (int i = 0; i < n; i++) nr += !inS[i];
    int nn = nr + k;
    int **ni = pmalloc(sizeof(int *) * (nn + 1));
    double **nv = pmalloc(sizeof(double *) * (nn + 1));
    int *nz = pmalloc(sizeof(int) * (nn + 1));
    int w = 0;
    for (int i = 0; i < n; i++) if (!inS[i]) { ni[w] = T->idx[i]; nv[w] = T->val[i]; nz[w] = T->nz[i]; T->idx[i] = NULL; T->val[i] = NULL; w++; }
    double *acc = pcalloc(no, sizeof(double));
    for (int a = 0; a < k; a++) {
        for (int j = 0; j < s; j++) {
            double v = N[j + (size_t)a * s];
            if (v == 0) continue;
            int c = S[j];
            for (int e = 0; e < T->nz[c]; e++) acc[T->idx[c][e]] += v * T->val[c][e];
        }
        col_from_dense(acc, no, &ni[w], &nv[w], &nz[w]);
        w++;
    }
    for (int c = 0; c < n; c++) { free(T->idx[c]); free(T->val[c]); }
    free(T->idx); free(T->val); free(T->nz);
    T->idx = ni; T->val = nv; T->nz = nz; T->n_cur = nn;
    free(acc);
}

/* ------------------------------------------------------------------ log */
Postsolve *ps_new(const Problem *P) {
    Postsolve *ps = pcalloc(1, sizeof(Postsolve));
    int nb = P->nblk;
    ps->nblk_orig = nb;
    ps->type = pmalloc(sizeof(int) * (nb + 1));
    ps->norig = pmalloc(sizeof(int) * (nb + 1));
    ps->basis = pcalloc(nb + 1, sizeof(PSBasis));
    ps->cur2orig = pmalloc(sizeof(int) * (nb + 1));
    for (int k = 0; k < nb; k++) {
        ps->type[k] = P->blk[k].type;
        ps->norig[k] = P->blk[k].n;
        ps->basis[k].n_orig = ps->basis[k].n_cur = P->blk[k].n;
        ps->basis[k].identity = 1;
        ps->cur2orig[k] = k;
    }
    ps->nblk_after = nb;
    ps->dual_ok = 1;
    ps->snap_cap = 1024.0 * 1024 * 1024;
    return ps;
}

static void part_free(PSPart *pp) {
    basis_free(&pp->T);
    free(pp->K); free(pp->J); free(pp->w); free(pp->S); free(pp->N); free(pp->U);
}
void ps_free(Postsolve *ps) {
    if (!ps) return;
    for (int r = 0; r < ps->nrec; r++) {
        for (int q = 0; q < ps->rec[r].np; q++) part_free(&ps->rec[r].parts[q]);
        free(ps->rec[r].parts);
    }
    free(ps->rec);
    for (int k = 0; k < ps->nblk_orig; k++) basis_free(&ps->basis[k]);
    free(ps->basis); free(ps->type); free(ps->norig); free(ps->cur2orig);
    for (int q = 0; q < ps->nch; q++) { free(ps->ch[q].cptr); free(ps->ch[q].cv); free(ps->ch[q].par); }
    if (ps->mom) ps_moment_free(ps->mom);
    free(ps->ch);
    free(ps->comb_ptr); free(ps->comb_i); free(ps->comb_w);
    free(ps);
}

static int orig_of(const Problem *P, const Block *B) {
    const Postsolve *ps = P->ps;
    if (!ps || !ps->blk0) return -1;
    long k = B - (const Block *)ps->blk0;
    if (k < 0 || k >= P->nblk) return -1;
    return ps->cur2orig[k];
}

void ps_hook_delete(Problem *P, const Block *B, const char *del) {
    int o = orig_of(P, B);
    if (o < 0) return;
    basis_delete(&P->ps->basis[o], del);
}
void ps_hook_rank1(Problem *P, const Block *B, const double *w, int p) {
    int o = orig_of(P, B);
    if (o < 0) return;
    basis_rank1(&P->ps->basis[o], w, p);
}
void ps_hook_face(Problem *P, const Block *B, const int *S, int s, const double *N, int k, const char *inS) {
    int o = orig_of(P, B);
    if (o < 0) return;
    basis_face(&P->ps->basis[o], S, s, N, k, inS);
}
void ps_hook_vanish(Problem *P, const Block *B) {
    int o = orig_of(P, B);
    if (o < 0) return;
    PSBasis *T = &P->ps->basis[o];
    char *del = pmalloc(T->n_cur + 1);
    memset(del, 1, T->n_cur + 1);
    basis_delete(T, del);
    free(del);
}

PSRecord *ps_record_begin(Problem *P, int con_orig, double sgn) {
    Postsolve *ps = P->ps;
    if (!ps) return NULL;
    if (ps->nrec == ps->rcap) {
        ps->rcap = ps->rcap ? 2 * ps->rcap : 64;
        ps->rec = realloc(ps->rec, sizeof(PSRecord) * ps->rcap);
        if (!ps->rec) { fprintf(stderr, "brisk: out of memory\n"); exit(1); }
    }
    PSRecord *r = &ps->rec[ps->nrec++];
    memset(r, 0, sizeof(*r));
    r->con = con_orig;
    r->sgn = sgn;
    return r;
}

void ps_record_part(Problem *P, PSRecord *r, const Block *B, int kind, const char *del,
                    const double *w, int p, const int *S, int s, const double *N, int kn, const double *U) {
    Postsolve *ps = P->ps;
    if (!ps || !r) return;
    int o = orig_of(P, B);
    if (o < 0) { ps->dual_ok = 0; return; }
    r->parts = realloc(r->parts, sizeof(PSPart) * (r->np + 1));
    PSPart *pp = &r->parts[r->np++];
    memset(pp, 0, sizeof(*pp));
    pp->kind = kind;
    pp->blk = o;
    const int n = B->n;
    pp->n_prev = n;
    if (ps->dual_ok) {
        basis_copy(&pp->T, &ps->basis[o]);
        ps->snap_bytes += 12.0 * basis_nnz(&pp->T) + 16.0 * n;
        if (ps->snap_bytes > ps->snap_cap) ps->dual_ok = 0;
    } else pp->T.identity = 1;
    if (kind == PS_DIAG || kind == PS_LP) {
        pp->K = pmalloc(sizeof(int) * (n + 1));
        pp->J = pmalloc(sizeof(int) * (n + 1));
        for (int i = 0; i < n; i++) { if (del[i]) pp->J[pp->nj++] = i; else pp->K[pp->nk++] = i; }
    } else if (kind == PS_RANK1) {
        pp->p = p;
        pp->w = pmalloc(sizeof(double) * (n + 1));
        memcpy(pp->w, w, sizeof(double) * n);
    } else {
        pp->s = s; pp->kn = kn;
        pp->S = pmalloc(sizeof(int) * (s + 1));
        memcpy(pp->S, S, sizeof(int) * s);
        pp->N = pmalloc(sizeof(double) * ((size_t)s * kn + 1));
        memcpy(pp->N, N, sizeof(double) * (size_t)s * kn);
        pp->U = pmalloc(sizeof(double) * ((size_t)s * (s - kn) + 1));
        memcpy(pp->U, U, sizeof(double) * (size_t)s * (s - kn));
        ps->snap_bytes += 8.0 * s * s;
    }
}

/* ------------------------------------------------------------------ original data */
PSOrig *ps_orig_build(const Problem *P) {
    PSOrig *O = pcalloc(1, sizeof(PSOrig));
    O->m = P->m;
    O->nblk = P->nblk;
    O->bs = pmalloc(sizeof(int) * (P->nblk + 1));
    O->b = pmalloc(sizeof(double) * (P->m + 1));
    memcpy(O->b, P->b0, sizeof(double) * P->m);
    size_t nnz = 0;
    for (int k = 0; k < P->nblk; k++) {
        const Block *B = &P->blk[k];
        O->bs[k] = B->type == BLK_LP ? -B->n : B->n;
        nnz += B->C.nnz;
        for (int t = 0; t < B->ncon; t++) nnz += B->A[t].nnz;
    }
    O->nnz = nnz;
    O->con = pmalloc(sizeof(int) * (nnz + 1));
    O->blk = pmalloc(sizeof(int) * (nnz + 1));
    O->ii = pmalloc(sizeof(int) * (nnz + 1));
    O->jj = pmalloc(sizeof(int) * (nnz + 1));
    O->v = pmalloc(sizeof(double) * (nnz + 1));
    size_t w = 0;
    double nC1 = 0;
    for (int k = 0; k < P->nblk; k++) {
        const Block *B = &P->blk[k];
        for (int t = -1; t < B->ncon; t++) {
            const SpSym *S = t < 0 ? &B->C : &B->A[t];
            for (int q = 0; q < S->nnz; q++) {
                O->con[w] = t < 0 ? -1 : B->con[t];
                O->blk[w] = k;
                O->ii[w] = S->row[q] < S->col[q] ? S->row[q] : S->col[q];
                O->jj[w] = S->row[q] < S->col[q] ? S->col[q] : S->row[q];
                O->v[w] = S->val[q];
                if (t < 0) nC1 += fabs(S->val[q]) * (S->row[q] == S->col[q] ? 1.0 : 2.0);
                w++;
            }
        }
    }
    O->nC1 = nC1;
    O->nb1 = 0;
    O->tbR = P->tbR;
    O->obj_off = P->obj_off;
    for (int i = 0; i < P->m; i++) if (!(P->tbR != 0 && P->b0[i] == P->tbR)) O->nb1 += fabs(P->b0[i]);
    return O;
}
void ps_orig_free(PSOrig *O) {
    if (!O) return;
    bound_gram_drop(O);          /* 4.42: the bound certificate's cached Gram factor of this problem */
    free(O->bs); free(O->b); free(O->con); free(O->blk); free(O->ii); free(O->jj); free(O->v);
    free(O);
}

/* ------------------------------------------------------------------ linear algebra */
static int chol_reg(int n, double *A, double *work) {
    /* Cholesky of A (lower, n x n), escalating a relative diagonal shift if needed;
     * returns 0 on success (A overwritten by the factor), 1 on failure */
    if (n == 0) return 0;
    double dmax = 0;
    for (int i = 0; i < n; i++) dmax = fmax(dmax, fabs(A[i + (size_t)i * n]));
    if (dmax == 0) dmax = 1;
    memcpy(work, A, sizeof(double) * (size_t)n * n);
    for (int attempt = 0; attempt < 9; attempt++) {
        double sh = attempt == 0 ? 0.0 : dmax * pow(10.0, -15.0 + attempt);
        if (attempt) {
            memcpy(A, work, sizeof(double) * (size_t)n * n);
            for (int i = 0; i < n; i++) A[i + (size_t)i * n] += sh;
        }
        int info;
        BL(dpotrf_)("L", &n, A, &n, &info);
        if (info == 0) return 0;
    }
    return 1;
}

/* smallest eigenvalue of symmetric M; returns 0 if M is numerically positive
 * definite (Cholesky succeeds), the exact value for n <= 4000, a Lanczos estimate
 * otherwise. work: n*n doubles. */
static double lanczos_min(int n, const double *M);
/* lambda_min when it matters: 0 if M is PSD (Cholesky), -delta if M + delta I is (a bound
 * that is all the error measures need, at a third of the cost of an eigensolver),
 * otherwise the exact value */
/* The same bounds from a sparse Cholesky when M is sparse (the dual slack of a
 * chordal-structured problem: its aggregate pattern): 0 if M is positive definite, else
 * -d for the smallest d = delta * 10^k (up to 1e-6 max|M_ii|) with M + d I definite.
 * Returns 1 (and nothing) when M is too dense or no shift up to that works. Each dense
 * Cholesky of the 3228 x 3228 AC-OPF block cost ~1 s; this costs milliseconds.        */
static int ps_lammin_sparse(int n, const double *M, double delta, double *out) {
    if (n < 400) return 1;
    size_t nz = 0;
    int *deg = calloc(n + 1, sizeof(int));
    for (int j = 0; j < n; j++)
        for (int i = j + 1; i < n; i++) if (M[i + (size_t)j * n] != 0) { deg[i]++; deg[j]++; nz++; }
    if ((double)nz > 0.03 * 0.5 * (double)n * n) { free(deg); return 1; }
    int **nbr = malloc(sizeof(int *) * (n + 1)), *fill = calloc(n + 1, sizeof(int));
    for (int i = 0; i < n; i++) nbr[i] = malloc(sizeof(int) * (deg[i] + 1));
    for (int j = 0; j < n; j++)
        for (int i = j + 1; i < n; i++) if (M[i + (size_t)j * n] != 0) { nbr[i][fill[i]++] = j; nbr[j][fill[j]++] = i; }
    SChol *S = schol_analyze_adj(n, deg, nbr, (size_t)fmin(3e7 * brisk_mem_scale(), 0.3 * (double)n * n) + 1000);
    for (int i = 0; i < n; i++) free(nbr[i]);
    free(nbr); free(fill); free(deg);
    if (!S) return 1;
    double dmax = 0;
    for (int i = 0; i < n; i++) dmax = fmax(dmax, fabs(M[i + (size_t)i * n]));
    /* entries and their offsets in the factor storage, once */
    size_t ne = nz + (size_t)n, w = 0;
    size_t *off = malloc(sizeof(size_t) * (ne + 1));
    double *val = malloc(sizeof(double) * (ne + 1));
    size_t *doff = malloc(sizeof(size_t) * (n + 1));
    for (int j = 0; j < n; j++) {
        doff[j] = schol_offset(S, j, j);
        for (int i = j + 1; i < n; i++) { const double v = M[i + (size_t)j * n]; if (v != 0) { off[w] = schol_offset(S, i, j); val[w] = v; w++; } }
    }
    int rc = 1;
    double *pm = schol_values(S);
    /* bisection-free ladder, but starting where the violation is: a first try at 0, then
     * decades from delta */
    for (int k = -1; k <= 12; k++) {
        const double d = k < 0 ? 0.0 : delta * pow(10.0, k);
        if (k >= 0 && d > 1e-6 * (dmax > 0 ? dmax : 1.0)) break;
        schol_zero(S);
        pm = schol_values(S);
        for (size_t q = 0; q < w; q++) pm[off[q]] += val[q];
        for (int j = 0; j < n; j++) pm[doff[j]] += M[j + (size_t)j * n] + d;
        if (schol_factor(S, 0.0) == 0) { *out = -d; rc = 0; break; }
        if (k < 0 && !(delta > 0)) break;
    }
    free(off); free(val); free(doff);
    schol_free(S);
    return rc;
}

/* ---- 23 October 2026: X of a block with arrow data held as its cone vector (brisk.h, xc_index). A second-order cone
 * read from a SeDuMi file (K.q) is written as a semidefinite block with arrow data and solved as a cone; its dense
 * completion, of order up to 17,012 on the owner's dg3d family (SDP with SOCP), made every pass of the postsolve, the
 * measures and the result a pass over n^2 entries (dg3d_1000: 92 s of postsolve in 300 s, 9.2 GB). Where nothing maps
 * the block (main.c decides), it stays a vector of n entries from the solve to the output. */
int g_xcone_accept = 0;
int *g_xcone_res = NULL;
static double xc_nrm2c(int n, const double *x) {
    /* |x_1..x_{n-1}| by compensated summation (the completion's trace reads it: see xc_complete) */
    double s2 = 0, s2c = 0;
    for (int i = 1; i < n; i++) { const double y = x[i] * x[i], t = s2 + y; s2c += fabs(s2) >= fabs(y) ? (s2 - t) + y : (y - t) + s2; s2 = t; }
    return sqrt(s2 + s2c);
}
void xc_complete(int n, int h, const double *x, double *X) {
    /* (22 October 2026: |xb|^2 compensated. The lower block's trace f |xb|^2 must be x_0 / 2 to working accuracy, since
     * <A, X> and the SeDuMi x_0 = tr X read it: on a ray of norm 1e18 (dg3d_1000) the plain sum's error was 3e6 in A(X)) */
    const double nb = xc_nrm2c(n, x);
    const size_t hh = (size_t)h;
    if (!(nb > 0)) {
        memset(X, 0, sizeof(double) * (size_t)n * n);
        X[hh + hh * n] = 0.5 * x[0];
        for (int t = 1; t < n; t++) { const size_t o = (size_t)xc_index(h, t); X[hh + o * n] = X[o + hh * n] = 0.5 * x[t]; X[o + o * n] = 0.5 * x[0] / (n - 1); }
        return;
    }
    const double f = 0.5 * x[0] / (nb * nb);
    X[hh + hh * n] = 0.5 * x[0];
    for (int t = 1; t < n; t++) X[(size_t)xc_index(h, t) + hh * n] = 0.5 * x[t];
    #pragma omp parallel for schedule(static) if (n >= 512)
    for (int t = 1; t < n; t++) {
        const size_t o = (size_t)xc_index(h, t);
        double *c = X + o * n;
        c[hh] = 0.5 * x[t];
        for (int r = 1; r < n; r++) c[xc_index(h, r)] = f * x[r] * x[t];
    }
}
void xc_of_dense(int n, int h, const double *X, double *x) {
    double tr = 0, trc = 0;      /* (compensated: the completion's diagonal holds x_0 / 2 at the hub and n - 1 small terms) */
    for (int i = 0; i < n; i++) { const double v = X[(size_t)i * n + i], t = tr + v; trc += fabs(tr) >= fabs(v) ? (tr - t) + v : (v - t) + tr; tr = t; }
    x[0] = tr + trc;
    for (int t = 1; t < n; t++) x[t] = 2.0 * X[(size_t)xc_index(h, t) + (size_t)h * n];
}
/* the completion's eigenvalues are (x_0 +- |xb|) / 2 and 0 */
double xc_lammin_x(int n, const double *x) { return fmin(0.0, 0.5 * (x[0] - xc_nrm2c(n, x))); }
/* Z = Arw(z): eigenvalues z_0 +- |zb| and z_0 */
static double xc_lammin_z(int n, const double *z) { return fmin(0.0, z[0] - xc_nrm2c(n, z)); }
int xc_expand(const PSOrig *O, double **Xo, int *xc) {
    if (!xc || !Xo) return 0;
    for (int o = 0; o < O->nblk; o++) {
        if (xc[o] < 0 || !Xo[o]) continue;
        const int n = O->bs[o];
        double *D = malloc(sizeof(double) * ((size_t)n * n + 1));
        if (!D) return 1;
        xc_complete(n, xc[o], Xo[o], D);
        free(Xo[o]); Xo[o] = D; xc[o] = -1;
    }
    return 0;
}
int ps_arrow_hub(const PSOrig *O, int o, const int *kept) {
    const int n = O->bs[o];
    if (n < 2) return -1;
    /* per matrix (C and each row): its diagonal value and how many diagonal entries it has; the hub candidates are the
     * indices in every off-diagonal entry (the test of solver.c's arrow_hub, on the triplets of the data as read) */
    double *dv = malloc(sizeof(double) * ((size_t)O->m + 2)); int *nd = calloc((size_t)O->m + 2, sizeof(int));
    int c1 = -1, c2 = -1, offd = 0, ok = dv && nd;
    for (size_t q = 0; q < O->nnz && ok; q++) {
        if (O->blk[q] != o) continue;
        const int r = O->con[q] + 1, i = O->ii[q], j = O->jj[q];
        if (r > 0 && kept && !kept[r - 1]) continue;
        const double v = O->v[q];
        if (i == j) {
            if (nd[r] == 0) dv[r] = v;
            else if (!(fabs(v - dv[r]) <= 1e-14 * fabs(dv[r]))) ok = 0;
            nd[r]++;
            continue;
        }
        offd++;
        if (c1 < 0 && c2 < 0) { c1 = i; c2 = j; continue; }
        if (c1 >= 0 && c1 != i && c1 != j) c1 = -1;
        if (c2 >= 0 && c2 != i && c2 != j) c2 = -1;
        if (c1 < 0 && c2 < 0) ok = 0;
    }
    for (int r = 0; r <= O->m && ok; r++) if (nd[r] != 0 && nd[r] != n) ok = 0;
    free(dv); free(nd);
    if (!ok || !offd) return -1;
    return c1 >= 0 ? c1 : c2;
}

static double g_lam_cap = INFINITY;   /* largest shift the ladder may report (absolute) */
static double lammin_eig(int n, const double *M, double *work);
/* 5.16: lambda_min in closed form for the two matrices a second-order cone read as a semidefinite
 * block leaves in the measures (n x n, column-major, full): 1 with *lam set to a lower bound within
 * delta, 0 when M has neither form (a general dense matrix fails after O(n) work).
 *  - an arrow matrix [[z0, zb'], [zb, z0 I]] (any hub h): eigenvalues z0 +- ||zb|| and z0. Entries off
 *    the arrow must be exact zeros (a Z built from arrow data has them); a spread s of the diagonal is
 *    allowed up to delta: lambda_min >= min diag - ||zb||, within s.
 *  - the rank-two completion [[a, b'], [b, (a/||b||^2) b b']] of a cone point (arrow_complete): on
 *    span{e_h, b} the 2 x 2 matrix [[a, ||b||], [||b||, a]], zero elsewhere, so lambda_min =
 *    min(0, a - ||b||). Its hub is the index whose diagonal is half the trace. A deviation e (largest
 *    entry) of the other block from the rank-one form moves the eigenvalues by at most n e; accepted
 *    when n e <= delta, and subtracted. */
static int ps_lammin_arrow(int n, const double *M, double delta, double *lam) {
    if (n < 64 || delta < 0) return 0;   /* (22 October 2026: delta = 0 too, for the completion: see below) */
    /* arrow: the hub is the one column with two or more off-diagonal entries */
    int h = -1;
    for (int j = 0; j < n; j++) {
        const double *c = M + (size_t)j * n;
        int cnt = 0;
        for (int i = 0; i < n && cnt < 2; i++) if (i != j && c[i] != 0) cnt++;
        if (cnt >= 2) { if (h >= 0) { h = -2; break; } h = j; }
    }
    if (h >= 0) {
        const double *ch = M + (size_t)h * n;
        int ok = 1;
        for (int j = 0; j < n && ok; j++) {
            if (j == h) continue;
            const double *c = M + (size_t)j * n;
            for (int i = 0; i < n; i++) if (i != j && i != h && c[i] != 0) { ok = 0; break; }
            if (ok && c[h] != ch[j]) ok = 0;
        }
        if (ok) {
            double dmin = M[0], dmax = M[0], nb = 0;
            for (int i = 0; i < n; i++) { const double d = M[i + (size_t)i * n]; dmin = fmin(dmin, d); dmax = fmax(dmax, d); if (i != h) nb += ch[i] * ch[i]; }
            if (dmax - dmin <= delta && delta > 0) { *lam = fmin(0.0, dmin - sqrt(nb)); return 1; }
            if (dmax == dmin) { *lam = fmin(0.0, dmin - sqrt(nb)); return 1; }    /* (exact: delta = 0 asks no more) */
            return 0;
        }
    }
    /* the completion */
    double tr = 0, dmx = -HUGE_VAL; int hc = -1;
    for (int i = 0; i < n; i++) { const double d = M[i + (size_t)i * n]; tr += d; if (d > dmx) { dmx = d; hc = i; } }
    if (hc < 0 || !(dmx > 0) || fabs(2.0 * dmx - tr) > 1e-12 * fabs(tr)) return 0;
    const double *b = M + (size_t)hc * n;
    double nb2 = 0;
    for (int i = 0; i < n; i++) if (i != hc) nb2 += b[i] * b[i];
    if (!(nb2 > 0)) return 0;
    const double g = dmx / nb2;
    /* 22 October 2026: the deviation E of the other block from g b b' bounds the eigenvalues' change by ||E||_2 <=
     * ||E||_F, summed here; the old bound n max|E_ij| is n times larger for deviations of rounding size, and on a ray
     * of norm 1e18 (dg3d_1000, SDP with SOCP, one cone of dimension 17,012) it rejected the closed form, so the measures
     * fell to dense Cholesky factorizations of order 17,012 (the postsolve 89 s of a 292 s run). Accepted when the bound
     * is at most delta, or of rounding size (64 eps of the block's trace) when delta = 0 asks for the value itself */
    const double fcap = delta > 0 ? delta : 64.0 * DBL_EPSILON * fabs(dmx);
    double f2 = 0;
    for (int j = 0; j < n; j++) {
        if (j == hc) continue;
        const double *c = M + (size_t)j * n, gb = g * b[j];
        for (int i = 0; i < n; i++) { if (i == hc) continue; const double d = c[i] - gb * b[i]; f2 += d * d; }
        if (f2 > fcap * fcap) return 0;
    }
    const double ef = sqrt(f2) + 4.0 * DBL_EPSILON * fabs(dmx);    /* (and the rounding of g and of the closed form) */
    if (ef > fcap && delta > 0) return 0;
    *lam = fmin(0.0, dmx - sqrt(nb2)) - ef;
    return 1;
}
double ps_lammin_tol(int n, const double *M, double *work, double delta) {
    { double v; if (ps_lammin_arrow(n, M, delta, &v)) return v; }
    if (n > 1 && delta > 0) { double v; if (ps_lammin_sparse(n, M, delta, &v) == 0) return v; }
    if (n <= 1 || !(delta > 0)) return ps_lammin(n, M, work);
    /* 5.12: the verdicts by brisk_dpotrf (tiled on the solver's threads): LAPACK's dpotrf ran on
     * one thread, 0.8 s at order 5000 for the test of X alone */
    memcpy(work, M, sizeof(double) * (size_t)n * n);
    int info;
    info = brisk_dpotrf(n, work, n);
    if (info == 0) return 0;
    memcpy(work, M, sizeof(double) * (size_t)n * n);
    for (int i = 0; i < n; i++) work[i + (size_t)i * n] += delta;
    info = brisk_dpotrf(n, work, n);
    if (info == 0) return -delta;
    if (n <= 500) return lammin_eig(n, M, work);   /* 4.42: small blocks: the eigenvalue itself (cheap, exact) */
    if (n > 4000) {
        /* 4.42: large blocks. Until 4.41 the capped ladder below was followed by a 120-step
         * Lanczos estimate "Ritz value - residual", which is neither converged nor a bound:
         * on the near-low-rank X of AC-OPF it read -2.4e-4 for a lambda_min of -1.5e-7 (case1888
         * with X requested: NUMERICAL DIFFICULTIES and a re-solve chain for a point good to
         * 7.5e-8). Now the Cholesky ladder runs on in decades of delta, up to 1e9 delta (an
         * error of 1e-3, beyond which the value is only a verdict), and three bisections
         * tighten the result: lambda_min >= -d with d at most 1.33 times too large. Each
         * step is one Cholesky (1 s at n = 4 357, 4 s at 8 185); only a matrix worse than
         * 1e-3 pays for the eigenvalue itself. */
        double d = delta;
        for (int k = 1; k <= 9; k++) {
            d *= 10;
            memcpy(work, M, sizeof(double) * (size_t)n * n);
            for (int i = 0; i < n; i++) work[i + (size_t)i * n] += d;
            info = brisk_dpotrf(n, work, n);
            if (info == 0) {
                /* three bisections in the exponent: the bound is within a factor 1.33 */
                double lo = d / 10, hi = d;
                for (int b = 0; b < 3; b++) {
                    const double h = sqrt(lo * hi);
                    memcpy(work, M, sizeof(double) * (size_t)n * n);
                    for (int i = 0; i < n; i++) work[i + (size_t)i * n] += h;
                    info = brisk_dpotrf(n, work, n);
                    if (info == 0) hi = h; else lo = h;
                }
                return -hi;
            }
        }
        return lammin_eig(n, M, work);
    }
    g_lam_cap = 1e3 * delta;          /* the bound must stay below what the tolerance measures */
    const double v = ps_lammin(n, M, work);
    g_lam_cap = INFINITY;
    return v;
}

double ps_lammin(int n, const double *M, double *work) {
    if (n <= 0) return 0;
    if (n == 1) return M[0] < 0 ? M[0] : 0;
    memcpy(work, M, sizeof(double) * (size_t)n * n);
    int info;
    BL(dpotrf_)("L", &n, work, &n, &info);
    if (info == 0) return 0;
    /* a boundary point (the optimum is singular): a Cholesky of M + delta I proves
     * lambda_min >= -delta, a conservative bound below any tolerance of interest, at a
     * third of the cost of the eigenvalue (dsyevr's reduction dominated the postsolve of
     * the large chordal problems) */
    {
        double dmax = 0;
        for (int i = 0; i < n; i++) dmax = fmax(dmax, fabs(M[i + (size_t)i * n]));
        for (double rel = 1e-12; rel <= 1.01e-8; rel *= 10) {
            const double delta = rel * (dmax > 0 ? dmax : 1.0);
            if (delta > g_lam_cap) break;     /* 4.20: not a measurement any more (hinf11: dmax 1.5e9) */
            memcpy(work, M, sizeof(double) * (size_t)n * n);
            for (int i = 0; i < n; i++) work[i + (size_t)i * n] += delta;
            BL(dpotrf_)("L", &n, work, &n, &info);
            if (info == 0) {
                /* 4.42: the decade step over-reported by up to 10 (a point good to 4e-11 read
                 * 3.7e-10 and missed -acc high): three bisections in the exponent, as for the
                 * large blocks, bring the bound within a factor 1.33 */
                double lo = delta / 10, hi = delta;
                for (int b = 0; b < 3; b++) {
                    const double h = sqrt(lo * hi);
                    memcpy(work, M, sizeof(double) * (size_t)n * n);
                    for (int i = 0; i < n; i++) work[i + (size_t)i * n] += h;
                    BL(dpotrf_)("L", &n, work, &n, &info);
                    if (info == 0) hi = h; else lo = h;
                }
                return -hi;
            }
        }
    }
    if (n > 4000) {
        /* 4.42: the ladder continues to 1e-4 max diag before the eigenvalue is computed
         * (the Lanczos estimate that stood here was not a bound, see ps_lammin_tol) */
        double dmax = 0;
        for (int i = 0; i < n; i++) dmax = fmax(dmax, fabs(M[i + (size_t)i * n]));
        for (double rel = 1e-7; rel <= 1.01e-4; rel *= 10) {
            const double delta = rel * (dmax > 0 ? dmax : 1.0);
            if (delta > g_lam_cap) break;
            memcpy(work, M, sizeof(double) * (size_t)n * n);
            for (int i = 0; i < n; i++) work[i + (size_t)i * n] += delta;
            BL(dpotrf_)("L", &n, work, &n, &info);
            if (info == 0) return -delta;
        }
    }
    return lammin_eig(n, M, work);
}

/* the smallest eigenvalue itself (dsyevr, values only); Lanczos only if LAPACK fails */
static double lammin_eig(int n, const double *M, double *work) {
    memcpy(work, M, sizeof(double) * (size_t)n * n);
    int il = 1, iu = 1, mf, lwork = -1, liwork = -1, iwq, isup[2], info;
    double vl = 0, vu = 0, abstol = 0, wq, z = 0;
    double *w = pmalloc(sizeof(double) * (n + 1));      /* dsyevr needs W(N) */
    int ldz = 1;
    BL(dsyevr_)("N", "I", "L", &n, work, &n, &vl, &vu, &il, &iu, &abstol, &mf, w, &z, &ldz, isup,
                &wq, &lwork, &iwq, &liwork, &info);
    lwork = (int)wq + 1; liwork = iwq + 1;
    double *wk = pmalloc(sizeof(double) * lwork);
    int *iwk = pmalloc(sizeof(int) * liwork);
    BL(dsyevr_)("N", "I", "L", &n, work, &n, &vl, &vu, &il, &iu, &abstol, &mf, w, &z, &ldz, isup,
                wk, &lwork, iwk, &liwork, &info);
    free(wk); free(iwk);
    double lam = w[0];
    free(w);
    if (info != 0 || mf < 1) return lanczos_min(n, M);
    return lam;
}

static double lanczos_min(int n, const double *M) {
    const int K = 120;
    double *Q = pmalloc(sizeof(double) * (size_t)n * (K + 1));
    double *wv = pmalloc(sizeof(double) * n);
    double al[121], be[121];
    double nrm = 0;
    for (int i = 0; i < n; i++) { Q[i] = 1.0 + 0.37 * sin(0.9 * i + 0.1); nrm += Q[i] * Q[i]; }
    nrm = 1 / sqrt(nrm);
    for (int i = 0; i < n; i++) Q[i] *= nrm;
    int kk = 0;
    for (int j = 0; j < K; j++) {
        double *qj = Q + (size_t)j * n;
        BL(dsymv_)("L", &n, &DONE_, M, &n, qj, &IONE_, &DZERO_, wv, &IONE_);
        al[j] = BL(ddot_)(&n, qj, &IONE_, wv, &IONE_);
        for (int pass = 0; pass < 2; pass++)
            for (int i = 0; i <= j; i++) {
                double c = -BL(ddot_)(&n, Q + (size_t)i * n, &IONE_, wv, &IONE_);
                BL(daxpy_)(&n, &c, Q + (size_t)i * n, &IONE_, wv, &IONE_);
            }
        be[j] = BL(dnrm2_)(&n, wv, &IONE_);
        kk = j + 1;
        if (be[j] <= 1e-14 * (fabs(al[j]) + 1e-300)) break;
        double inv = 1 / be[j];
        for (int i = 0; i < n; i++) Q[(size_t)(j + 1) * n + i] = wv[i] * inv;
    }
    double d[121], e[121], zv[121 * 121], wk[242];
    int info;
    memcpy(d, al, sizeof(double) * kk);
    for (int i = 0; i + 1 < kk; i++) e[i] = be[i];
    BL(dstev_)("V", &kk, d, e, zv, &kk, wk, &info);
    double lam = d[0] - fabs(be[kk - 1] * zv[kk - 1]);
    free(Q); free(wv);
    return lam;
}

/* lambda_max of the symmetric k x k matrix S (lower), overwritten */
static double lammax_small(int k, double *S) {
    if (k == 1) return S[0];
    int il = k, iu = k, mf, lwork = -1, liwork = -1, iwq, isup[2], info, ldz = 1;
    double vl = 0, vu = 0, abstol = 0, wq, z = 0;
    double *w = pmalloc(sizeof(double) * (k + 1));      /* dsyevr needs W(N) */
    BL(dsyevr_)("N", "I", "L", &k, S, &k, &vl, &vu, &il, &iu, &abstol, &mf, w, &z, &ldz, isup,
                &wq, &lwork, &iwq, &liwork, &info);
    lwork = (int)wq + 1; liwork = iwq + 1;
    double *wk = pmalloc(sizeof(double) * lwork);
    int *iwk = pmalloc(sizeof(int) * liwork);
    BL(dsyevr_)("N", "I", "L", &k, S, &k, &vl, &vu, &il, &iu, &abstol, &mf, w, &z, &ldz, isup,
                wk, &lwork, iwk, &liwork, &info);
    free(wk); free(iwk);
    double lam = w[0];
    free(w);
    return (info == 0 && mf >= 1) ? lam : NAN;
}

/* ------------------------------------------------------------------ dual recovery */
typedef struct {            /* entries of the original data grouped by constraint */
    size_t *ptr;            /* m + 2 offsets: constraint -1 (C) first */
    size_t *ord;
} ConIndex;
/* 23 October 2026: the entries of the constraint a record removed, each with its weight: one constraint of the file
 * (weight 1), or a combination of the file's constraints that the presolve added (lifted_faces, con = -2 - j: the
 * table of the postsolve being undone, g_ps_comb, set by recover_pass) */
static const Postsolve *g_ps_comb = NULL;
#define REC_ENTRIES_BEGIN(CIx, r, q, wq) \
    { const int c0_ = (r)->con, j_ = (r)->con <= -2 ? -2 - (r)->con : -1; \
      const int n_ = c0_ >= 0 ? 1 : (j_ >= 0 && g_ps_comb && j_ < g_ps_comb->ncomb ? g_ps_comb->comb_ptr[j_ + 1] - g_ps_comb->comb_ptr[j_] : 0); \
      for (int u_ = 0; u_ < n_; u_++) { \
        const int ci_ = c0_ >= 0 ? c0_ : g_ps_comb->comb_i[g_ps_comb->comb_ptr[j_] + u_]; \
        const double wq = c0_ >= 0 ? 1.0 : g_ps_comb->comb_w[g_ps_comb->comb_ptr[j_] + u_]; \
        for (size_t e_ = (CIx)->ptr[ci_ + 1]; e_ < (CIx)->ptr[ci_ + 2]; e_++) { const size_t q = (CIx)->ord[e_];
#define REC_ENTRIES_END }}}
static void conindex_build(const PSOrig *O, ConIndex *X) {
    int m = O->m;
    X->ptr = pcalloc(m + 3, sizeof(size_t));
    for (size_t q = 0; q < O->nnz; q++) X->ptr[O->con[q] + 2]++;
    for (int i = 0; i < m + 1; i++) X->ptr[i + 2] += X->ptr[i + 1];
    X->ord = pmalloc(sizeof(size_t) * (O->nnz + 1));
    size_t *fill = pmalloc(sizeof(size_t) * (m + 2));
    memcpy(fill, X->ptr, sizeof(size_t) * (m + 1));        /* group g = con + 1 starts at ptr[g] */
    for (size_t q = 0; q < O->nnz; q++) X->ord[fill[O->con[q] + 1]++] = q;
    free(fill);
}
/* entries of constraint c (-1 = C): X->ord[X->ptr[c+1] .. X->ptr[c+2]) */

/* Zp = T' Z T  (Z: n_orig x n_orig full; T: n_orig x np sparse columns) */
static void congruence(const PSBasis *T, const double *Z, double *Zp, double *ZT) {
    int no = T->n_orig, np = T->n_cur;
    if (T->identity) { memcpy(Zp, Z, sizeof(double) * (size_t)no * no); return; }
    memset(ZT, 0, sizeof(double) * (size_t)no * np);
    for (int a = 0; a < np; a++) {
        double *dst = ZT + (size_t)a * no;
        for (int e = 0; e < T->nz[a]; e++) {
            const double v = T->val[a][e];
            const double *src = Z + (size_t)T->idx[a][e] * no;
            for (int i = 0; i < no; i++) dst[i] += v * src[i];
        }
    }
    for (int c = 0; c < np; c++)
        for (int a = 0; a < np; a++) {
            double s = 0;
            const double *zc = ZT + (size_t)c * no;
            for (int e = 0; e < T->nz[a]; e++) s += T->val[a][e] * zc[T->idx[a][e]];
            Zp[a + (size_t)c * np] = s;
        }
}

/* t for one SDP part; Z = dual slack of the original block (full, n_orig^2) */

/* S += G' A^{-1} G for a sparse SPD A (dense storage, n x n) by a sparse Cholesky with the
 * same shift ladder as chol_reg (4.20: the dense Cholesky of the n x n block in every
 * recovery pass took 0.6 s at n = 4347; the AC-OPF slack is a network-sparse matrix).
 * Returns 0 on success, 1 if not applicable or failed (the caller then goes dense), 2 if
 * no shift made it positive definite.                                                   */
static int sparse_gtag(int n, const double *A, const double *G, int kw, double *S) {
    if (n < 400 || getenv("BRISK_PSDENSE")) return 1;
    size_t nz = 0;
    int *deg = calloc(n + 1, sizeof(int));
    for (int j = 0; j < n; j++)
        for (int i = j + 1; i < n; i++) if (A[i + (size_t)j * n] != 0) { deg[i]++; deg[j]++; nz++; }
    if ((double)nz > 0.03 * 0.5 * (double)n * n) { free(deg); return 1; }
    int **nbr = malloc(sizeof(int *) * (n + 1)), *fill = calloc(n + 1, sizeof(int));
    for (int i = 0; i < n; i++) nbr[i] = malloc(sizeof(int) * (deg[i] + 1));
    for (int j = 0; j < n; j++)
        for (int i = j + 1; i < n; i++) if (A[i + (size_t)j * n] != 0) { nbr[i][fill[i]++] = j; nbr[j][fill[j]++] = i; }
    SChol *F = schol_analyze_adj(n, deg, nbr, (size_t)fmin(3e7 * brisk_mem_scale(), 0.3 * (double)n * n) + 1000);
    for (int i = 0; i < n; i++) free(nbr[i]);
    free(nbr); free(fill); free(deg);
    if (!F) return 1;
    double dmax = 0;
    for (int i = 0; i < n; i++) dmax = fmax(dmax, fabs(A[i + (size_t)i * n]));
    if (dmax == 0) dmax = 1;
    int ok = 0;
    for (int attempt = 0; attempt < 9 && !ok; attempt++) {
        const double sh = attempt == 0 ? 0.0 : dmax * pow(10.0, -15.0 + attempt);
        schol_gather_dense(F, A);
        /* absolute shift: schol_factor's shift is relative to each diagonal, so add it here */
        if (sh > 0) for (int i = 0; i < n; i++) { size_t o = schol_offset(F, i, i); schol_values(F)[o] += sh; }
        if (schol_factor(F, 0.0) == 0) ok = 1;
    }
    if (!ok) { schol_free(F); return 2; }
    double *x = malloc(sizeof(double) * n), *wk = malloc(sizeof(double) * n);
    for (int b = 0; b < kw; b++) {
        memcpy(x, G + (size_t)b * n, sizeof(double) * n);
        schol_solve(F, x, wk);
        for (int a = b; a < kw; a++) {
            const double *ga = G + (size_t)a * n;
            double v = 0;
            for (int i = 0; i < n; i++) v += ga[i] * x[i];
            S[a + (size_t)b * kw] += v;
        }
    }
    free(x); free(wk);
    schol_free(F);
    return 0;
}

/* 4.22: sparse postsolve of a large chordal (moment-form) block. The completion of an
 * n x n block (n = 13678 on case6468_rte, 27318 on case13659) needs several dense copies
 * (X, the completion's workspace, Z, the recovery copy): 6-9 GB on case6468. The measures
 * need X only on the data pattern (A(X), <C,X>, <X,Z> with Z = C - A'y on that pattern)
 * and positive semidefiniteness: the cliques of a chordal pattern being PSD is equivalent
 * to a PSD completion existing (Grone et al.), so lambda_min(X) is measured on the
 * cliques and lambda_min(Z) by a sparse Cholesky of Z on its own pattern. X itself is not
 * returned for such blocks (no -x file, no crossover).                                 */
typedef struct {
    int o, n0, ne;          /* original block, its order, distinct data positions */
    uint64_t *key;          /* sorted keys i * n0 + j (i <= j) of the data positions */
    double *xs, *zs;        /* X and Z at those positions */
    int *qe;                /* data triplet -> position (-1 outside the block) */
    SChol *F;               /* recovery: analysis of the whole data pattern (lazy) */
    int Ffail;
    size_t *off, *doff;     /* factor offsets of the positions (off-diagonal) and diagonals */
} PSSpX;
static int spx_find(const PSSpX *s, uint64_t k) {
    int lo = 0, hi = s->ne - 1;
    while (lo <= hi) { const int mid = (lo + hi) >> 1; if (s->key[mid] < k) lo = mid + 1; else if (s->key[mid] > k) hi = mid - 1; else return mid; }
    return -1;
}
static int cmp_u64(const void *a, const void *b) { const uint64_t x = *(const uint64_t *)a, y = *(const uint64_t *)b; return x < y ? -1 : x > y; }
/* lambda_min bound of the sparse symmetric matrix given by its upper triangle at the
 * positions of s (values v): as ps_lammin_sparse (0, or -d of the ladder). Returns 1 if
 * the pattern could not be factored or no shift up to 1e-6 max|diag| works. */
static int spx_lammin(const PSSpX *s, const double *v, double delta, double *out) {
    const int n = s->n0;
    int *deg = pcalloc(n + 1, sizeof(int));
    for (int e = 0; e < s->ne; e++) { const int i = (int)(s->key[e] / (uint64_t)n), j = (int)(s->key[e] % (uint64_t)n); if (i != j && v[e] != 0) { deg[i]++; deg[j]++; } }
    int **nbr = pmalloc(sizeof(int *) * (n + 1)), *fill = pcalloc(n + 1, sizeof(int));
    for (int i = 0; i < n; i++) nbr[i] = pmalloc(sizeof(int) * (deg[i] + 1));
    for (int e = 0; e < s->ne; e++) { const int i = (int)(s->key[e] / (uint64_t)n), j = (int)(s->key[e] % (uint64_t)n); if (i != j && v[e] != 0) { nbr[i][fill[i]++] = j; nbr[j][fill[j]++] = i; } }
    SChol *S = schol_analyze_adj(n, deg, nbr, (size_t)fmin(2e8 * brisk_mem_scale(), 0.3 * (double)n * n) + 1000);
    for (int i = 0; i < n; i++) free(nbr[i]);
    free(nbr); free(fill); free(deg);
    if (!S) return 1;
    double *dg = pcalloc(n + 1, sizeof(double)), dmax = 0;
    size_t *off = pmalloc(sizeof(size_t) * (s->ne + 1)), *doff = pmalloc(sizeof(size_t) * (n + 1));
    for (int e = 0; e < s->ne; e++) {
        const int i = (int)(s->key[e] / (uint64_t)n), j = (int)(s->key[e] % (uint64_t)n);
        if (i == j) { dg[i] += v[e]; off[e] = (size_t)-1; }
        else off[e] = v[e] != 0 ? schol_offset(S, j, i) : (size_t)-1;   /* row j > col i */
    }
    for (int j = 0; j < n; j++) { doff[j] = schol_offset(S, j, j); dmax = fmax(dmax, fabs(dg[j])); }
    int rc = 1;
    for (int k = -1; k <= 12; k++) {
        const double d = k < 0 ? 0.0 : delta * pow(10.0, k);
        if (k >= 0 && d > 1e-6 * (dmax > 0 ? dmax : 1.0)) break;
        schol_zero(S);
        double *pm = schol_values(S);
        for (int e = 0; e < s->ne; e++) if (off[e] != (size_t)-1) pm[off[e]] += v[e];
        for (int j = 0; j < n; j++) pm[doff[j]] += dg[j] + d;
        if (schol_factor(S, 0.0) == 0) { *out = -d; rc = 0; break; }
        if (k < 0 && !(delta > 0)) break;
    }
    free(off); free(doff); free(dg);
    schol_free(S);
    return rc;
}

/* analysis of the block's whole data pattern, once for all recovery passes */
static int spx_analyze(PSSpX *s) {
    if (s->F) return 0;
    if (s->Ffail) return 1;
    const int n = s->n0;
    int *deg = pcalloc(n + 1, sizeof(int));
    for (int e = 0; e < s->ne; e++) { const int i = (int)(s->key[e] / (uint64_t)n), j = (int)(s->key[e] % (uint64_t)n); if (i != j) { deg[i]++; deg[j]++; } }
    int **nbr = pmalloc(sizeof(int *) * (n + 1)), *fill = pcalloc(n + 1, sizeof(int));
    for (int i = 0; i < n; i++) nbr[i] = pmalloc(sizeof(int) * (deg[i] + 1));
    for (int e = 0; e < s->ne; e++) { const int i = (int)(s->key[e] / (uint64_t)n), j = (int)(s->key[e] % (uint64_t)n); if (i != j) { nbr[i][fill[i]++] = j; nbr[j][fill[j]++] = i; } }
    s->F = schol_analyze_adj(n, deg, nbr, (size_t)fmin(2e8 * brisk_mem_scale(), 0.3 * (double)n * n) + 1000);
    for (int i = 0; i < n; i++) free(nbr[i]);
    free(nbr); free(fill); free(deg);
    if (!s->F) { s->Ffail = 1; return 1; }
    s->off = pmalloc(sizeof(size_t) * (s->ne + 1));
    s->doff = pmalloc(sizeof(size_t) * (n + 1));
    for (int e = 0; e < s->ne; e++) {
        const int i = (int)(s->key[e] / (uint64_t)n), j = (int)(s->key[e] % (uint64_t)n);
        s->off[e] = i != j ? schol_offset(s->F, j, i) : (size_t)-1;
    }
    for (int j = 0; j < n; j++) s->doff[j] = schol_offset(s->F, j, j);
    return 0;
}
/* part_t_sdp for a diagonal-type reduction (deleted indices J, kept K) of the sparse block:
 * Zn = Z[K,K] factored on the whole pattern with the other indices decoupled (identity),
 * G = Z[K,J], H = Z[J,J]; the same t = lambda_max(Aw^-1/2 (G' Zn^-1 G - H) Aw^-1/2).  */
static double part_t_sparse(const PSOrig *O, const ConIndex *CI, const PSRecord *r, const PSPart *pp,
                            PSSpX *s, int *fail) {
    const PSBasis *T = &pp->T;
    const int n = s->n0, np = pp->n_prev, kw = pp->nj;
    if (pp->kind != PS_DIAG || kw <= 0 || spx_analyze(s)) { *fail = 1; return 0; }
    int *po = pmalloc(sizeof(int) * (np + 1));
    for (int c = 0; c < np; c++) {
        po[c] = T->identity ? c : (T->nz[c] == 1 && T->val[c][0] == 1.0 ? T->idx[c][0] : -1);
        if (po[c] < 0) { free(po); *fail = 1; return 0; }
    }
    char *role = pcalloc(n + 1, 1);
    int *jpos = pmalloc(sizeof(int) * (n + 1));
    for (int a = 0; a < pp->nk; a++) role[po[pp->K[a]]] = 1;
    for (int b = 0; b < kw; b++) { role[po[pp->J[b]]] = 2; jpos[po[pp->J[b]]] = b; }
    double *G = pcalloc((size_t)n * kw + 1, sizeof(double)), *H = pcalloc((size_t)kw * kw + 1, sizeof(double));
    double *zd = pcalloc(n + 1, sizeof(double)), dmax = 0;
    for (int e = 0; e < s->ne; e++) {
        const int i = (int)(s->key[e] / (uint64_t)n), j = (int)(s->key[e] % (uint64_t)n);
        const double v = s->zs[e];
        const int ri = role[i], rj = role[j];
        if (i == j) { zd[i] += v; if (ri == 2) H[jpos[i] * (size_t)(kw + 1)] += v; continue; }
        if (ri == 1 && rj == 2) G[i + (size_t)jpos[j] * n] += v;
        else if (ri == 2 && rj == 1) G[j + (size_t)jpos[i] * n] += v;
        else if (ri == 2 && rj == 2) { H[jpos[i] + (size_t)jpos[j] * kw] += v; H[jpos[j] + (size_t)jpos[i] * kw] += v; }
    }
    for (int i = 0; i < n; i++) if (role[i] == 1) dmax = fmax(dmax, fabs(zd[i]));
    if (dmax == 0) dmax = 1;
    int ok = 0;
    for (int attempt = 0; attempt < 9 && !ok; attempt++) {
        const double sh = attempt == 0 ? 0.0 : dmax * pow(10.0, -15.0 + attempt);
        schol_zero(s->F);
        double *pm = schol_values(s->F);
        for (int e = 0; e < s->ne; e++) {
            if (s->off[e] == (size_t)-1) continue;
            const int i = (int)(s->key[e] / (uint64_t)n), j = (int)(s->key[e] % (uint64_t)n);
            if (role[i] == 1 && role[j] == 1) pm[s->off[e]] += s->zs[e];
        }
        for (int j = 0; j < n; j++) pm[s->doff[j]] += role[j] == 1 ? zd[j] + sh : 1.0;
        if (schol_factor(s->F, 0.0) == 0) ok = 1;
    }
    double t = 0;
    double *S = pmalloc(sizeof(double) * ((size_t)kw * kw + 1)), *Aw = pcalloc((size_t)kw * kw + 1, sizeof(double));
    if (!ok) { *fail = 1; goto out; }
    for (size_t q = 0; q < (size_t)kw * kw; q++) S[q] = -H[q];
    {
        double *x = pmalloc(sizeof(double) * (n + 1)), *wk = pmalloc(sizeof(double) * (n + 1));
        for (int b = 0; b < kw; b++) {
            memcpy(x, G + (size_t)b * n, sizeof(double) * n);
            schol_solve(s->F, x, wk);
            for (int a = b; a < kw; a++) {
                const double *ga = G + (size_t)a * n;
                double v = 0;
                for (int i = 0; i < n; i++) v += ga[i] * x[i];
                S[a + (size_t)b * kw] += v;
            }
        }
        free(x); free(wk);
    }
    REC_ENTRIES_BEGIN(CI, r, q, wq)
        if (O->blk[q] != s->o) continue;
        const int i = O->ii[q], j = O->jj[q];
        if (role[i] != 2 || role[j] != 2) continue;
        const double v = r->sgn * wq * O->v[q];
        const int a = jpos[i], b = jpos[j];
        if (i == j) Aw[a + (size_t)a * kw] += v;
        else { const int lo = a < b ? a : b, hi = a < b ? b : a; Aw[hi + (size_t)lo * kw] += v; if (a == b) Aw[a + (size_t)a * kw] += v; }
    REC_ENTRIES_END
    {
        double *wk = pmalloc(sizeof(double) * ((size_t)kw * kw + 1));
        for (int b = 0; b < kw; b++) for (int a = 0; a < b; a++) Aw[a + (size_t)b * kw] = Aw[b + (size_t)a * kw];
        if (chol_reg(kw, Aw, wk)) { *fail = 1; free(wk); goto out; }
        free(wk);
        for (int b = 0; b < kw; b++) for (int a = 0; a < b; a++) S[a + (size_t)b * kw] = S[b + (size_t)a * kw];
        BL(dtrsm_)("L", "L", "N", "N", &kw, &kw, &DONE_, Aw, &kw, S, &kw);
        BL(dtrsm_)("R", "L", "T", "N", &kw, &kw, &DONE_, Aw, &kw, S, &kw);
        for (int b = 0; b < kw; b++) for (int a = b + 1; a < kw; a++) {
            double v = 0.5 * (S[a + (size_t)b * kw] + S[b + (size_t)a * kw]);
            S[a + (size_t)b * kw] = S[b + (size_t)a * kw] = v;
        }
        t = lammax_small(kw, S);
        if (!isfinite(t)) { *fail = 1; t = 0; }
    }
out:
    free(S); free(Aw); free(G); free(H); free(zd); free(po); free(role); free(jpos);
    return t;
}

static double part_t_sdp(const PSOrig *O, const ConIndex *CI, const PSRecord *r, const PSPart *pp,
                         const double *Z, int *fail) {
    const PSBasis *T = &pp->T;
    const int no = T->n_orig, np = pp->n_prev;
    double *Zp = pmalloc(sizeof(double) * ((size_t)np * np + 1));
    double *ZT = T->identity ? NULL : pmalloc(sizeof(double) * ((size_t)no * np + 1));
    congruence(T, Z, Zp, ZT);
    free(ZT);
    /* new coordinates V (nn columns) and complement W (kw columns), in previous coords */
    int nn, kw;
    double *Zn, *G, *H, *TW;     /* nn x nn, nn x kw, kw x kw, no x kw */
    if (pp->kind == PS_DIAG) {
        nn = pp->nk; kw = pp->nj;
        Zn = pmalloc(sizeof(double) * ((size_t)nn * nn + 1));
        G = pmalloc(sizeof(double) * ((size_t)nn * kw + 1));
        H = pmalloc(sizeof(double) * ((size_t)kw * kw + 1));
        for (int b = 0; b < nn; b++) for (int a = 0; a < nn; a++) Zn[a + (size_t)b * nn] = Zp[pp->K[a] + (size_t)pp->K[b] * np];
        for (int b = 0; b < kw; b++) for (int a = 0; a < nn; a++) G[a + (size_t)b * nn] = Zp[pp->K[a] + (size_t)pp->J[b] * np];
        for (int b = 0; b < kw; b++) for (int a = 0; a < kw; a++) H[a + (size_t)b * kw] = Zp[pp->J[a] + (size_t)pp->J[b] * np];
        TW = pmalloc(sizeof(double) * ((size_t)no * kw + 1));
        for (int b = 0; b < kw; b++) basis_col(T, pp->J[b], TW + (size_t)b * no);
    } else if (pp->kind == PS_RANK1) {
        nn = np - 1; kw = 1;
        const int p = pp->p;
        const double *w = pp->w;
        int *map = pmalloc(sizeof(int) * np);
        for (int i = 0, c = 0; i < np; i++) map[i] = (i == p) ? -1 : c++;
        Zn = pmalloc(sizeof(double) * ((size_t)nn * nn + 1));
        G = pmalloc(sizeof(double) * (nn + 1));
        H = pmalloc(sizeof(double));
        const double zpp = Zp[p + (size_t)p * np];
        for (int j = 0; j < np; j++) {
            if (j == p) continue;
            for (int i = 0; i < np; i++) {
                if (i == p) continue;
                Zn[map[i] + (size_t)map[j] * nn] = Zp[i + (size_t)j * np] + w[i] * Zp[p + (size_t)j * np]
                    + w[j] * Zp[i + (size_t)p * np] + w[i] * w[j] * zpp;
            }
            G[map[j]] = Zp[j + (size_t)p * np] + w[j] * zpp;
        }
        H[0] = zpp;
        free(map);
        TW = pmalloc(sizeof(double) * (no + 1));
        basis_col(T, p, TW);
    } else {                                   /* PS_FACE */
        const int s = pp->s, k = pp->kn;
        kw = s - k;
        char *inS = pcalloc(np, 1);
        for (int j = 0; j < s; j++) inS[pp->S[j]] = 1;
        int nr = 0;
        for (int i = 0; i < np; i++) nr += !inS[i];
        nn = nr + k;
        double *V = pcalloc((size_t)np * nn + 1, sizeof(double));
        double *Wd = pcalloc((size_t)np * kw + 1, sizeof(double));
        for (int i = 0, c = 0; i < np; i++) if (!inS[i]) V[i + (size_t)(c++) * np] = 1.0;
        for (int a = 0; a < k; a++) for (int j = 0; j < s; j++) V[pp->S[j] + (size_t)(nr + a) * np] = pp->N[j + (size_t)a * s];
        for (int a = 0; a < kw; a++) for (int j = 0; j < s; j++) Wd[pp->S[j] + (size_t)a * np] = pp->U[j + (size_t)a * s];
        double *ZV = pmalloc(sizeof(double) * ((size_t)np * (nn > kw ? nn : kw) + 1));
        Zn = pmalloc(sizeof(double) * ((size_t)nn * nn + 1));
        G = pmalloc(sizeof(double) * ((size_t)nn * kw + 1));
        H = pmalloc(sizeof(double) * ((size_t)kw * kw + 1));
        if (nn > 0) {
            BL(dgemm_)("N", "N", &np, &nn, &np, &DONE_, Zp, &np, V, &np, &DZERO_, ZV, &np);
            BL(dgemm_)("T", "N", &nn, &nn, &np, &DONE_, V, &np, ZV, &np, &DZERO_, Zn, &nn);
        }
        BL(dgemm_)("N", "N", &np, &kw, &np, &DONE_, Zp, &np, Wd, &np, &DZERO_, ZV, &np);
        if (nn > 0) BL(dgemm_)("T", "N", &nn, &kw, &np, &DONE_, V, &np, ZV, &np, &DZERO_, G, &nn);
        BL(dgemm_)("T", "N", &kw, &kw, &np, &DONE_, Wd, &np, ZV, &np, &DZERO_, H, &kw);
        TW = pcalloc((size_t)no * kw + 1, sizeof(double));
        double *tc = pmalloc(sizeof(double) * (no + 1));
        for (int i = 0; i < np; i++) {
            basis_col(T, i, tc);
            for (int a = 0; a < kw; a++) {
                double wia = Wd[i + (size_t)a * np];
                if (wia != 0) for (int q = 0; q < no; q++) TW[q + (size_t)a * no] += wia * tc[q];
            }
        }
        free(tc); free(V); free(Wd); free(ZV); free(inS);
    }
    free(Zp);
    /* Aw = sgn (TW)' A_r (TW) over the entries of A_r in this block */
    double *Aw = pcalloc((size_t)kw * kw + 1, sizeof(double));
    REC_ENTRIES_BEGIN(CI, r, q, wq)
        if (O->blk[q] != pp->blk) continue;
        const int i = O->ii[q], j = O->jj[q];
        const double v = r->sgn * wq * O->v[q];
        for (int b = 0; b < kw; b++) {
            const double tib = TW[i + (size_t)b * no], tjb = TW[j + (size_t)b * no];
            for (int a = b; a < kw; a++) {
                const double tia = TW[i + (size_t)a * no], tja = TW[j + (size_t)a * no];
                Aw[a + (size_t)b * kw] += (i == j) ? v * tia * tib : v * (tia * tjb + tja * tib);
            }
        }
    REC_ENTRIES_END
    free(TW);
    double t = 0;
    /* S = G' Zn^{-1} G - H */
    double *S = pmalloc(sizeof(double) * ((size_t)kw * kw + 1));
    for (size_t q = 0; q < (size_t)kw * kw; q++) S[q] = -H[q];
    if (nn > 0) {
        double *wk = pmalloc(sizeof(double) * ((size_t)nn * nn + 1));
        if (getenv("BRISK_PSDEBUG")) {
            double lz = ps_lammin(nn, Zn, wk), dm = 0;
            for (int i = 0; i < nn; i++) dm = fmax(dm, fabs(Zn[i + (size_t)i * nn]));
            printf("   [ps part blk %d kind %d: lambda_min(Zn) %.3e (max diag %.2e), H %.3e]\n", pp->blk, pp->kind, lz, dm, H[0]);
            if (nn <= 40 && kw == 1 && getenv("BRISK_PSDEBUG2")) {   /* the spectrum of Zn and the components of g on its eigenvectors */
                double *Zc = pmalloc(sizeof(double) * ((size_t)nn * nn + 1)), *ev = pmalloc(sizeof(double) * (nn + 1)), *w2 = pmalloc(sizeof(double) * (8 * nn + 64));
                memcpy(Zc, Zn, sizeof(double) * (size_t)nn * nn);
                int lw = 8 * nn + 64, info = 0;
                BL(dsyev_)("V", "L", &nn, Zc, &nn, ev, w2, &lw, &info);
                printf("   [   eig(Zn) / |g'v|:");
                for (int i = 0; i < nn; i++) { double gv = 0; for (int q = 0; q < nn; q++) gv += G[q] * Zc[q + (size_t)i * nn]; printf(" %.1e/%.1e", ev[i], fabs(gv)); }
                printf("]\n");

                free(Zc); free(ev); free(w2);
            }
        }
        const int sp = sparse_gtag(nn, Zn, G, kw, S);
        if (sp == 2) { *fail = 1; free(wk); goto out; }
        if (sp == 1) {
            if (chol_reg(nn, Zn, wk)) { *fail = 1; free(wk); goto out; }
            BL(dtrsm_)("L", "L", "N", "N", &nn, &kw, &DONE_, Zn, &nn, G, &nn);          /* Y = L^{-1} G */
            BL(dsyrk_)("L", "T", &kw, &nn, &DONE_, G, &nn, &DONE_, S, &kw);            /* S += Y'Y */
        }
        free(wk);
    }
    {
        double *wk = pmalloc(sizeof(double) * ((size_t)kw * kw + 1));
        for (int b = 0; b < kw; b++) for (int a = 0; a < b; a++) Aw[a + (size_t)b * kw] = Aw[b + (size_t)a * kw];
        if (chol_reg(kw, Aw, wk)) { *fail = 1; free(wk); goto out; }
        free(wk);
        for (int b = 0; b < kw; b++) for (int a = 0; a < b; a++) S[a + (size_t)b * kw] = S[b + (size_t)a * kw];
        BL(dtrsm_)("L", "L", "N", "N", &kw, &kw, &DONE_, Aw, &kw, S, &kw);
        BL(dtrsm_)("R", "L", "T", "N", &kw, &kw, &DONE_, Aw, &kw, S, &kw);
        for (int b = 0; b < kw; b++) for (int a = b + 1; a < kw; a++) {
            double v = 0.5 * (S[a + (size_t)b * kw] + S[b + (size_t)a * kw]);
            S[a + (size_t)b * kw] = S[b + (size_t)a * kw] = v;
        }
        t = lammax_small(kw, S);
        if (!isfinite(t)) { *fail = 1; t = 0; }
        if (getenv("BRISK_PSDEBUG")) { double gn = 0; for (size_t q = 0; q < (size_t)nn * kw; q++) gn = fmax(gn, fabs(G[q])); printf("   [ps part blk %d: kw %d H00 %.3e |Y|max %.3e Aw00(chol) %.3e S00(scaled) %.3e t %.3e]\n", pp->blk, kw, H[0], gn, Aw[0], S[0], t); }
    }
out:
    free(S); free(Aw); free(Zn); free(G); free(H);
    return t;
}

/* one pass of dual recovery through the reduction log, newest reduction first; each removed
 * constraint gets y_r = -sgn (1 + mf) t* (+ tiny), t* the least value that makes the slack of
 * the space before that reduction PSD */
static int recover_pass(const Postsolve *ps, const PSOrig *O, const ConIndex *CIp, double *yo, double **Zo,
                        double mf, int *rec, int *unrec, PSSpX *sp) {
    const ConIndex CI = *CIp;
    g_ps_comb = ps;
    for (int ri = ps->nrec - 1; ri >= 0; ri--) {
        const PSRecord *r = &ps->rec[ri];
        if (r->np == 0) continue;
        double t = 0, scale = 0;
        int fail = 0;
        for (int q = 0; q < r->np; q++) {
            const PSPart *pp = &r->parts[q];
            const int o = pp->blk, no = abs(O->bs[o]);
            if (sp && o == sp->o) {
                double tq = part_t_sparse(O, CIp, r, pp, sp, &fail);
                if (fail) break;
                t = fmax(t, tq);
                double dmax = 0;
                for (int e = 0; e < sp->ne; e++) if (sp->key[e] / (uint64_t)sp->n0 == sp->key[e] % (uint64_t)sp->n0) dmax = fmax(dmax, fabs(sp->zs[e]));
                scale = fmax(scale, dmax);
                continue;
            }
            if (pp->kind == PS_LP) {
                for (int b = 0; b < pp->nj; b++) {
                    /* original variable of previous coordinate J[b] */
                    const PSBasis *T = &pp->T;
                    int var = T->identity ? pp->J[b] : (T->nz[pp->J[b]] ? T->idx[pp->J[b]][0] : -1);
                    if (var < 0) continue;
                    double a = 0;
                    REC_ENTRIES_BEGIN(&CI, r, qq, wq)
                        if (O->blk[qq] == o && O->ii[qq] == var) a += r->sgn * wq * O->v[qq];
                    REC_ENTRIES_END
                    if (a > 0) t = fmax(t, -Zo[o][var] / a);
                    scale = fmax(scale, fabs(Zo[o][var]) / (a > 0 ? a : 1));
                }
            } else {
                double tq = part_t_sdp(O, CIp, r, pp, Zo[o], &fail);
                if (fail) break;
                t = fmax(t, tq);
                double dmax = 0;
                for (int i = 0; i < no; i++) dmax = fmax(dmax, fabs(Zo[o][i + (size_t)i * no]));
                scale = fmax(scale, dmax);
            }
        }
        if (getenv("BRISK_PSDEBUG")) {
            printf("   [ps rec %d con %d np %d kind %d nj %d blk %d n_prev %d: t* %.3e scale %.2e%s]\n",
                   ri, r->con, r->np, r->parts[0].kind, r->parts[0].nj, r->parts[0].blk,
                   r->parts[0].n_prev, t, scale, fail ? " FAIL" : "");
        }
        if (fail) { (*unrec)++; continue; }
        /* back off from t* so that the next (outer) reduction sees a strictly feasible interior
         * point of the previous-space slack rather than a boundary point */
        double tu = (1.0 + mf) * fmax(t, 0.0) + 1e-9 * (1.0 + scale);
        /* a diverging chain (the dual optimum is not attained: singularity degree >= 2)
         * cannot be repaired this way; give up early instead of producing 1e300s */
        if (!(tu < 1e12 * (1.0 + O->nC1))) return 1;
        double dy = -r->sgn * tu;
        if (r->con >= 0) yo[r->con] += dy;
        else if (r->con <= -2 && -2 - r->con < ps->ncomb)   /* a combination the presolve added: its weights on the file's rows */
            for (int u = ps->comb_ptr[-2 - r->con]; u < ps->comb_ptr[-1 - r->con]; u++) yo[ps->comb_i[u]] += dy * ps->comb_w[u];
        REC_ENTRIES_BEGIN(&CI, r, q, wq)
            const int o = O->blk[q], i = O->ii[q], j = O->jj[q], no = abs(O->bs[o]);
            const double c = -dy * wq * O->v[q];
            if (sp && o == sp->o) { sp->zs[sp->qe[q]] += c; continue; }
            if (O->bs[o] < 0) Zo[o][i] += c;
            else { Zo[o][i + (size_t)j * no] += c; if (i != j) Zo[o][j + (size_t)i * no] += c; }
        REC_ENTRIES_END
        (*rec)++;
        if (!sp && getenv("BRISK_PSDEBUG") && r->np == 1 && r->parts[0].kind != PS_LP) {
            const PSPart *pp = &r->parts[0];
            const int o = pp->blk, no = abs(O->bs[o]), np = pp->n_prev;
            double *Zp = pmalloc(sizeof(double) * (size_t)np * np), *ZT = pmalloc(sizeof(double) * ((size_t)no * np + 1));
            double *wk = pmalloc(sizeof(double) * (size_t)np * np);
            congruence(&pp->T, Zo[o], Zp, ZT);
            printf("   [ps after rec %d%s: lambda_min(prev-space Z) %.3e, t used %.3e]\n", ri, r->coupled ? " (coupled)" : "", ps_lammin(np, Zp, wk), tu);
            free(Zp); free(ZT); free(wk);
        }
    }
    return 0;
}

/* Rounding uncertainty of Z = C - A'y. With huge recovered duals the slack is a
 * difference of large numbers: its entries are only determined to within
 * |E_ij| <= eps * (sum of the |contributions| to entry ij), and since ||E||_2 is at most
 * the largest row sum of |E|, lambda_min(Z) is only known to that accuracy. Measured on
 * taha1a: two evaluations of the same y (this program's order and a sparse A'y product)
 * disagreed by 1.5e-5 in err4 while each looked feasible to itself. Such a y is not a
 * usable dual, so the bound counts as a violation. (It is pessimistic when the large
 * terms happen to cancel exactly, but that cannot be relied on.)                        */
static double z_round_est(const PSOrig *O, const double *yo) {
    size_t tot = 0;
    size_t *off = pmalloc(sizeof(size_t) * (O->nblk + 1));
    for (int o = 0; o < O->nblk; o++) { off[o] = tot; tot += (size_t)abs(O->bs[o]); }
    double *rs = pcalloc(tot + 1, sizeof(double));
    for (size_t q = 0; q < O->nnz; q++) {
        const int o = O->blk[q];
        const double c = fabs(O->con[q] < 0 ? O->v[q] : yo[O->con[q]] * O->v[q]);
        rs[off[o] + O->ii[q]] += c;
        if (O->bs[o] > 0 && O->ii[q] != O->jj[q]) rs[off[o] + O->jj[q]] += c;
    }
    double mx = 0;
    for (size_t i = 0; i < tot; i++) mx = fmax(mx, rs[i]);
    free(rs); free(off);
    return 2.2e-16 * mx;
}

/* Minimal-norm correction of X onto A(X) = b (moment form of the chordal conversion):
 * there X comes from the multipliers of the moment rows, and A(X) = b holds only as well as
 * the dual feasibility of the free columns; with multipliers y of size 1e3-1e4 (AC-OPF) a
 * residual of 2e-8 made the gap measure <X, Z> 4.5e-7. dX = A'(A A')^{-1} r by
 * Jacobi-preconditioned CG on the sparse A A'; kept if it cuts the residual and its norm
 * (the most it can move lambda_min(X)) is small against what it removes from <X, Z>. */
typedef struct { uint64_t key; int q; } PQKey;
static int cmp_pqkey(const void *a, const void *b) {
    const PQKey *x = a, *y = b;
    return x->key < y->key ? -1 : x->key > y->key ? 1 : 0;
}
static void ps_project_x(const PSOrig *O, double **Xo, const double *yo, int verbose, const PSSpX *sp,
                         const PSChordal *spch, const int *spsel, double **spXc, int spncl,
                         const double *clam_o, const PSChordal *const *cch_o, int *const *csel_o) {
    const int mo = O->m;
    size_t na = 0;
    for (size_t q = 0; q < O->nnz; q++) if (O->con[q] >= 0) na++;
    if (!na) return;
    PQKey *K = pmalloc(sizeof(PQKey) * na);
    size_t w = 0;
    for (size_t q = 0; q < O->nnz; q++) if (O->con[q] >= 0) {
        K[w].key = ((uint64_t)O->blk[q] << 42) | ((uint64_t)O->ii[q] << 21) | (uint64_t)O->jj[q];
        K[w].q = (int)q; w++;
    }
    qsort(K, na, sizeof(PQKey), cmp_pqkey);
    int *eid = pmalloc(sizeof(int) * na), *qs = pmalloc(sizeof(int) * na), ne = 0;
    for (size_t t = 0; t < na; t++) { if (t == 0 || K[t].key != K[t - 1].key) ne++; eid[t] = ne - 1; qs[t] = K[t].q; }
    int *eq = pmalloc(sizeof(int) * (ne + 1));           /* a representative triplet per entry */
    for (size_t t = 0; t < na; t++) eq[eid[t]] = qs[t];
    uint64_t *ekey = pmalloc(sizeof(uint64_t) * (ne + 1));   /* the entries' keys, increasing (the measure below) */
    for (size_t t = 0; t < na; t++) ekey[eid[t]] = K[t].key;
    free(K);
    /* weights: 2 for off-diagonal SDP entries */
    double *cw = pmalloc(sizeof(double) * (ne + 1));
    for (int e = 0; e < ne; e++) { const int q = eq[e]; cw[e] = (O->bs[O->blk[q]] > 0 && O->ii[q] != O->jj[q]) ? 2.0 : 1.0; }
    double *tw = pmalloc(sizeof(double) * (ne + 1));
    /* 4.20: entries shared by many rows (the constant moment X_00 of a Shor relaxation is in
     * every constraint with a constant term) would make K = A W A' dense; they are kept
     * fixed (weight 0) and the others absorb the correction */
    {
        int *cnt = pcalloc(ne + 1, sizeof(int));
        for (size_t t = 0; t < na; t++) cnt[eid[t]]++;
        for (int e = 0; e < ne; e++) { tw[e] = cw[e]; if (cnt[e] > 64) cw[e] = 0.0; }
        free(cnt);
    }
    double *xe = pmalloc(sizeof(double) * (ne + 1));
    for (int e = 0; e < ne; e++) {
        const int q = eq[e], o = O->blk[q], no = abs(O->bs[o]);
        if (sp && o == sp->o) { xe[e] = sp->xs[sp->qe[q]]; continue; }
        xe[e] = O->bs[o] < 0 ? Xo[o][O->ii[q]] : Xo[o][O->ii[q] + (size_t)O->jj[q] * no];
    }
    double *r = pcalloc(mo + 1, sizeof(double)), *u = pcalloc(mo + 1, sizeof(double)), *p = pmalloc(sizeof(double) * (mo + 1));
    double *z = pmalloc(sizeof(double) * (mo + 1)), *Kp = pmalloc(sizeof(double) * (mo + 1)), *dg = pcalloc(mo + 1, sizeof(double));
    double *de = pmalloc(sizeof(double) * (ne + 1));
    for (int i = 0; i < mo; i++) r[i] = O->b[i];
    for (size_t t = 0; t < na; t++) { const int q = qs[t]; r[O->con[q]] -= tw[eid[t]] * O->v[q] * xe[eid[t]]; dg[O->con[q]] += cw[eid[t]] * O->v[q] * O->v[q]; }
    double r0 = 0, yr = 0;
    for (int i = 0; i < mo; i++) { r0 += r[i] * r[i]; yr += yo[i] * r[i]; }
    r0 = sqrt(r0);
    /* K v = A(A'v) */
#define KMV(vin, vout) do { \
        for (int e = 0; e < ne; e++) de[e] = 0; \
        for (size_t t = 0; t < na; t++) de[eid[t]] += (vin)[O->con[qs[t]]] * O->v[qs[t]]; \
        for (int i = 0; i < mo; i++) (vout)[i] = 0; \
        for (size_t t = 0; t < na; t++) (vout)[O->con[qs[t]]] += cw[eid[t]] * O->v[qs[t]] * de[eid[t]]; } while (0)
    double *res = pmalloc(sizeof(double) * (mo + 1));
    int it = 0;
    double rn = r0;
    /* 4.20: direct solve of K = A W A' by a sparse Cholesky (Jacobi-PCG stalled: case588
     * 1e-8 -> 2.4e-7 after 500 steps), refined against the exact K */
    int direct = 0;
    if (!getenv("BRISK_PROJCG") && mo >= 2) {
        /* rows of each entry (triplets are grouped by entry: eid is non-decreasing in t) */
        int *deg = pcalloc(mo + 1, sizeof(int)), **nbr = pmalloc(sizeof(int *) * (mo + 1)), *cap = pcalloc(mo + 1, sizeof(int));
        for (size_t t0 = 0; t0 < na; ) {
            size_t t1 = t0; while (t1 < na && eid[t1] == eid[t0]) t1++;
            if (cw[eid[t0]] == 0) { t0 = t1; continue; }
            for (size_t a = t0; a < t1; a++) for (size_t b = t0; b < t1; b++) {
                const int i = O->con[qs[a]], j = O->con[qs[b]];
                if (i == j) continue;
                if (deg[i] == cap[i]) { cap[i] = cap[i] ? 2 * cap[i] : 8; nbr[i] = realloc(deg[i] ? nbr[i] : NULL, sizeof(int) * cap[i]); }
                nbr[i][deg[i]++] = j;
            }
            t0 = t1;
        }
        /* deduplicate */
        int *mk = pmalloc(sizeof(int) * (mo + 1));
        for (int i = 0; i < mo; i++) mk[i] = -1;
        for (int i = 0; i < mo; i++) {
            int w2 = 0;
            for (int q = 0; q < deg[i]; q++) { const int j = nbr[i][q]; if (mk[j] != i) { mk[j] = i; nbr[i][w2++] = j; } }
            deg[i] = w2;
            if (!cap[i]) nbr[i] = NULL;
        }
        int **nb2 = pmalloc(sizeof(int *) * (mo + 1)); int dummy = 0;
        for (int i = 0; i < mo; i++) nb2[i] = nbr[i] ? nbr[i] : &dummy;
        SChol *F = schol_analyze_adj(mo, deg, nb2, (size_t)fmin(4e7, 0.45 * (double)mo * mo) + 1000);
        free(nb2);
        if (getenv("BRISK_PSTIME")) { long sd = 0; int md = 0; for (int i = 0; i < mo; i++) { sd += deg[i]; if (deg[i] > md) md = deg[i]; } printf("   [projection: K analysis %s, m %d, pattern %ld, max degree %d]\n", F ? "ok" : "failed", mo, sd, md); }
        if (F) {
            /* values of K */
            double *pm = schol_values(F);
            memset(pm, 0, 0);
            schol_zero(F);
            for (size_t t0 = 0; t0 < na; ) {
                size_t t1 = t0; while (t1 < na && eid[t1] == eid[t0]) t1++;
                const double we = cw[eid[t0]];
                if (we == 0) { t0 = t1; continue; }
                for (size_t a = t0; a < t1; a++) for (size_t b = t0; b < t1; b++) {
                    const int i = O->con[qs[a]], j = O->con[qs[b]];
                    if (i < j) continue;
                    schol_add(F, (size_t)i, (size_t)j, we * O->v[qs[a]] * O->v[qs[b]]);
                }
                t0 = t1;
            }
            double kmax = 0;
            for (int i = 0; i < mo; i++) kmax = fmax(kmax, dg[i]);
            int okf = 0;
            for (int at = 0; at < 6 && !okf; at++) {
                const double sh = at == 0 ? 0.0 : kmax * pow(10.0, -16.0 + 2 * at);
                if (schol_factor(F, sh > 0 ? sh / fmax(kmax, 1e-300) : 0.0) == 0) okf = 1;
                if (getenv("BRISK_PSTIME")) printf("   [projection: factor shift %.0e %s]\n", sh / fmax(kmax, 1e-300), okf ? "ok" : "failed");
            }
            if (okf) {
                double *wk = pmalloc(sizeof(double) * (mo + 1)), *du = pmalloc(sizeof(double) * (mo + 1));
                memcpy(res, r, sizeof(double) * mo);
                for (int pass = 0; pass < 4; pass++) {
                    memcpy(du, res, sizeof(double) * mo);
                    schol_solve(F, du, wk);
                    for (int i = 0; i < mo; i++) u[i] += du[i];
                    KMV(u, Kp);
                    double rn2 = 0;
                    for (int i = 0; i < mo; i++) { res[i] = r[i] - Kp[i]; rn2 += res[i] * res[i]; }
                    rn2 = sqrt(rn2);
                    it++;
                    if (!(rn2 < 0.5 * rn)) { if (rn2 > rn) for (int i = 0; i < mo; i++) u[i] -= du[i]; else rn = rn2; break; }
                    rn = rn2;
                }
                free(wk); free(du);
                direct = 1;
            }
            schol_free(F);
        }
        for (int i = 0; i < mo; i++) free(nbr[i]);
        free(nbr); free(deg); free(cap); free(mk);
    }
    memcpy(res, r, sizeof(double) * mo);
    if (direct) goto projected;
    for (int i = 0; i < mo; i++) { z[i] = dg[i] > 0 ? res[i] / dg[i] : 0; p[i] = z[i]; }
    double rz = 0; for (int i = 0; i < mo; i++) rz += res[i] * z[i];
    for (; it < 500 && rn > 1e-4 * r0 && r0 > 0; it++) {
        KMV(p, Kp);
        double pKp = 0; for (int i = 0; i < mo; i++) pKp += p[i] * Kp[i];
        if (!(pKp > 0)) break;
        const double a = rz / pKp;
        rn = 0;
        for (int i = 0; i < mo; i++) { u[i] += a * p[i]; res[i] -= a * Kp[i]; rn += res[i] * res[i]; }
        rn = sqrt(rn);
        double rz2 = 0; for (int i = 0; i < mo; i++) { z[i] = dg[i] > 0 ? res[i] / dg[i] : 0; rz2 += res[i] * z[i]; }
        const double bt = rz2 / rz; rz = rz2;
        for (int i = 0; i < mo; i++) p[i] = z[i] + bt * p[i];
    }
projected:
    /* dX = A'u and its (symmetric Frobenius) norm */
    for (int e = 0; e < ne; e++) de[e] = 0;
    for (size_t t = 0; t < na; t++) de[eid[t]] += u[O->con[qs[t]]] * O->v[qs[t]];
    for (int e = 0; e < ne; e++) if (cw[e] == 0) de[e] = 0;
    double dn = 0;
    for (int e = 0; e < ne; e++) dn += cw[e] * de[e] * de[e];
    dn = sqrt(dn);
    /* accept: the residual drops tenfold, and the norm (a bound on the change of
     * lambda_min(X)) is well below the part of the gap measure it removes */
    /* 24 October 2026 (ISSUES 273): the correction is kept when it lowers the largest of the errors it changes, measured
     * as the final measure measures them (DIMACS errors 1, 2, 5 and 6; y and Z = C - A'y stay): the residual, lambda_min
     * of every block and clique it touches (ps_lammin_tol, the sparse block on its cliques), the gap and the
     * complementarity, <X, C - A'y> = <C,X> - b'y + y'r with r = b - A(X). The test before (kept as the fallback when
     * nothing can be measured, BRISK_PROJBOUND=1 always) compared the norm of the correction, a bound on the change of
     * lambda_min that the spread of the correction over many blocks makes loose, with the complementarity it removes in
     * the units of ||b||_1 instead of those of the objective: on the AC power flows in the moment form it rejected 18 of
     * 24 corrections, among them acopf_case793_goc's (SDP with SOCP; the residual 4.0e-8 -> 1.7e-9, the norm 2.1e-3,
     * y'r 2.5e-5), which leaves the complementarity at 1.0e-6 on the problem as read. */
    const double gain = fabs(yr) / (1.0 + fabs(O->nb1));
    int ok = rn <= 0.1 * r0 && dn < 0.3 * fmax(gain, r0) && r0 <= 1e-4 * (1.0 + fabs(O->nb1));   /* a polish, not a repair */
    double m0 = -1, m1 = -1;
    if (!ok && rn < r0 && r0 > 0 && r0 <= 1e-4 * (1.0 + fabs(O->nb1)) && !getenv("BRISK_PROJBOUND")) {
        const double dtol = 1e-12 * (1.0 + O->nb1);
        double pobj = 0, dpo = 0, dobj = 0, yr1 = 0;
        for (size_t q = 0; q < O->nnz; q++) {
            if (O->con[q] >= 0) continue;
            const int o = O->blk[q], no = abs(O->bs[o]), i = O->ii[q], j = O->jj[q];
            const double w2 = (O->bs[o] > 0 && i != j) ? 2.0 : 1.0;
            const double x = sp && o == sp->o ? (sp->qe[q] >= 0 ? sp->xs[sp->qe[q]] : 0.0) : O->bs[o] < 0 ? Xo[o][i] : Xo[o][i + (size_t)j * no];
            pobj += w2 * O->v[q] * x;
            const uint64_t key = ((uint64_t)o << 42) | ((uint64_t)i << 21) | (uint64_t)j;
            int lo = 0, hi = ne - 1, e = -1;
            while (lo <= hi) { const int mid = (lo + hi) >> 1; if (ekey[mid] < key) lo = mid + 1; else if (ekey[mid] > key) hi = mid - 1; else { e = mid; break; } }
            if (e >= 0) dpo += w2 * O->v[q] * de[e];
        }
        /* the residual after the correction, b - A(X + dX), from dX itself (res is not it on every path: the direct
         * passes leave the last pass's, and the copy before the jump past the CG overwrites it) */
        double *r1 = pmalloc(sizeof(double) * (mo + 1)), rn1 = 0;
        memcpy(r1, r, sizeof(double) * mo);
        for (size_t t = 0; t < na; t++) r1[O->con[qs[t]]] -= cw[eid[t]] * O->v[qs[t]] * de[eid[t]];
        for (int i = 0; i < mo; i++) { dobj += O->b[i] * yo[i]; yr1 += yo[i] * r1[i]; rn1 += r1[i] * r1[i]; }
        rn1 = sqrt(rn1); free(r1);
        const double po1 = pobj + dpo;
        const double den0 = 1 + fabs(pobj + O->obj_off) + fabs(dobj + O->obj_off), den1 = 1 + fabs(po1 + O->obj_off) + fabs(dobj + O->obj_off);
        /* lambda_min of what the correction touches, before and after */
        double lam0 = INFINITY, lam1 = INFINITY;
        char *tb = pcalloc((size_t)O->nblk + 1, 1);
        for (int e = 0; e < ne; e++) if (de[e] != 0) {
            const int q = eq[e], o = O->blk[q];
            if (O->bs[o] < 0) { const double x = Xo[o][O->ii[q]]; lam0 = fmin(lam0, x); lam1 = fmin(lam1, x + de[e]); }
            else tb[o] = 1;
        }
        for (int o = 0; o < O->nblk && m0 != -2; o++) {
            if (!tb[o] || (sp && o == sp->o)) continue;
            const int no = O->bs[o];
            /* 24 October 2026 (ISSUES 275): a block the final measure takes on its cliques (the sparse measure is on and
             * the block was decomposed: X is not completed outside its cliques) is measured on its cliques here too; its
             * whole matrix gave lambda_min -14 on pglib_opf_case588_sdet (SDP), the same before and after, which decided
             * the test. Without a clique basis (not a selection) the final measure keeps the value of the reduced
             * problem, and the change cannot be measured: the bound test decides. */
            const int oncl = sp && clam_o && clam_o[o] == clam_o[o];
            if (oncl && !(cch_o && cch_o[o])) { m0 = -2; break; }
            double *M = pmalloc(sizeof(double) * ((size_t)no * no + 1));
            memcpy(M, Xo[o], sizeof(double) * (size_t)no * no);
            for (int e = 0; e < ne; e++) {
                const int q = eq[e];
                if (O->blk[q] != o || de[e] == 0) continue;
                const int i = O->ii[q], j = O->jj[q];
                M[i + (size_t)j * no] += de[e]; if (i != j) M[j + (size_t)i * no] += de[e];
            }
            if (oncl) {
                const PSChordal *ch = cch_o[o]; const int *sel = csel_o[o];
                int cm = 1; for (int t = 0; t < ch->ncl; t++) { const int c = ch->cptr[t + 1] - ch->cptr[t]; if (c > cm) cm = c; }
                double *G0 = pmalloc(sizeof(double) * ((size_t)cm * cm + 1)), *G1 = pmalloc(sizeof(double) * ((size_t)cm * cm + 1)), *wk = pmalloc(sizeof(double) * ((size_t)cm * cm + 1));
                for (int t = 0; t < ch->ncl; t++) {
                    const int c = ch->cptr[t + 1] - ch->cptr[t]; const int *cv = ch->cv + ch->cptr[t];
                    int touched = 0;
                    for (int b = 0; b < c; b++)
                        for (int a = 0; a < c; a++) {
                            const size_t ix = (size_t)sel[cv[a]] + (size_t)sel[cv[b]] * no;
                            G0[a + (size_t)b * c] = Xo[o][ix]; G1[a + (size_t)b * c] = M[ix];
                            if (M[ix] != Xo[o][ix]) touched = 1;
                        }
                    if (touched) { lam0 = fmin(lam0, ps_lammin_tol(c, G0, wk, dtol)); lam1 = fmin(lam1, ps_lammin_tol(c, G1, wk, dtol)); }
                }
                free(G0); free(G1); free(wk);
            } else {
                double *wk = pmalloc(sizeof(double) * ((size_t)no * no + 1));
                lam0 = fmin(lam0, ps_lammin_tol(no, Xo[o], wk, dtol));
                lam1 = fmin(lam1, ps_lammin_tol(no, M, wk, dtol));
                free(wk);
            }
            free(M);
        }
        if (sp && sp->o >= 0 && tb[sp->o]) {
            if (!spch || !spsel || !spXc) m0 = -2;            /* the sparse block without its cliques: not measured */
            else {
                double *dsp = pcalloc((size_t)sp->ne + 1, sizeof(double));
                for (int e = 0; e < ne; e++) { const int q = eq[e]; if (O->blk[q] == sp->o && sp->qe[q] >= 0) dsp[sp->qe[q]] += de[e]; }
                const uint64_t n0 = (uint64_t)sp->n0;
                for (int t = 0; t < spncl; t++) {
                    const int c = spch->cptr[t + 1] - spch->cptr[t];
                    const int *cv = spch->cv + spch->cptr[t];
                    double *G = pmalloc(sizeof(double) * ((size_t)c * c + 1)), *wk = pmalloc(sizeof(double) * ((size_t)c * c + 1));
                    memcpy(G, spXc[t], sizeof(double) * (size_t)c * c);
                    int touched = 0;
                    for (int b = 0; b < c; b++)
                        for (int a = 0; a < c; a++) {
                            int i = spsel[cv[a]], j = spsel[cv[b]];
                            if (i > j) { const int tt = i; i = j; j = tt; }
                            const int p = spx_find(sp, (uint64_t)i * n0 + (uint64_t)j);
                            if (p >= 0 && dsp[p] != 0) { G[a + (size_t)b * c] = sp->xs[p] + dsp[p]; touched = 1; }
                        }
                    if (touched) { lam0 = fmin(lam0, ps_lammin_tol(c, spXc[t], wk, dtol)); lam1 = fmin(lam1, ps_lammin_tol(c, G, wk, dtol)); }
                    free(G); free(wk);
                }
                free(dsp);
            }
        }
        free(tb);
        if (m0 != -2) {
            const double nb = 1.0 + O->nb1;
            m0 = fmax(fmax(r0 / nb, lam0 < 0 ? -lam0 / nb : 0.0), fmax(fabs(pobj - dobj) / den0, fabs(pobj - dobj + yr) / den0));
            m1 = fmax(fmax(rn1 / nb, lam1 < 0 ? -lam1 / nb : 0.0), fmax(fabs(po1 - dobj) / den1, fabs(po1 - dobj + yr1) / den1));
            ok = m1 < m0;
        }
    }
    if (ok)
        for (int e = 0; e < ne; e++) {
            const int q = eq[e], o = O->blk[q], no = abs(O->bs[o]), i = O->ii[q], j = O->jj[q];
            if (sp && o == sp->o) { sp->xs[sp->qe[q]] += de[e]; continue; }
            if (O->bs[o] < 0) Xo[o][i] += de[e];
            else { Xo[o][i + (size_t)j * no] += de[e]; if (i != j) Xo[o][j + (size_t)i * no] += de[e]; }
        }
    if (verbose >= 2 || getenv("BRISK_PSTIME"))
        printf("postsolve: X projection onto A(X) = b: |r| %.2e -> %.2e (%d %s), |dX| %.2e, y'r %.2e%s: %s\n",
               r0, rn, it, direct ? "direct passes" : "CG", dn, yr, m1 >= 0 ? (m1 < m0 ? ", the errors it changes lower" : ", the errors it changes not lower") : "",
               ok ? "applied" : "rejected");
    if ((verbose >= 2 || getenv("BRISK_PSTIME")) && m1 >= 0) printf("   [projection: the largest error it changes %.2e -> %.2e]\n", m0, m1);
#undef KMV
    free(eid); free(qs); free(eq); free(ekey); free(cw); free(tw); free(xe); free(r); free(u); free(p); free(z); free(Kp); free(dg); free(de); free(res);
}

/* The removed duals through complementarity with the primal (ISSUES 93). The reduced solve
 * lifts to a primal optimum X of the problem as read; any dual optimum satisfies Z X = 0, so Z
 * vanishes on the range of X: with N a basis of that range (per block, from the eigenvalues of
 * X; for an LP block the slots where x > 0), Z(y) N = 0 is a linear system in y, and the lift
 * of the reduced dual nearly solves it: the kept rows are at the solve's accuracy, only the rows
 * of the removed indices (g' N, with g the coupling of a removed index to the kept ones) are
 * off, and that is the condition the recovery through the certificates needs (g in the range
 * of the reduced slack, the complement of N). The least-norm correction dy with
 * Z(y + dy) N = 0 moves the dual within the reduced dual optimal face (every point of which has
 * the same N, by complementarity with the same X) to the face that lifts.
 * Two kinds of rows: (Z N)_ik in the standard basis, consistent only to first order in the
 * error e of N (Z* N = Z* e), and the projected rows N' Z N, consistent to second order
 * (N*' Z* = 0 as well). lambda_min(Z) and the gap <Z, X> = tr(N' Z N Lambda) depend on the
 * projected rows, which get the weight 1; the others only keep the coupling of N to its
 * complement small and get the weight omega. One more row pins b'y = <C, X>.
 * The removed constraints' duals are unknowns like the others (a removed constraint's matrix
 * may couple its index to a kept one in the data as read: it is diagonal only on the face of
 * an earlier reduction). Their values are set by the passes through the certificates; those add
 * t_k A_k and so move the couplings again, which is why the correction is applied repeatedly
 * from the passes' result (the fixed point has Z N = 0 and the certificate margins).
 * The system is dense in the smaller of its dimensions (rows, m), factored once (the matrix
 * does not change between applications), with a few refinement passes against the exact
 * operator.                                                                             */
typedef struct {
    int mo, R, Rz, Rstd;      /* columns, rows with the objective row, without, standard rows */
    int *cptr, *rind; double *aval; size_t na;   /* A by columns */
    double *K; int dimK; int gram;   /* the Cholesky factor of A A' (gram 0, R x R) or A' A (gram 1, m x m) */
    double *crow;             /* the right-hand side of the rows for y = 0 (C N, N' C N, the objective) */
    double *cscale;           /* column scales (dy = cscale dz) */
    double omega;
} PSFix;

static void ps_fix_free(PSFix *F) {
    free(F->cptr); free(F->rind); free(F->aval); free(F->K); free(F->crow); free(F->cscale);
    memset(F, 0, sizeof(*F));
}

/* builds the system; returns 0 on success, 1 when not applicable (too large, no range) */
static int ps_fix_setup(PSFix *F, const PSOrig *O, double **Xo, const Postsolve *ps, int verbose) {
    const int mo = O->m, nbo = O->nblk;
    /* the removed constraints' duals are set by the passes; the correction prefers to compensate
     * their couplings through the kept duals, so their columns are scaled down (a change of
     * variables dy_k = s dz_k, least norm in z) */
    const double srem = getenv("BRISK_PSSREM") ? atof(getenv("BRISK_PSSREM")) : 1e-3;
    char *isrem = pcalloc(mo + 1, 1);
    if (ps) for (int ri = 0; ri < ps->nrec; ri++) if (ps->rec[ri].con >= 0 && ps->rec[ri].con < mo) isrem[ps->rec[ri].con] = 1;
    const double tau = getenv("BRISK_PSRANKTOL") ? atof(getenv("BRISK_PSRANKTOL")) : 1e-6;
    const double omega = getenv("BRISK_PSEW") ? atof(getenv("BRISK_PSEW")) : 1.0;
    const double gapmax = getenv("BRISK_PSGAP") ? atof(getenv("BRISK_PSGAP")) : 1e-3;
    const double t0 = ps_now();
    const int dbg = verbose >= 2 || getenv("BRISK_PSDEBUG") != NULL;
    memset(F, 0, sizeof(*F));
    F->mo = mo; F->omega = omega;
    double ecost = 0;
    for (int o = 0; o < nbo; o++) if (O->bs[o] > 0) ecost += pow((double)O->bs[o], 3.0);
    if (ecost > 4e10) { if (dbg) printf("postsolve: dual correction skipped (eigendecompositions %.1e)\n", ecost); return 1; }
    double **N = pcalloc(nbo + 1, sizeof(double *)), *lmax = pcalloc(nbo + 1, sizeof(double));
    double **W = pcalloc(nbo + 1, sizeof(double *));
    int *rk = pcalloc(nbo + 1, sizeof(int)), *rowoff = pcalloc(nbo + 2, sizeof(int)), *soff = pcalloc(nbo + 2, sizeof(int));
    int **pidx = pcalloc(nbo + 1, sizeof(int *));
    double **wc = pcalloc(nbo + 1, sizeof(double *));   /* row weights of the standard rows */
    int ret = 1;
    double gmax = 0;
    for (int o = 0; o < nbo; o++) {
        const int n = abs(O->bs[o]);
        if (O->bs[o] < 0) { for (int i = 0; i < n; i++) lmax[o] = fmax(lmax[o], Xo[o][i]); gmax = fmax(gmax, lmax[o]); continue; }
        N[o] = pmalloc(sizeof(double) * ((size_t)n * n + 1));
        W[o] = pmalloc(sizeof(double) * (n + 1));
        memcpy(N[o], Xo[o], sizeof(double) * (size_t)n * n);
        int lwork = -1, info = 0; double wq = 0;
        BL(dsyev_)("V", "L", &n, N[o], &n, W[o], &wq, &lwork, &info);
        lwork = (int)wq + 1;
        double *wk = pmalloc(sizeof(double) * (lwork + 1));
        BL(dsyev_)("V", "L", &n, N[o], &n, W[o], wk, &lwork, &info);
        free(wk);
        if (info != 0) { if (dbg) printf("postsolve: dual correction skipped (eigensolver failed on block %d)\n", o + 1); goto out; }
        lmax[o] = W[o][n - 1];
        gmax = fmax(gmax, lmax[o]);
    }
    if (!(gmax > 0)) goto out;
    int R = 0;
    for (int o = 0; o < nbo; o++) {
        const int n = abs(O->bs[o]);
        const double thr = tau * fmax(lmax[o], 1e-3 * gmax);
        rowoff[o] = R;
        if (O->bs[o] < 0) {
            pidx[o] = pmalloc(sizeof(int) * (n + 1));
            wc[o] = pmalloc(sizeof(double) * (n + 1));
            int np = 0;
            for (int i = 0; i < n; i++) { if (Xo[o][i] > thr) { wc[o][np] = 1.0; pidx[o][i] = np++; } else pidx[o][i] = -1; }
            rk[o] = np; R += np;
        } else {
            int r = 0;
            for (int i = n - 1; i >= 0 && W[o][i] > thr; i--) r++;
            /* the range must be separated: the largest eigenvalue left out well below the
             * smallest kept (taha1b: eigenvalues decaying continuously from 0.8 to 1e-7, no face
             * to speak of; the rows would be inconsistent and the correction useless) */
            if (r > 0 && r < n && W[o][n - r - 1] > gapmax * W[o][n - r]) {
                if (dbg) printf("postsolve: dual correction skipped (the range of X is not separated in block %d: eigenvalues %.1e / %.1e)\n", o + 1, W[o][n - r], W[o][n - r - 1]);
                goto out;
            }
            if (r > 0 && r < n) memmove(N[o], N[o] + (size_t)(n - r) * n, sizeof(double) * (size_t)n * r);   /* the range to the front */
            wc[o] = pmalloc(sizeof(double) * (r + 1));
            for (int c = 0; c < r; c++) wc[o][c] = omega * sqrt(W[o][n - r + c] / gmax);
            rk[o] = r; R += n * r;
            if (getenv("BRISK_PSDEBUG2")) {
                printf("   [range blk %d n %d: rank %d, eigenvalues", o + 1, n, r);
                for (int i = n - 1; i >= 0 && i >= n - r - 2; i--) printf(" %.2e", W[o][i]);
                printf("]\n");
            }
        }
    }
    rowoff[nbo] = R;
    if (R == 0) { if (dbg) printf("postsolve: dual correction skipped (the primal has no range)\n"); goto out; }
    const int Rstd = R;
    for (int o = 0; o < nbo; o++) { soff[o] = R; if (O->bs[o] > 0) R += rk[o] * (rk[o] + 1) / 2; }
    soff[nbo] = R;
    const int Rz = R;
    R++;                                            /* the objective row */
    double pobj = 0, bn = 0;
    for (size_t q = 0; q < O->nnz; q++) {
        if (O->con[q] >= 0) continue;
        const int o = O->blk[q], i = O->ii[q], j = O->jj[q], no = abs(O->bs[o]);
        pobj += O->bs[o] < 0 ? O->v[q] * Xo[o][i] : (i == j ? 1.0 : 2.0) * O->v[q] * Xo[o][i + (size_t)j * no];
    }
    for (int j = 0; j < mo; j++) bn += O->b[j] * O->b[j];
    const double wb = 10.0 / fmax(sqrt(bn), 1e-300);
    const int small = R < mo ? R : mo;
    if (small > 10000) { if (dbg) printf("postsolve: dual correction skipped (system %d x %d)\n", R, mo); goto out; }
    F->R = R; F->Rz = Rz; F->Rstd = Rstd;
    F->cscale = pmalloc(sizeof(double) * (mo + 1));
    for (int j = 0; j < mo; j++) F->cscale[j] = isrem[j] ? srem : 1.0;
    /* A by columns (constraints), and the rows' values at y = 0 */
    {
        ConIndex CI;
        conindex_build(O, &CI);
        size_t cap = O->nnz * 4 + 16, na = 0;
        int *cptr = pmalloc(sizeof(int) * (mo + 2)), *rind = pmalloc(sizeof(int) * cap);
        double *aval = pmalloc(sizeof(double) * cap);
        double *acc = pcalloc(R + 1, sizeof(double));
        int *touched = pmalloc(sizeof(int) * (R + 1)), nt = 0;
        double *crow = pcalloc(R + 1, sizeof(double));
#define ACC_ENTRY(q, v) do { \
        const int o_ = O->blk[q], i_ = O->ii[q], l_ = O->jj[q], n_ = abs(O->bs[o_]); \
        if (O->bs[o_] < 0) { const int p_ = pidx[o_][i_]; if (p_ >= 0) { const int rr = rowoff[o_] + p_; if (acc[rr] == 0) { touched[nt++] = rr; } acc[rr] += (v) * wc[o_][p_]; } } \
        else { for (int c_ = 0; c_ < rk[o_]; c_++) { \
            const double *nc = N[o_] + (size_t)c_ * n_; const double w_ = wc[o_][c_]; \
            int rr = rowoff[o_] + c_ * n_ + i_; \
            if (acc[rr] == 0) { touched[nt++] = rr; } acc[rr] += (v) * nc[l_] * w_; \
            if (i_ != l_) { rr = rowoff[o_] + c_ * n_ + l_; if (acc[rr] == 0) { touched[nt++] = rr; } acc[rr] += (v) * nc[i_] * w_; } \
            for (int d_ = 0; d_ <= c_; d_++) { \
                const double *nd = N[o_] + (size_t)d_ * n_; \
                const double s_ = i_ == l_ ? (v) * nc[i_] * nd[i_] : (v) * (nc[i_] * nd[l_] + nc[l_] * nd[i_]); \
                rr = soff[o_] + c_ * (c_ + 1) / 2 + d_; \
                if (acc[rr] == 0) { touched[nt++] = rr; } acc[rr] += s_; \
            } } } } while (0)
        for (int j = -1; j < mo; j++) {
            nt = 0;
            for (size_t e = CI.ptr[j + 1]; e < CI.ptr[j + 2]; e++) { const size_t q = CI.ord[e]; ACC_ENTRY(q, O->v[q]); }
            if (j < 0) { for (int t = 0; t < nt; t++) { crow[touched[t]] = acc[touched[t]]; acc[touched[t]] = 0; } crow[Rz] = wb * pobj; continue; }
            if (O->b[j] != 0) { if (acc[Rz] == 0) { touched[nt++] = Rz; } acc[Rz] += wb * O->b[j]; }
            cptr[j] = (int)na;
            if (na + nt > cap) { cap = 2 * (na + nt) + 16; rind = realloc(rind, sizeof(int) * cap); aval = realloc(aval, sizeof(double) * cap); }
            const double cs = isrem[j] ? srem : 1.0;
            for (int t = 0; t < nt; t++) {
                const int rr = touched[t];
                if (acc[rr] != 0) { rind[na] = rr; aval[na] = cs * acc[rr]; na++; }
                acc[rr] = 0;
            }
        }
        cptr[mo] = (int)na;
#undef ACC_ENTRY
        free(CI.ptr); free(CI.ord); free(acc); free(touched);
        F->cptr = cptr; F->rind = rind; F->aval = aval; F->na = na; F->crow = crow;
        /* the cost of the Gram matrix: the squares of the column (or row) counts */
        double gcost = 0;
        if (R <= mo) for (int j = 0; j < mo; j++) { const double c = cptr[j + 1] - cptr[j]; gcost += c * c; }
        else { int *rc = pcalloc(R + 1, sizeof(int)); for (size_t a = 0; a < na; a++) rc[rind[a]]++; for (int i = 0; i < R; i++) gcost += (double)rc[i] * rc[i]; free(rc); }
        gcost += pow((double)small, 3.0) / 3.0;
        if (gcost > 4e10) { if (dbg) printf("postsolve: dual correction skipped (system %d x %d, %.1e flops)\n", R, mo, gcost); goto out; }
    }
    /* the factor */
    if (R <= mo) {
        double *K = pcalloc((size_t)R * R + 1, sizeof(double)), *wk = pmalloc(sizeof(double) * ((size_t)R * R + 1));
        for (int j = 0; j < mo; j++)
            for (int a = F->cptr[j]; a < F->cptr[j + 1]; a++) {
                const int ra = F->rind[a]; const double va = F->aval[a];
                for (int b = F->cptr[j]; b < F->cptr[j + 1]; b++) if (F->rind[b] <= ra) K[ra + (size_t)F->rind[b] * R] += va * F->aval[b];
            }
        const int ok = chol_reg(R, K, wk) == 0;
        free(wk);
        if (!ok) { free(K); if (dbg) printf("postsolve: dual correction skipped (A A' could not be factored)\n"); goto out; }
        F->K = K; F->dimK = R; F->gram = 0;
    } else {
        int *rptr = pcalloc(R + 2, sizeof(int)), *cind = pmalloc(sizeof(int) * (F->na + 1));
        double *rval = pmalloc(sizeof(double) * (F->na + 1));
        for (size_t a = 0; a < F->na; a++) rptr[F->rind[a] + 2]++;
        for (int i = 0; i < R; i++) rptr[i + 2] += rptr[i + 1];
        for (int j = 0; j < mo; j++) for (int a = F->cptr[j]; a < F->cptr[j + 1]; a++) { const int w = rptr[F->rind[a] + 1]++; cind[w] = j; rval[w] = F->aval[a]; }
        double *G = pcalloc((size_t)mo * mo + 1, sizeof(double)), *wk = pmalloc(sizeof(double) * ((size_t)mo * mo + 1));
        for (int i = 0; i < R; i++)
            for (int a = rptr[i]; a < rptr[i + 1]; a++) {
                const int ja = cind[a]; const double va = rval[a];
                for (int b = rptr[i]; b < rptr[i + 1]; b++) if (cind[b] <= ja) G[ja + (size_t)cind[b] * mo] += va * rval[b];
            }
        free(rptr); free(cind); free(rval);
        const int ok = chol_reg(mo, G, wk) == 0;
        free(wk);
        if (!ok) { free(G); if (dbg) printf("postsolve: dual correction skipped (A' A could not be factored)\n"); goto out; }
        F->K = G; F->dimK = mo; F->gram = 1;
    }
    ret = 0;
    if (dbg) printf("postsolve: dual correction through complementarity: %d + %d rows (Z N, N'Z N) + the objective, %d columns, %zu nonzeros, omega %g, set up in %.2fs\n",
                    Rstd, Rz - Rstd, mo, F->na, omega, ps_now() - t0);
out:
    for (int o = 0; o < nbo; o++) { free(N[o]); free(W[o]); free(pidx[o]); free(wc[o]); }
    free(N); free(W); free(lmax); free(rk); free(rowoff); free(soff); free(pidx); free(wc); free(isrem);
    if (ret) ps_fix_free(F);
    return ret;
}

/* yout = yin + dy, dy the least-norm solution of A dy = (rows at yin); stats: |rows| before and
 * after, |dy|. Returns the residual after. */
static double ps_fix_apply(const PSFix *F, const double *yin, double *yout, int verbose, double *dnorm) {
    const int mo = F->mo, R = F->R;
    const int dbg = verbose >= 2 || getenv("BRISK_PSDEBUG") != NULL;
    double *rhs = pmalloc(sizeof(double) * (R + 1)), *res = pmalloc(sizeof(double) * (R + 1)), *dy = pcalloc(mo + 1, sizeof(double));
    memcpy(rhs, F->crow, sizeof(double) * R);
    for (int j = 0; j < mo; j++) { const double yz = yin[j] / F->cscale[j]; for (int a = F->cptr[j]; a < F->cptr[j + 1]; a++) rhs[F->rind[a]] -= yz * F->aval[a]; }
    double r0 = 0;
    for (int i = 0; i < R; i++) r0 += rhs[i] * rhs[i];
    r0 = sqrt(r0);
    double rn = r0;
    int passes = 0;
    double *u = pmalloc(sizeof(double) * (F->dimK + 1));
    memcpy(res, rhs, sizeof(double) * R);
    for (int pass = 0; pass < 4; pass++) {
        int one = 1, info = 0;
        if (!F->gram) {
            memcpy(u, res, sizeof(double) * R);
            BL(dpotrs_)("L", &R, &one, F->K, &R, u, &R, &info);
            for (int j = 0; j < mo; j++) { double s = 0; for (int a = F->cptr[j]; a < F->cptr[j + 1]; a++) s += F->aval[a] * u[F->rind[a]]; dy[j] += s; }
        } else {
            for (int j = 0; j < mo; j++) { double s = 0; for (int a = F->cptr[j]; a < F->cptr[j + 1]; a++) s += F->aval[a] * res[F->rind[a]]; u[j] = s; }
            BL(dpotrs_)("L", &mo, &one, F->K, &mo, u, &mo, &info);
            for (int j = 0; j < mo; j++) dy[j] += u[j];
        }
        memcpy(res, rhs, sizeof(double) * R);
        for (int j = 0; j < mo; j++) for (int a = F->cptr[j]; a < F->cptr[j + 1]; a++) res[F->rind[a]] -= F->aval[a] * dy[j];
        double rn2 = 0; for (int i = 0; i < R; i++) rn2 += res[i] * res[i];
        rn2 = sqrt(rn2); passes++;
        if (!(rn2 < 0.5 * rn)) { rn = fmin(rn, rn2); break; }
        rn = rn2;
    }
    double dn = 0;
    for (int j = 0; j < mo; j++) { dy[j] *= F->cscale[j]; yout[j] = yin[j] + dy[j]; dn += dy[j] * dy[j]; }
    dn = sqrt(dn);
    if (dnorm) *dnorm = dn;
    if (dbg) {
        double re = 0, rs = 0;
        for (int i = 0; i < F->Rstd; i++) re += res[i] * res[i];
        for (int i = F->Rstd; i < F->Rz; i++) rs += res[i] * res[i];
        printf("postsolve: dual correction: |rows| %.2e -> %.2e (%.1e coupling / omega, %.1e projected, %.1e objective) in %d passes, |dy| %.2e\n",
               r0, rn, sqrt(re), sqrt(rs), fabs(res[F->Rz]), passes, dn);
    }
    free(rhs); free(res); free(dy); free(u);
    return rn;
}

/* ------------------------------------------------------------------ main entry */
static inline size_t bsize(int bs) { return bs > 0 ? (size_t)bs * bs : (size_t)(-bs); }
/* 23 October 2026: Z of the problem as read with its cone-vector blocks (zc[o] the hub; brisk.h, xc_index): the length
 * of block o, a data entry's term added (the arrow's entries: the hub's diagonal is z_0, (h, i_t) is z_t; the other
 * diagonal entries repeat z_0's data and are skipped), and lambda_min */
static inline size_t ps_zlen(const PSOrig *O, const int *zc, int o) { return zc && zc[o] >= 0 ? (size_t)O->bs[o] : bsize(O->bs[o]); }
static inline void ps_zadd(const PSOrig *O, const int *zc, double **Z, size_t q, double coef) {
    const int o = O->blk[q], i = O->ii[q], j = O->jj[q], no = abs(O->bs[o]);
    if (O->bs[o] < 0) Z[o][i] += coef;
    else if (zc && zc[o] >= 0) { const int h = zc[o]; if (i == j) { if (i == h) Z[o][0] += coef; } else { const int u = i == h ? j : i; Z[o][u < h ? u + 1 : u] += coef; } }
    else { Z[o][i + (size_t)j * no] += coef; if (i != j) Z[o][j + (size_t)i * no] += coef; }
}
static inline double ps_zlam(const PSOrig *O, const int *zc, double *const *Z, int o, double *work, double delta) {
    const int no = abs(O->bs[o]);
    if (O->bs[o] < 0) { double v = 0; for (int i = 0; i < no; i++) v = fmin(v, Z[o][i]); return v; }
    if (zc && zc[o] >= 0) return xc_lammin_z(no, Z[o]);
    return ps_lammin_tol(no, Z[o], work, delta);
}

int postsolve(const Problem *P, const PSOrig *O, double **Xred, const double *yred,
              double ***Xo_out, double *yo, PSResult *res, int verbose, const int *xc_red, int *xc_o) {
    const double t0 = ps_now();
    const Postsolve *ps = P->ps;
    const int nbo = O->nblk, mo = O->m;
    memset(res, 0, sizeof(*res));
    /* 23 October 2026: blocks of Xred held as cone vectors (xc_red, brisk.h). A chordal or moment-form conversion maps
     * blocks into others: there they are completed first (main.c does not ask for vectors then; this is the safety net) */
    if (xc_o) for (int o = 0; o < nbo; o++) xc_o[o] = -1;
    double **Xred_d = NULL;
    if (xc_red && Xred && ps && (ps->mom || (ps->chordal && ps->nch > 0))) {
        Xred_d = pmalloc(sizeof(double *) * (P->nblk + 1));
        for (int k = 0; k < P->nblk; k++) {
            Xred_d[k] = Xred[k];
            if (xc_red[k] < 0) continue;
            const int n = P->blk[k].n;
            Xred_d[k] = pmalloc(sizeof(double) * ((size_t)n * n + 1));
            xc_complete(n, xc_red[k], Xred[k], Xred_d[k]);
        }
        Xred = Xred_d;
    }
    const int *xc_r = Xred_d ? NULL : xc_red;
    /* sparse mode for a large moment-form chordal block (see PSSpX) */
    PSSpX spx; memset(&spx, 0, sizeof(spx)); spx.o = -1;
    double *clam_k = NULL, *clam_o = NULL;   /* 4.42: lambda_min over the cliques of a completed decomposed block (by current / original block), NAN: none */
    const PSChordal **cch_k = NULL, **cch_o = NULL;   /* ... its clique tree, and (csel_o) the original index of each of its indices when the basis is a selection */
    int **csel_o = NULL;
    double **spXc = NULL;
    int *spsel = NULL, spncl = 0;
    /* ---- y */
    for (int i = 0; i < mo; i++) yo[i] = 0;
    for (int i = 0; i < P->m; i++) if (P->orig[i] >= 0) yo[P->orig[i]] = yred[i];
    /* (23 October 2026) a combination the presolve added and the facial reduction did not remove: its dual on the file's rows */
    if (ps) for (int i = 0; i < P->m; i++) if (P->orig[i] <= -2 && -2 - P->orig[i] < ps->ncomb)
        for (int u = ps->comb_ptr[-2 - P->orig[i]]; u < ps->comb_ptr[-1 - P->orig[i]]; u++) yo[ps->comb_i[u]] += yred[i] * ps->comb_w[u];
    /* ---- X */
    /* chordal conversions are undone first: the clique blocks are replaced by the
     * positive semidefinite completion of the converted block, at its old position */
    int nbl = P->nblk, nall = 0;
    double **Xl = NULL, **Xall = NULL;
    int *nsz = NULL;
    if (Xred) {
        Xl = pmalloc(sizeof(double *) * (P->nblk + 1));
        nsz = pmalloc(sizeof(int) * (P->nblk + 1));
        for (int k = 0; k < P->nblk; k++) { Xl[k] = Xred[k]; nsz[k] = P->blk[k].n; }
        if (ps && ps->chordal && ps->nch > 0) {
            Xall = pmalloc(sizeof(double *) * (ps->nch + 1));
            for (int q = ps->nch - 1; q >= 0; q--) {
                const PSChordal *r = &ps->ch[q];
                double *D = chordal_complete(r, Xl + (nbl - r->ncl));
                Xall[nall++] = D;
                nbl -= r->ncl;
                for (int k = nbl; k > r->pos; k--) { Xl[k] = Xl[k - 1]; nsz[k] = nsz[k - 1]; }
                Xl[r->pos] = D; nsz[r->pos] = r->n;
                nbl++;
            }
        }
    }
    /* moment form: y from the free pairs, X from the multipliers of the moment rows */
    double *wfull = NULL;
    if (ps && ps->mom) {
        const PSMoment *M = ps->mom;
        if (M->rowmap) {
            /* the multipliers of the rows as built: dropped slack rows from the dual
             * constraint of the eliminated free column, w_j = -(b_i + sum_r A_ir w_r) / a */
            wfull = pcalloc(M->nrow + 1, sizeof(double));
            for (int r0 = 0; r0 < M->nrow; r0++) if (M->rowmap[r0] >= 0) wfull[r0] = yred[M->rowmap[r0]];
            for (int q = 0; q < M->ne; q++) {
                double t = M->e_b[q];
                for (int e = M->e_ptr[q]; e < M->e_ptr[q + 1]; e++) t += M->e_v[e] * yred[M->e_r[e]];
                wfull[M->e_row[q]] = -t / M->e_a[q];
            }
            yred = wfull;
        }
        if (Xred) {
            const double *xl = Xred[M->tlp];
            if (M->rowmap) {
                for (int i = 0; i < M->m0; i++) if (M->orig0[i] >= 0 && M->ymap[i] >= 0)
                    yo[M->orig0[i]] = (xl[M->off + M->ymap[i]] - xl[M->off + M->np + M->ymap[i]]) * M->ysc[i];
                for (int q = 0; q < M->ne; q++) if (M->orig0[M->e_i[q]] >= 0)
                    yo[M->orig0[M->e_i[q]]] = -xl[M->e_s[q]] * fabs(M->e_a[q]) * M->ysc[M->e_i[q]] / M->e_a[q];
            } else
                for (int i = 0; i < M->m0; i++) if (M->orig0[i] >= 0) yo[M->orig0[i]] = xl[M->off + i] - xl[M->off + M->m0 + i];
        }
        for (int q = 0; q < nall; q++) free(Xall[q]);
        free(Xall); free(Xl); free(nsz);
        nall = 0;
        nbl = M->nblk0;
        if (Xred && M->chpos >= 0 && M->chpos < nbl && ps->nblk_after == nbl) {
            /* 4.22: default for every moment-form block unless X itself is wanted (-x file,
             * crossover): the max-determinant completion of near-singular cliques is
             * ill-conditioned (dense case1888_rte: lambda_min of the completion -2.4e-4
             * relative, of the cliques -1.4e-8), and it costs several dense n x n copies */
            const char *en = getenv("BRISK_PSSPARSE_N");
            /* 4.42: g_ps_need_x = 2 is the library's request (MATLAB, Python, Julia: every call):
             * X is returned unless the dense form of the block (the long double completion,
             * X, Z and work space: about 48 n^2 bytes) exceeds 30% of the memory. Until 4.41
             * every library call completed the block (case13659: 10 GB for n = 35 503). */
            int soft_sparse = 0;
            if (g_ps_need_x == 2 && !en) {
                extern double brisk_mem_limit(void);
                soft_sparse = 48.0 * (double)M->ch.n * (double)M->ch.n > 0.3 * brisk_mem_limit();
                if (soft_sparse && verbose >= 0)
                    printf("postsolve: X is not returned: the dense form of the decomposed block (n = %d) needs %.1f GB; the point is verified on its cliques (-returnx 1 forces X)\n",
                           M->ch.n, 48.0 * (double)M->ch.n * (double)M->ch.n / 1e9);
            }
            const int nth = en ? atoi(en) : ((g_ps_need_x && !soft_sparse) ? 1 << 30 : 1);
            const int o = ps->cur2orig[M->chpos];
            const PSBasis *T = &ps->basis[o];
            int ok = nth > 0 && M->ch.n >= nth && O->bs[o] > 0 && T->n_cur == M->ch.n;
            if (ok && !T->identity)
                for (int c = 0; c < T->n_cur && ok; c++) ok = T->nz[c] == 1 && T->val[c][0] == 1.0;
            if (ok) {
                spx.o = o; spx.n0 = O->bs[o];
                spsel = pmalloc(sizeof(int) * (M->ch.n + 1));
                for (int c = 0; c < M->ch.n; c++) spsel[c] = T->identity ? c : T->idx[c][0];
                if (verbose >= 2 || getenv("BRISK_PSTIME")) printf("   [postsolve: block %d (n %d) measured on its cliques and data pattern]\n", o, spx.n0);
            }
        }
        Xl = pmalloc(sizeof(double *) * (nbl + 1));
        nsz = pmalloc(sizeof(int) * (nbl + 1));
        Xall = pmalloc(sizeof(double *) * (nbl + 1));
        clam_k = pmalloc(sizeof(double) * (nbl + 1));
        cch_k = pcalloc(nbl + 1, sizeof(PSChordal *));
        for (int k = 0; k < nbl; k++) {
            const int bs = M->bs0[k], n = abs(bs);
            clam_k[k] = NAN;
            nsz[k] = n;
            int dq = -1;
            for (int d = 0; d < M->nd; d++) if (M->dpos[d] == k) dq = d;
            if (dq >= 0) {
                /* a decomposed block (4.34: any number of them; the sparse measure takes the
                 * largest, dq = 0, the others are completed) */
                const PSChordal *ch = &M->dch[dq];
                const int ncl = ch->ncl;
                double **Xc = pmalloc(sizeof(double *) * (ncl + 1));
                for (int t = 0; t < ncl; t++) {
                    const int c = ch->cptr[t + 1] - ch->cptr[t];
                    Xc[t] = pmalloc(sizeof(double) * ((size_t)c * c + 1));
                    size_t q = M->dcoff[dq][t];
                    for (int b = 0; b < c; b++)
                        for (int a = 0; a <= b; a++) {
                            const double x = -yred[M->dcrow[dq][q++]] * (a == b ? 1.0 : 0.5);
                            Xc[t][a + (size_t)b * c] = Xc[t][b + (size_t)a * c] = x;
                        }
                }
                if (spx.o >= 0 && dq == 0) { spXc = Xc; spncl = ncl; Xl[k] = pcalloc(2, sizeof(double)); }
                else {
                    /* 4.42: a positive semidefinite completion exists iff every clique is
                     * (Grone et al.): lambda_min over the cliques is the measure of this
                     * block when X itself is not returned */
                    { int cm_ = 1; for (int t = 0; t < ncl; t++) { const int c = ch->cptr[t + 1] - ch->cptr[t]; if (c > cm_) cm_ = c; }
                      double *wkc = pmalloc(sizeof(double) * ((size_t)cm_ * cm_ + 1)), lm = 0;
                      for (int t = 0; t < ncl; t++) { const int c = ch->cptr[t + 1] - ch->cptr[t]; lm = fmin(lm, ps_lammin_tol(c, Xc[t], wkc, 1e-12 * (1.0 + O->nb1))); }
                      clam_k[k] = lm; cch_k[k] = ch; free(wkc); }
                    if (spx.o >= 0) {
                        /* 4.42: X is not returned: only the entries on the data pattern are
                         * read (A(X), <X,Z>, the projection), and they lie in the cliques. The
                         * clique entries are scattered into the block instead of the
                         * max-determinant completion (a long double n x n matrix and O(n^3)
                         * work: 1.1 GB for the 8 185 block of case13659) */
                        const int nn_ = ch->n;
                        double *D = pcalloc((size_t)nn_ * nn_ + 1, sizeof(double));
                        for (int t = 0; t < ncl; t++) {
                            const int c = ch->cptr[t + 1] - ch->cptr[t]; const int *cvv = ch->cv + ch->cptr[t];
                            for (int b2 = 0; b2 < c; b2++) for (int a2 = 0; a2 < c; a2++) D[cvv[a2] + (size_t)cvv[b2] * nn_] = Xc[t][a2 + (size_t)b2 * c];
                        }
                        Xl[k] = D;
                    } else
                    Xl[k] = chordal_complete(ch, Xc);
                    for (int t = 0; t < ncl; t++) free(Xc[t]);
                    free(Xc);
                }
            } else if (bs > 0) {
                Xl[k] = pcalloc((size_t)n * n + 1, sizeof(double));
                for (int q2 = 0; q2 < n; q2++)
                    for (int p2 = 0; p2 <= q2; p2++) {
                        const double x = -yred[M->brow0[k] + q2 * (q2 + 1) / 2 + p2] * (p2 == q2 ? 1.0 : 0.5);
                        Xl[k][p2 + (size_t)q2 * n] = Xl[k][q2 + (size_t)p2 * n] = x;
                    }
            } else {
                Xl[k] = pcalloc(n + 1, sizeof(double));
                for (int j = 0; j < n; j++) Xl[k][j] = -yred[M->brow0[k] + j];
            }
            Xall[nall++] = Xl[k];
        }
        free(wfull);
    }
    const int mapped = ps && (!ps->chordal || ps->nch > 0 || ps->mom) && ps->nblk_after == nbl;
    double **Xo = NULL;
    if (Xred && mapped) {
        Xo = pcalloc(nbo + 1, sizeof(double *));
        int *cur = pmalloc(sizeof(int) * (nbo + 1));
        for (int o = 0; o < nbo; o++) cur[o] = -1;
        for (int k = 0; k < nbl; k++) cur[ps->cur2orig[k]] = k;
        if (clam_k) {
            clam_o = pmalloc(sizeof(double) * (nbo + 1)); cch_o = pcalloc(nbo + 1, sizeof(PSChordal *)); csel_o = pcalloc(nbo + 1, sizeof(int *));
            for (int o = 0; o < nbo; o++) {
                clam_o[o] = cur[o] >= 0 ? clam_k[cur[o]] : NAN;
                if (cur[o] < 0 || !cch_k[cur[o]]) continue;
                const PSBasis *T = &ps->basis[o];
                const PSChordal *ch = cch_k[cur[o]];
                int sel = T->n_cur == ch->n;
                if (sel && !T->identity) for (int c = 0; c < T->n_cur && sel; c++) sel = T->nz[c] == 1 && T->val[c][0] == 1.0;
                if (!sel) continue;
                cch_o[o] = ch;
                csel_o[o] = pmalloc(sizeof(int) * (ch->n + 1));
                for (int c = 0; c < ch->n; c++) csel_o[o][c] = T->identity ? c : T->idx[c][0];
            }
        }
        for (int o = 0; o < nbo; o++) {
            const int bs = O->bs[o], no = abs(bs);
            if (o == spx.o) { Xo[o] = pcalloc(2, sizeof(double)); continue; }
            int k = cur[o];
            const PSBasis *T = &ps->basis[o];
            double *Xkd = NULL;                          /* a cone vector's completion, when the block is mapped */
            if (k >= 0 && xc_r && xc_r[k] >= 0) {
                if (T->identity && bs > 0 && nsz[k] == no) {
                    /* 23 October 2026: the block as read, held as its cone vector */
                    Xo[o] = pcalloc((size_t)no + 1, sizeof(double));
                    memcpy(Xo[o], Xl[k], sizeof(double) * (size_t)no);
                    if (xc_o) xc_o[o] = xc_r[k];
                    else { double *D = pcalloc(bsize(bs) + 1, sizeof(double)); xc_complete(no, xc_r[k], Xo[o], D); free(Xo[o]); Xo[o] = D; }
                    continue;
                }
                Xkd = pmalloc(sizeof(double) * ((size_t)nsz[k] * nsz[k] + 1));
                xc_complete(nsz[k], xc_r[k], Xl[k], Xkd);
            }
            Xo[o] = pcalloc(bsize(bs) + 1, sizeof(double));
            if (k < 0) continue;                         /* block removed entirely: X = 0 */
            const int nc = nsz[k];
            const double *Xk = Xkd ? Xkd : Xl[k];
            if (T->identity) { memcpy(Xo[o], Xk, sizeof(double) * bsize(bs)); free(Xkd); continue; }
            if (bs < 0) {                                /* LP: selection */
                for (int c = 0; c < nc; c++)
                    for (int e = 0; e < T->nz[c]; e++) Xo[o][T->idx[c][e]] += T->val[c][e] * Xk[c];
                free(Xkd);
                continue;
            }
            /* Y = T Xk (no x nc), Xo = Y T' */
            double *Y = pcalloc((size_t)no * nc + 1, sizeof(double));
            for (int a = 0; a < nc; a++)
                for (int e = 0; e < T->nz[a]; e++) {
                    const double v = T->val[a][e];
                    const int i = T->idx[a][e];
                    for (int c = 0; c < nc; c++) Y[i + (size_t)c * no] += v * Xk[a + (size_t)c * nc];
                }
            for (int a = 0; a < nc; a++)
                for (int e = 0; e < T->nz[a]; e++) {
                    const double v = T->val[a][e];
                    double *dst = Xo[o] + (size_t)T->idx[a][e] * no;
                    const double *src = Y + (size_t)a * no;
                    for (int i = 0; i < no; i++) dst[i] += v * src[i];
                }
            free(Y); free(Xkd);
        }
        free(cur);
        res->have_x = 1;
    }
    for (int q = 0; q < nall; q++) free(Xall[q]);
    free(Xall); free(Xl); free(nsz);
    const PSChordal *spch = (spx.o >= 0) ? &ps->mom->ch : NULL;
    if (spx.o >= 0) {
        /* the data positions of the block, X there from the cliques */
        const uint64_t n0 = (uint64_t)spx.n0;
        size_t nq = 0;
        for (size_t q = 0; q < O->nnz; q++) if (O->blk[q] == spx.o) nq++;
        spx.key = pmalloc(sizeof(uint64_t) * (nq + 1));
        nq = 0;
        for (size_t q = 0; q < O->nnz; q++) if (O->blk[q] == spx.o) spx.key[nq++] = (uint64_t)O->ii[q] * n0 + (uint64_t)O->jj[q];
        qsort(spx.key, nq, sizeof(uint64_t), cmp_u64);
        size_t w = 0;
        for (size_t t = 0; t < nq; t++) if (w == 0 || spx.key[t] != spx.key[w - 1]) spx.key[w++] = spx.key[t];
        spx.ne = (int)w;
        spx.qe = pmalloc(sizeof(int) * (O->nnz + 1));
        for (size_t q = 0; q < O->nnz; q++)
            spx.qe[q] = O->blk[q] == spx.o ? spx_find(&spx, (uint64_t)O->ii[q] * n0 + (uint64_t)O->jj[q]) : -1;
        spx.xs = pcalloc(spx.ne + 1, sizeof(double));
        spx.zs = pcalloc(spx.ne + 1, sizeof(double));
        for (int t = 0; t < spncl; t++) {
            const int c = spch->cptr[t + 1] - spch->cptr[t];
            const int *cv = spch->cv + spch->cptr[t];
            for (int b = 0; b < c; b++)
                for (int a = 0; a < c; a++) {
                    int i = spsel[cv[a]], j = spsel[cv[b]];
                    if (i > j) continue;
                    const int e = spx_find(&spx, (uint64_t)i * n0 + (uint64_t)j);
                    if (e >= 0) spx.xs[e] = spXc[t][a + (size_t)b * c];
                }
        }
        res->have_x = 1;
    }
    if (getenv("BRISK_PSTIME")) printf("   [postsolve %.3fs: X mapped]\n", ps_now() - t0);
    /* ---- Z = C - A'y (full storage; 23 October 2026: a cone-vector block of X has Z as its cone vector, zc, unless a
     * constraint removed by the presolve has data in the block: the recovery passes add its terms to Z there) */
    int *zc = NULL;
    if (xc_o && Xo) {
        for (int o = 0; o < nbo; o++) if (xc_o[o] >= 0) {
            if (!zc) { zc = pmalloc(sizeof(int) * (nbo + 1)); for (int q = 0; q < nbo; q++) zc[q] = -1; }
            zc[o] = xc_o[o];
        }
        if (zc && P->m < mo) {
            char *kept = pcalloc((size_t)mo + 1, 1);
            for (int i = 0; i < P->m; i++) if (P->orig[i] >= 0) kept[P->orig[i]] = 1;
            for (size_t q = 0; q < O->nnz; q++) if (O->con[q] >= 0 && !kept[O->con[q]] && zc[O->blk[q]] >= 0) zc[O->blk[q]] = -1;
            free(kept);
        }
    }
    double **Zo = pmalloc(sizeof(double *) * (nbo + 1));
    for (int o = 0; o < nbo; o++) Zo[o] = pcalloc(o == spx.o ? 2 : ps_zlen(O, zc, o) + 1, sizeof(double));
    for (size_t q = 0; q < O->nnz; q++) {
        const int o = O->blk[q];
        const double coef = O->con[q] < 0 ? O->v[q] : -yo[O->con[q]] * O->v[q];
        if (coef == 0) continue;
        if (o == spx.o) { spx.zs[spx.qe[q]] += coef; continue; }
        ps_zadd(O, zc, Zo, q, coef);
    }
    if (getenv("BRISK_PSTIME")) printf("   [postsolve %.3fs: Z formed]\n", ps_now() - t0);
    /* ---- dual recovery for constraints removed by facial reduction. The back-off margin
     * matters on deep chains (a slack left on the boundary of one level is amplified by the
     * next), and the best value is problem dependent; the passes are cheap next to the solve,
     * so a few margins are tried and the most dual-feasible result kept. */
    if (ps && ps->nrec > 0 && ps_late()) {
        res->unrecovered = ps->nrec; res->late = 1;
        if (verbose >= 0) printf("postsolve: the time limit is reached: the duals of the %d constraint(s) the presolve removed are not recovered\n", ps->nrec);
    } else if (ps && ps->nrec > 0 && spx.o >= 0) {
        /* sparse mode: the recovery passes of the dense path, the sparse block's slack on
         * its data pattern (diagonal-type reductions; others are left unrecovered) */
        size_t mx = 1;
        for (int o = 0; o < nbo; o++) if (o != spx.o && bsize(O->bs[o]) > mx) mx = bsize(O->bs[o]);
        double *wk = pmalloc(sizeof(double) * mx);
        const double dtol = 1e-12 * (1.0 + O->nC1);
#define SP_LAM(out) do { double b_ = 0, v_ = 0; \
            for (int o = 0; o < nbo; o++) { const int no = abs(O->bs[o]); \
                if (o == spx.o) { if (spx_lammin(&spx, spx.zs, dtol, &v_)) v_ = -INFINITY; b_ = fmin(b_, v_); } \
                else if (O->bs[o] < 0) { for (int i = 0; i < no; i++) b_ = fmin(b_, Zo[o][i]); } \
                else b_ = fmin(b_, ps_lammin_tol(no, Zo[o], wk, dtol)); } \
            (out) = b_; } while (0)
        double lz0;
        SP_LAM(lz0);
        double best = -lz0;
        if (verbose >= 2 || getenv("BRISK_PSTIME")) printf("postsolve: removed duals at zero: lambda_min(Z) %.2e (1 + nC1 = %.2e)\n", lz0, 1.0 + O->nC1);
        if (best > 1e-9 * (1.0 + O->nC1)) {
            if (!ps->dual_ok) res->unrecovered = ps->nrec;
            else {
                ConIndex CI;
                conindex_build(O, &CI);
                static const double margins[] = { 3.0, 1.0, 10.0, 0.1 };
                double *y0 = pmalloc(sizeof(double) * (mo + 1)), *ybest = pmalloc(sizeof(double) * (mo + 1));
                double *zs0 = pmalloc(sizeof(double) * (spx.ne + 1));
                double **Z0 = pmalloc(sizeof(double *) * (nbo + 1));
                for (int o = 0; o < nbo; o++) {
                    const size_t len = o == spx.o ? 1 : bsize(O->bs[o]);
                    Z0[o] = pmalloc(sizeof(double) * (len + 1));
                    memcpy(Z0[o], Zo[o], sizeof(double) * len);
                }
                memcpy(y0, yo, sizeof(double) * mo); memcpy(ybest, yo, sizeof(double) * mo);
                memcpy(zs0, spx.zs, sizeof(double) * spx.ne);
                int brec = 0, bunrec = ps->nrec;
                double prevlz = -1;
                for (int k = 0; k < 4; k++) {
                    if (k > 0 && ps_late()) { res->late = 1; break; }   /* (ISSUES 284: the best margin so far) */
                    if (k > 0) {
                        memcpy(yo, y0, sizeof(double) * mo);
                        memcpy(spx.zs, zs0, sizeof(double) * spx.ne);
                        for (int o = 0; o < nbo; o++) if (o != spx.o) memcpy(Zo[o], Z0[o], sizeof(double) * bsize(O->bs[o]));
                    }
                    int rc = 0, un = 0;
                    double lz = 0;
                    if (recover_pass(ps, O, &CI, yo, Zo, margins[k], &rc, &un, &spx)) lz = -INFINITY;
                    else SP_LAM(lz);
                    if (!(lz == lz)) lz = -INFINITY;
                    if (lz > -INFINITY) lz = fmin(lz, -z_round_est(O, yo));
                    if (verbose >= 2 || getenv("BRISK_PSTIME")) printf("postsolve: recovery margin %g: lambda_min(Z) %.2e (%d recovered, %d not)\n", margins[k], lz, rc, un);
                    if (k == 0 && fabs(-lz - best) <= 1e-3 * best) break;
                    /* (23 October 2026: not after a pass that diverged, prevlz = inf, where inf <= 1e-3 inf held and
                     * the search stopped before the next margin's result was kept; ISSUES 259) */
                    if (k > 0 && isfinite(prevlz) && fabs(-lz - prevlz) <= 1e-3 * fmax(prevlz, 1e-300)) break;
                    prevlz = -lz;
                    if (-lz < best) { best = -lz; brec = rc; bunrec = un; memcpy(ybest, yo, sizeof(double) * mo); }
                    if (best <= 1e-10 * (1.0 + O->nC1)) break;
                }
                memcpy(yo, ybest, sizeof(double) * mo);
                /* rebuild Z from the kept y */
                for (int o = 0; o < nbo; o++) if (o != spx.o) memset(Zo[o], 0, sizeof(double) * bsize(O->bs[o]));
                memset(spx.zs, 0, sizeof(double) * spx.ne);
                for (size_t q = 0; q < O->nnz; q++) {
                    const int o = O->blk[q], i = O->ii[q], j = O->jj[q], no = abs(O->bs[o]);
                    const double coef = O->con[q] < 0 ? O->v[q] : -yo[O->con[q]] * O->v[q];
                    if (coef == 0) continue;
                    if (o == spx.o) spx.zs[spx.qe[q]] += coef;
                    else if (O->bs[o] < 0) Zo[o][i] += coef;
                    else { Zo[o][i + (size_t)j * no] += coef; if (i != j) Zo[o][j + (size_t)i * no] += coef; }
                }
                res->recovered = brec; res->unrecovered = bunrec;
                for (int o = 0; o < nbo; o++) free(Z0[o]);
                free(Z0); free(y0); free(ybest); free(zs0);
                free(CI.ptr); free(CI.ord);
            }
        }
#undef SP_LAM
        free(wk);
    } else if (ps && ps->nrec > 0) {
        if (!ps->dual_ok) {
            if (verbose >= 0) printf("postsolve: reduction log over its memory cap, duals of removed constraints not recovered\n");
            res->unrecovered = ps->nrec;
        } else {
            ConIndex CI;
            conindex_build(O, &CI);
            static const double margins[] = { 3.0, 1.0, 10.0, 0.1 };
            const char *me = getenv("BRISK_PSMARGIN");
            const int nm = me ? 1 : (int)(sizeof(margins) / sizeof(margins[0]));
            size_t maxn2 = 1;
            for (int o = 0; o < nbo; o++) if (!(zc && zc[o] >= 0) && bsize(O->bs[o]) > maxn2) maxn2 = bsize(O->bs[o]);
            double *wk = pmalloc(sizeof(double) * maxn2);
            double *y0 = pmalloc(sizeof(double) * (mo + 1)), *ybest = pmalloc(sizeof(double) * (mo + 1));
            double **Z0 = pmalloc(sizeof(double *) * (nbo + 1));
            for (int o = 0; o < nbo; o++) {
                Z0[o] = pmalloc(sizeof(double) * (ps_zlen(O, zc, o) + 1));
                memcpy(Z0[o], Zo[o], sizeof(double) * ps_zlen(O, zc, o));
            }
            memcpy(y0, yo, sizeof(double) * mo);
            /* candidate 0: the removed constraints' duals left at zero, which is often
             * already dual feasible (then nothing is to be recovered) */
            double best = 0;
            for (int o = 0; o < nbo; o++) best = fmin(best, ps_zlam(O, zc, Zo, o, wk, 1e-12 * (1.0 + O->nC1)));
            best = best == best ? -best : INFINITY;
            if (getenv("BRISK_PSTIME")) printf("   [postsolve %.3fs: lambda_min with the removed duals at zero]\n", ps_now() - t0);
            double best0 = best;
            memcpy(ybest, yo, sizeof(double) * mo);
            int brec = 0, bunrec = 0;
            if (verbose >= 2 || getenv("BRISK_PSTIME")) printf("postsolve: removed duals at zero: lambda_min(Z) %.2e (1 + nC1 = %.2e)\n", -best, 1.0 + O->nC1);
            /* nothing to recover below a tenth of the default tolerance (the passes cost a
             * few dense factorizations of the original blocks each: 3.3 s on the 1781 x
             * 1781 AC-OPF block for a violation of 2e-10) */
            double prevlz = -1;
            /* round 0: the passes from the lift of the reduced dual. Later rounds (ISSUES 93): when
             * they leave a violation, the base is corrected through complementarity with the primal
             * (ps_fix_apply, from the previous round's best point with its certificate duals) and the
             * passes run again from the corrected base, until the correction is small or no round
             * improves */
            PSFix fix; int have_fix = 0, bround = 0;
            double *ybr = pmalloc(sizeof(double) * (mo + 1));   /* the best point of the round */
            double bestr = best;
            memcpy(ybr, y0, sizeof(double) * mo);
            const int maxround = getenv("BRISK_PSROUNDS") ? atoi(getenv("BRISK_PSROUNDS")) : 2;
            for (int round = 0; round < maxround; round++) {
            if (round >= 1 && ps_late()) { res->late = 1; break; }   /* (ISSUES 284) */
            if (round >= 1) {
                if (!(best > 1e-9 * (1.0 + O->nC1)) || !Xo || !g_ps_fr_correct) break;
                if (round == 1) {
                    /* (the correction reads X densely: cone-vector blocks are completed for it; Z keeps its form) */
                    if (xc_expand(O, Xo, xc_o)) break;
                    if (ps_fix_setup(&fix, O, Xo, ps, verbose)) break;
                    have_fix = 1;
                }
                double dn = 0;
                ps_fix_apply(&fix, ybr, y0, verbose, &dn);
                if (getenv("BRISK_PSTIME")) printf("   [postsolve %.3fs: dual correction round %d]\n", ps_now() - t0, round);
                for (int o = 0; o < nbo; o++) memset(Z0[o], 0, sizeof(double) * ps_zlen(O, zc, o));
                for (size_t q = 0; q < O->nnz; q++) {
                    const double coef = O->con[q] < 0 ? O->v[q] : -y0[O->con[q]] * O->v[q];
                    if (coef == 0) continue;
                    ps_zadd(O, zc, Z0, q, coef);
                }
                double lb = 0;
                for (int o = 0; o < nbo; o++) lb = fmin(lb, ps_zlam(O, zc, Z0, o, wk, 1e-12 * (1.0 + O->nC1)));
                best0 = lb == lb ? -lb : INFINITY;
                if (verbose >= 2 || getenv("BRISK_PSTIME")) printf("postsolve: corrected base (round %d): lambda_min(Z) %.2e, |dy| %.2e\n", round, -best0, dn);
                if (best0 < best) { best = best0; brec = 0; bunrec = 0; bround = round; memcpy(ybest, y0, sizeof(double) * mo); }
                prevlz = -1;
            }
            bestr = INFINITY;   /* the round's best pass result; a diverging chain leaves its partial lift as the next base */
            for (int k = 0; k < nm && best > 1e-9 * (1.0 + O->nC1); k++) {
                if ((k > 0 || round > 0) && ps_late()) { res->late = 1; break; }   /* (ISSUES 284: the best so far) */
                if (k > 0 || round > 0) {
                    memcpy(yo, y0, sizeof(double) * mo);
                    for (int o = 0; o < nbo; o++) memcpy(Zo[o], Z0[o], sizeof(double) * ps_zlen(O, zc, o));
                }
                int rc = 0, un = 0;
                double lz = 0;
                if (recover_pass(ps, O, &CI, yo, Zo, me ? atof(me) : margins[k], &rc, &un, NULL)) { lz = -INFINITY; if (round >= 1 && bestr == INFINITY) memcpy(ybr, yo, sizeof(double) * mo); }
                if (getenv("BRISK_PSTIME")) printf("   [postsolve %.3fs: recovery pass]\n", ps_now() - t0);
                if (lz == 0) for (int o = 0; o < nbo; o++)   /* (4.22: was skipped under BRISK_PSTIME) */
                    lz = fmin(lz, ps_zlam(O, zc, Zo, o, wk, 1e-12 * (1.0 + O->nC1)));
                if (!(lz == lz)) lz = -INFINITY;
                if (lz > -INFINITY) lz = fmin(lz, -z_round_est(O, yo));
                if (verbose >= 2 || getenv("BRISK_PSTIME")) printf("postsolve: recovery margin %g: lambda_min(Z) %.2e\n", me ? atof(me) : margins[k], lz);
                /* the removed constraints do not matter for lambda_min (same value as with
                 * their duals at zero): the violation comes from elsewhere, stop (each pass
                 * costs dense factorizations: 7 s on the 3228 x 3228 AC-OPF block) */
                if (k == 0 && fabs(-lz - best0) <= 1e-3 * best0) { if (verbose >= 2 || getenv("BRISK_PSTIME")) printf("postsolve: recovery does not change lambda_min, stopped\n"); break; }
                /* 4.20: a margin that changes nothing (case1888_rte: -6.62e-7 for all four,
                 * 10 s per pass) ends the search */
                /* 23 October 2026 (ISSUES 259): only after a pass that ended finite. After a diverging pass (margin 1 on
                 * acopf_case118, SDP with SOCP) prevlz is inf and inf <= 1e-3 inf held: the search stopped before the
                 * next margin's result (-1.8e-6 against -2.1e-4) was kept */
                if (k > 0 && isfinite(prevlz) && fabs(-lz - prevlz) <= 1e-3 * fmax(prevlz, 1e-300)) { if (verbose >= 2 || getenv("BRISK_PSTIME")) printf("postsolve: margin does not change lambda_min, stopped\n"); break; }
                prevlz = -lz;
                if (-lz < bestr) { bestr = -lz; memcpy(ybr, yo, sizeof(double) * mo); }
                if (-lz < best) {
                    best = -lz; brec = rc; bunrec = un; bround = round;
                    memcpy(ybest, yo, sizeof(double) * mo);
                }
                if (best <= 1e-10 * (1.0 + O->nC1)) break;
            }
            }   /* rounds */
            if (have_fix) ps_fix_free(&fix);
            free(ybr);
            res->fr_corrected = bround >= 1;
            if (bround >= 1 && (verbose >= 2 || getenv("BRISK_PSTIME"))) printf("postsolve: the corrected lift is kept (round %d): lambda_min(Z) %.2e\n", bround, -best);
            memcpy(yo, ybest, sizeof(double) * mo);
            /* rebuild Z from the kept y */
            for (int o = 0; o < nbo; o++) memset(Zo[o], 0, sizeof(double) * ps_zlen(O, zc, o));
            for (size_t q = 0; q < O->nnz; q++) {
                const double coef = O->con[q] < 0 ? O->v[q] : -yo[O->con[q]] * O->v[q];
                if (coef == 0) continue;
                ps_zadd(O, zc, Zo, q, coef);
            }
            res->recovered = brec; res->unrecovered = bunrec;
            for (int o = 0; o < nbo; o++) free(Z0[o]);
            free(Z0); free(y0); free(ybest); free(wk);
            free(CI.ptr); free(CI.ord);
        }
    }
    if (getenv("BRISK_PSTIME")) printf("   [postsolve %.3fs: dual recovery]\n", ps_now() - t0);
    /* 24 October 2026 (ISSUES 275): the projection of X onto A(X) = b after the recovery of the duals of the constraints
     * the facial reduction removed, so that it is judged on the y that is returned: before it, b'y lacked their terms,
     * and the gap the projection measured was that of a point never returned (7.1 on pglib_opf_case588_sdet, SDP, one
     * constraint removed, before and after the correction alike). The recovery reads X only in its correction rounds
     * on the dense path (ps_fix_setup), which now read X as the solve left it. */
    if (Xo && ps && ps->mom && !getenv("BRISK_NOPROJ") && !ps_late())   /* 4.22: default (direct solve); ISSUES 284 */
        ps_project_x(O, Xo, yo, verbose, spx.o >= 0 ? &spx : NULL, spch, spsel, spXc, spncl, clam_o, (const PSChordal *const *)cch_o, csel_o);
    if (spx.o >= 0) {
        /* the projected values back into the cliques */
        const uint64_t n0 = (uint64_t)spx.n0;
        for (int t = 0; t < spncl; t++) {
            const int c = spch->cptr[t + 1] - spch->cptr[t];
            const int *cv = spch->cv + spch->cptr[t];
            for (int b = 0; b < c; b++)
                for (int a = 0; a < c; a++) {
                    int i = spsel[cv[a]], j = spsel[cv[b]];
                    if (i > j) { const int tt = i; i = j; j = tt; }
                    const int e = spx_find(&spx, (uint64_t)i * n0 + (uint64_t)j);
                    if (e >= 0) spXc[t][a + (size_t)b * c] = spx.xs[e];
                }
        }
    }
    /* ---- errors in the original problem */
    /* (23 October 2026: a cone-vector block of X whose Z is held densely is completed: <X, Z> reads every entry) */
    if (xc_o && Xo) for (int o = 0; o < nbo; o++) if (xc_o[o] >= 0 && !(zc && zc[o] >= 0)) {
        const int no = O->bs[o];
        double *D = pmalloc(sizeof(double) * ((size_t)no * no + 1));
        xc_complete(no, xc_o[o], Xo[o], D); free(Xo[o]); Xo[o] = D; xc_o[o] = -1;
    }
    double *rp = pcalloc(mo + 1, sizeof(double));
    double pobj = 0, dobj = 0, xz = 0, lamx = 0, lamz = 0;
    for (int i = 0; i < mo; i++) dobj += O->b[i] * yo[i];
    size_t maxn2 = 1;
    for (int o = 0; o < nbo; o++) if (O->bs[o] > 0 && o != spx.o && bsize(O->bs[o]) > maxn2 && !(zc && zc[o] >= 0 && Xo && xc_o && xc_o[o] >= 0)) maxn2 = bsize(O->bs[o]);
    for (int t = 0; t < spncl; t++) { const size_t c = (size_t)(spch->cptr[t + 1] - spch->cptr[t]); if (c * c > maxn2) maxn2 = c * c; }
    double *work = pmalloc(sizeof(double) * maxn2);
    if (getenv("BRISK_PSTIME")) printf("   [postsolve %.3fs: the errors' measure begins]\n", ps_now() - t0);
    for (int o = 0; o < nbo; o++) {
        const int no = abs(O->bs[o]);
        if (o == spx.o) {
            double v = 0;
            /* (ISSUES 284: past the time limit not measured: on a point far from Z >= 0 every one of the up to 14 shifted
             * factorizations fails, 57 s on the block of order 27,317 of pglib_opf_case13659_pegase at its starting point) */
            if (ps_late()) {
                if (verbose >= 0) printf("postsolve: lambda_min(Z) of block %d not measured (the time limit is reached)\n", o + 1);
                v = -INFINITY; res->late = 1;
            } else if (spx_lammin(&spx, spx.zs, 1e-12 * (1.0 + O->nC1), &v)) {
                if (verbose >= 0) printf("postsolve: lambda_min(Z) of block %d not measured (sparse factorization failed)\n", o + 1);
                v = -INFINITY;
            }
            lamz = fmin(lamz, v);
            continue;
        }
        if (O->bs[o] < 0) {
            for (int i = 0; i < no; i++) lamz = fmin(lamz, Zo[o][i]);
        } else if (zc && zc[o] >= 0) lamz = fmin(lamz, ps_zlam(O, zc, Zo, o, work, 0.0));
        else if (no > 500 && ps_late()) {   /* (ISSUES 284: past the time limit a large block is not measured; on a point far
                                              * from Z >= 0 the Cholesky ladder fails nine times and the eigenvalues follow) */
            if (verbose >= 0) printf("postsolve: lambda_min(Z) of block %d not measured (the time limit is reached)\n", o + 1);
            lamz = -INFINITY; res->late = 1;
        }
        else { double lz_ = ps_lammin_tol(no, Zo[o], work, 1e-12 * (1.0 + O->nC1)); if (ENV_ON("BRISK_LZDBG")) { printf("   [block %d n %d lammin(Z) %.3e]\n", o, no, lz_); if (no <= 6) for (int i = 0; i < no; i++) { for (int j = 0; j < no; j++) printf(" %.6e", Zo[o][i + (size_t)j * no]); printf("\n"); } } lamz = fmin(lamz, lz_); }
    }
    if (getenv("BRISK_PSTIME")) printf("   [postsolve %.3fs: lambda_min(Z) measured]\n", ps_now() - t0);
    if (Xo) {
        for (size_t q = 0; q < O->nnz; q++) {
            const int o = O->blk[q], i = O->ii[q], j = O->jj[q], no = abs(O->bs[o]);
            double c;
            if (xc_o && xc_o[o] >= 0) {
                /* a cone-vector block (its data are arrow matrices, main.c): <A, X> = a_0 x_0 + sum_t a_t x_t, the hub's
                 * diagonal entry standing for the whole diagonal */
                const int h = xc_o[o];
                if (i == j) c = i == h ? O->v[q] * Xo[o][0] : 0.0;
                else { const int u = i == h ? j : i; c = O->v[q] * Xo[o][u < h ? u + 1 : u]; }
            } else {
                const double x = o == spx.o ? spx.xs[spx.qe[q]] : O->bs[o] < 0 ? Xo[o][i] : Xo[o][i + (size_t)j * no];
                c = (O->bs[o] > 0 && i != j) ? 2.0 * O->v[q] * x : O->v[q] * x;
            }
            if (O->con[q] < 0) pobj += c; else rp[O->con[q]] += c;
        }
        if (spx.o >= 0) {
            const uint64_t n0 = (uint64_t)spx.n0;
            for (int e = 0; e < spx.ne; e++) xz += (spx.key[e] / n0 == spx.key[e] % n0 ? 1.0 : 2.0) * spx.xs[e] * spx.zs[e];
            for (int t = 0; t < spncl; t++) {
                const int c = spch->cptr[t + 1] - spch->cptr[t];
                lamx = fmin(lamx, ps_lammin_tol(c, spXc[t], work, 1e-12 * (1.0 + O->nb1)));
            }
        }
        for (int o = 0; o < nbo; o++) {
            if (o == spx.o) continue;
            const int no = abs(O->bs[o]);
            if (xc_o && xc_o[o] >= 0) {
                /* both cone vectors (above): <X, Z> = x'z */
                for (int t = 0; t < no; t++) xz += Xo[o][t] * Zo[o][t];
                lamx = fmin(lamx, xc_lammin_x(no, Xo[o]));
                continue;
            }
            if (zc && zc[o] >= 0) {
                /* X dense (completed for the correction above), Z = Arw(z): <X, Z> = z_0 tr X + 2 sum_t z_t X_{h, i_t} */
                const int h = zc[o];
                double tr = 0;
                for (int i = 0; i < no; i++) tr += Xo[o][i + (size_t)i * no];
                xz += Zo[o][0] * tr;
                for (int t = 1; t < no; t++) xz += 2.0 * Zo[o][t] * Xo[o][(size_t)xc_index(h, t) + (size_t)h * no];
                lamx = fmin(lamx, ps_lammin_tol(no, Xo[o], work, 1e-12 * (1.0 + O->nb1)));
                continue;
            }
            const size_t len = bsize(O->bs[o]);
            for (size_t q = 0; q < len; q++) xz += Xo[o][q] * Zo[o][q];
            if (O->bs[o] < 0) { for (int i = 0; i < no; i++) lamx = fmin(lamx, Xo[o][i]); }
            else if (spx.o >= 0 && clam_o && clam_o[o] == clam_o[o]) {
                /* 4.42: X is not returned (the sparse measure is on): a decomposed block is
                 * measured on its cliques, like the largest one, not on its completion. The
                 * cliques are read from the block as it is now (after the projection onto
                 * A(X) = b) when its basis is a selection */
                if (cch_o && cch_o[o]) {
                    const PSChordal *ch = cch_o[o]; const int *sel = csel_o[o];
                    int cm_ = 1; for (int t = 0; t < ch->ncl; t++) { const int c = ch->cptr[t + 1] - ch->cptr[t]; if (c > cm_) cm_ = c; }
                    double *G = pmalloc(sizeof(double) * ((size_t)cm_ * cm_ + 1)), *wkc = pmalloc(sizeof(double) * ((size_t)cm_ * cm_ + 1)), lm = 0;
                    for (int t = 0; t < ch->ncl; t++) {
                        const int c = ch->cptr[t + 1] - ch->cptr[t]; const int *cvv = ch->cv + ch->cptr[t];
                        for (int b2 = 0; b2 < c; b2++) for (int a2 = 0; a2 < c; a2++) G[a2 + (size_t)b2 * c] = Xo[o][sel[cvv[a2]] + (size_t)sel[cvv[b2]] * no];
                        lm = fmin(lm, ps_lammin_tol(c, G, wkc, 1e-12 * (1.0 + O->nb1)));
                    }
                    free(G); free(wkc);
                    clam_o[o] = lm;
                }
                lamx = fmin(lamx, clam_o[o]);
                if (getenv("BRISK_PSTIME")) printf("   [postsolve: block %d (n %d) measured on its cliques: lambda_min %.3e]\n", o, no, clam_o[o]);
            }
            else if (no > 500 && ps_late()) { lamx = -INFINITY; res->late = 1; }   /* (ISSUES 284, as for Z above) */
            else lamx = fmin(lamx, ps_lammin_tol(no, Xo[o], work, 1e-12 * (1.0 + O->nb1)));
        }
        if (getenv("BRISK_PSTIME")) printf("   [postsolve %.3fs: <C,X>, A(X), <X,Z> and lambda_min(X) measured]\n", ps_now() - t0);
        double r2 = 0;
        for (int i = 0; i < mo; i++) {
            double v = rp[i] - O->b[i];
            if (O->tbR != 0 && O->b[i] == O->tbR) v *= (1.0 + O->nb1) / O->tbR;   /* 4.30: the trace-bound row, relative to R */
            r2 += v * v;
        }
        if (getenv("BRISK_PSTIME") && ps && ps->mom && ps->mom->rowmap) {
            const PSMoment *M = ps->mom;
            double re = 0, rk = 0, yr = 0;
            char *isel = pcalloc(mo + 1, 1);
            for (int q = 0; q < M->ne; q++) if (M->orig0[M->e_i[q]] >= 0) isel[M->orig0[M->e_i[q]]] = 1;
            for (int i = 0; i < mo; i++) { double v = rp[i] - O->b[i]; if (isel[i]) re += v * v; else rk += v * v; yr += yo[i] * v; }
            printf("   [postsolve: |A(X)-b| eliminated rows %.2e, kept rows %.2e, y'(b - A(X)) %.2e]\n", sqrt(re), sqrt(rk), -yr);
            int wq = -1; double wv = 0;
            for (int q = 0; q < M->ne; q++) { const int i = M->orig0[M->e_i[q]]; if (i < 0) continue; double v = fabs(rp[i] - O->b[i]); if (v > wv) { wv = v; wq = q; } }
            if (wq >= 0) {
                const int i = M->orig0[M->e_i[wq]];
                printf("   [worst eliminated: orig %d res %.2e b_file %.6e b_conv %.6e a %.3e nterms %d]\n", i, wv, O->b[i], M->e_b[wq], M->e_a[wq], M->e_ptr[wq + 1] - M->e_ptr[wq]);
            }
            free(isel);
        }
        double den = 1 + fabs(pobj + O->obj_off) + fabs(dobj + O->obj_off);
        res->err[1] = sqrt(r2) / (1 + O->nb1);
        res->err[2] = fmax(0.0, -lamx) / (1 + O->nb1);
        res->err[5] = (pobj - dobj) / den;
        res->err[6] = xz / den;
    }
    res->err[3] = 0;
    /* rounding uncertainty of Z (matters only after a recovery with large duals) */
    if (ps && ps->nrec > 0 && res->recovered > 0) lamz = fmin(lamz, -z_round_est(O, yo));
    res->err[4] = fmax(0.0, -lamz) / (1 + O->nC1);
    res->pobj = pobj;
    res->dobj = dobj;
    free(work); free(rp); free(clam_k); free(clam_o); free(cch_k); free(cch_o);
    if (csel_o) { for (int o = 0; o < nbo; o++) free(csel_o[o]); free(csel_o); }
    for (int o = 0; o < nbo; o++) free(Zo[o]);
    free(Zo);
    if (getenv("BRISK_PSTIME")) printf("   [postsolve %.3fs: end]\n", ps_now() - t0);
    if (spx.o >= 0) {
        /* X of the sparse block is not formed: nothing is returned */
        if (Xo) { for (int o = 0; o < nbo; o++) free(Xo[o]); free(Xo); Xo = NULL; }
        for (int t = 0; t < spncl; t++) free(spXc[t]);
        free(spXc); free(spsel); free(spx.key); free(spx.xs); free(spx.zs); free(spx.qe);
        if (spx.F) schol_free(spx.F);
        free(spx.off); free(spx.doff);
    }
    if (Xred_d) { for (int k = 0; k < P->nblk; k++) if (xc_red[k] >= 0) free(Xred_d[k]); free(Xred_d); }
    free(zc);
    res->t = ps_now() - t0;
    *Xo_out = Xo;
    return 0;
}

/* 4.30: the DIMACS errors of a given (Xo, yo) on O (dense blocks; Z = C - A'y formed here).
 * Used after the dual form is mapped back to the problem as read. */
/* 5.8 (a user's report): is the returned point a certificate of the infeasibility verdict on
 * the data as read?  Primal infeasible: a Farkas ray y with b'y > 0 and A'y <= 0; dual
 * infeasible: X >= 0 with A(X) = 0 and <C,X> < 0.  Returns 1 (verified), 0 (not: the verdict
 * rests on the presolve, e.g. a face found by the facial reduction, the problem is weakly
 * infeasible and no ray exists, or the verdict is wrong), -1 (no point to check).  The measures
 * returned for the log: obj = b'y or <C,X> with the ray scaled to unit norm, viol = the violation
 * of the certificate normalized to b'y = 1 or <C,X> = -1.
 * 22 October 2026 (the full benchmark: dg3d_1000, SDP with SOCP, reported DUAL INFEASIBLE "with
 * a certificate" on a ray whose descent is small against its norm, -4.8e-11 of |X| |C|): the violations
 * were measured relative to the ray's norm, so such a ray passed even with |A(X)| of 1e-7 |X|. A ray
 * proves nothing unless its descent dominates its residual: with <C,X'> = -1, a dual y feasible
 * for C - A'y >= 0 needs <y, A(X')> <= -1, so the ray excludes only the y with |y| < 1/|A(X')|
 * (likewise a Farkas y normalized to b'y' = 1 excludes only the X with tr X < 1/lambda_max(A'y')).
 * The violations are therefore measured after the normalization of the certificate's objective to
 * one, absolutely, as MOSEK states its certificates: accepted when at most 1e-6. */
int ps_infeas_check(const PSOrig *O, double **Xo, const double *yo, int status, double *obj, double *viol, const int *xc) {
    const int nbo = O->nblk, mo = O->m;
    *obj = 0; *viol = 0;
    /* 23 October 2026: xc, the cone-vector blocks of Xo (brisk.h; their data are arrow matrices): A'y there is held as its
     * cone vector as well (ps_zadd) */
    if (status == 1) {
        if (!yo) return -1;
        double ny = 0; for (int i = 0; i < mo; i++) ny += yo[i] * yo[i];
        ny = sqrt(ny); if (!(ny > 0)) return 0;
        double **W = pmalloc(sizeof(double *) * (nbo + 1));
        for (int o = 0; o < nbo; o++) W[o] = pcalloc(ps_zlen(O, xc, o) + 1, sizeof(double));
        double nw = 0;
        for (size_t q = 0; q < O->nnz; q++) {
            if (O->con[q] < 0) continue;
            const double coef = -yo[O->con[q]] / ny * O->v[q];      /* -A'y, scaled */
            if (coef == 0) continue;
            ps_zadd(O, xc, W, q, coef);
        }
        size_t maxn2 = 1;
        for (int o = 0; o < nbo; o++) if (O->bs[o] > 0 && !(xc && xc[o] >= 0) && bsize(O->bs[o]) > maxn2) maxn2 = bsize(O->bs[o]);
        double *work = pmalloc(sizeof(double) * maxn2);
        double lam = 0;
        for (int o = 0; o < nbo; o++) {
            const int no = abs(O->bs[o]);
            if (xc && xc[o] >= 0) {     /* |Arw(w)|_F^2 = n w_0^2 + 2 |wb|^2 */
                nw += (double)no * W[o][0] * W[o][0];
                for (int t = 1; t < no; t++) nw += 2.0 * W[o][t] * W[o][t];
                lam = fmin(lam, ps_zlam(O, xc, W, o, work, 0.0));
                continue;
            }
            const size_t len = bsize(O->bs[o]);
            for (size_t q = 0; q < len; q++) nw += W[o][q] * W[o][q];
            if (O->bs[o] < 0) { for (int i = 0; i < no; i++) lam = fmin(lam, W[o][i]); }
            else lam = fmin(lam, ps_lammin_tol(no, W[o], work, 0.0));
        }
        nw = sqrt(nw);
        double by = 0, byc = 0;   /* (compensated, as below) */
        for (int i = 0; i < mo; i++) { const double x = O->b[i] * yo[i], t = by + x; byc += fabs(by) >= fabs(x) ? (by - t) + x : (x - t) + by; by = t; }
        by += byc;
        *obj = by / ny / (1.0 + O->nb1);
        /* W = -A'y / |y|: its negative part, for y normalized to b'y = 1 (|y| / b'y times it) */
        *viol = by > 0 ? fmax(0.0, -lam) * ny / by : INFINITY;
        free(work);
        for (int o = 0; o < nbo; o++) free(W[o]);
        free(W);
        return (by > 0 && *viol <= 1e-6) ? 1 : 0;
    }
    if (status == 2) {
        if (!Xo) return -1;
        double nx = 0, lam = 0, cx = 0;
        size_t maxn2 = 1;
        for (int o = 0; o < nbo; o++) if (O->bs[o] > 0 && !(xc && xc[o] >= 0) && bsize(O->bs[o]) > maxn2) maxn2 = bsize(O->bs[o]);
        double *work = pmalloc(sizeof(double) * maxn2);
        for (int o = 0; o < nbo; o++) {
            const int no = abs(O->bs[o]); const size_t len = bsize(O->bs[o]);
            if (xc && xc[o] >= 0) {     /* the completion: |X|_F^2 = (x_0^2 + |xb|^2) / 2 */
                double s2 = Xo[o][0] * Xo[o][0];
                for (int t = 1; t < no; t++) s2 += Xo[o][t] * Xo[o][t];
                nx += 0.5 * s2;
                lam = fmin(lam, xc_lammin_x(no, Xo[o]));
                continue;
            }
            for (size_t q = 0; q < len; q++) nx += Xo[o][q] * Xo[o][q];
            if (O->bs[o] < 0) { for (int i = 0; i < no; i++) lam = fmin(lam, Xo[o][i]); }
            else lam = fmin(lam, ps_lammin_tol(no, Xo[o], work, 0.0));
        }
        free(work);
        nx = sqrt(nx); if (!(nx > 0)) return 0;
        /* 22 October 2026: compensated sums (Neumaier). A ray's entries are large (1e18 on dg3d_1000, SDP with SOCP) and
         * A(X) = 0 is a sum of terms that cancel: the plain sums left a rounding error of 3.4e6 in |A(X)|, 3.5e-6 of the
         * descent <C,X>, where the ray itself, read as written, gives 1.2e-8 */
        double *rp = pcalloc(mo + 1, sizeof(double)), *rpc = pcalloc(mo + 1, sizeof(double)), cxc = 0;
        #define PS_NSUM(s_, c_, x_) do { const double x__ = (x_), t__ = (s_) + x__; (c_) += fabs(s_) >= fabs(x__) ? ((s_) - t__) + x__ : (x__ - t__) + (s_); (s_) = t__; } while (0)
        for (size_t q = 0; q < O->nnz; q++) {
            const int o = O->blk[q], i = O->ii[q], j = O->jj[q], no = abs(O->bs[o]);
            double c;
            if (xc && xc[o] >= 0) {     /* <A, X> = a_0 x_0 + sum_t a_t x_t (as in postsolve) */
                const int h = xc[o];
                if (i == j) { if (i != h) continue; c = O->v[q] * Xo[o][0]; }
                else { const int u = i == h ? j : i; c = O->v[q] * Xo[o][u < h ? u + 1 : u]; }
            } else {
                const double x = O->bs[o] < 0 ? Xo[o][i] : Xo[o][i + (size_t)j * no];
                c = (O->bs[o] > 0 && i != j) ? 2.0 * O->v[q] * x : O->v[q] * x;
            }
            if (O->con[q] < 0) PS_NSUM(cx, cxc, c); else PS_NSUM(rp[O->con[q]], rpc[O->con[q]], c);
        }
        #undef PS_NSUM
        cx += cxc;
        double r2 = 0; for (int i = 0; i < mo; i++) { const double r = rp[i] + rpc[i]; r2 += r * r; }
        free(rp); free(rpc);
        *obj = cx / nx / (1.0 + O->nC1);
        /* the ray normalized to <C,X> = -1: |A(X)| and the negative part of X, absolutely */
        *viol = cx < 0 ? fmax(sqrt(r2), fmax(0.0, -lam)) / -cx : INFINITY;
        return (cx < 0 && *viol <= 1e-6) ? 1 : 0;
    }
    return -1;
}
int ps_measure(const PSOrig *O, double **Xo, const double *yo, PSResult *res) {
    const double t0 = ps_now();
    const int nbo = O->nblk, mo = O->m;
    memset(res, 0, sizeof(*res));
    double **Zo = pmalloc(sizeof(double *) * (nbo + 1));
    for (int o = 0; o < nbo; o++) Zo[o] = pcalloc(bsize(O->bs[o]) + 1, sizeof(double));
    for (size_t q = 0; q < O->nnz; q++) {
        const int o = O->blk[q], i = O->ii[q], j = O->jj[q], no = abs(O->bs[o]);
        const double coef = O->con[q] < 0 ? O->v[q] : -yo[O->con[q]] * O->v[q];
        if (coef == 0) continue;
        if (O->bs[o] < 0) Zo[o][i] += coef;
        else { Zo[o][i + (size_t)j * no] += coef; if (i != j) Zo[o][j + (size_t)i * no] += coef; }
    }
    size_t maxn2 = 1;
    for (int o = 0; o < nbo; o++) if (O->bs[o] > 0 && bsize(O->bs[o]) > maxn2) maxn2 = bsize(O->bs[o]);
    double *work = pmalloc(sizeof(double) * maxn2);
    double pobj = 0, dobj = 0, xz = 0, lamx = 0, lamz = 0;
    for (int i = 0; i < mo; i++) dobj += O->b[i] * yo[i];
    for (int o = 0; o < nbo; o++) {
        const int no = abs(O->bs[o]);
        if (O->bs[o] < 0) { for (int i = 0; i < no; i++) lamz = fmin(lamz, Zo[o][i]); }
        else lamz = fmin(lamz, ps_lammin_tol(no, Zo[o], work, 1e-12 * (1.0 + O->nC1)));
    }
    if (Xo) {
        double *rp = pcalloc(mo + 1, sizeof(double));
        for (size_t q = 0; q < O->nnz; q++) {
            const int o = O->blk[q], i = O->ii[q], j = O->jj[q], no = abs(O->bs[o]);
            const double x = O->bs[o] < 0 ? Xo[o][i] : Xo[o][i + (size_t)j * no];
            const double c = (O->bs[o] > 0 && i != j) ? 2.0 * O->v[q] * x : O->v[q] * x;
            if (O->con[q] < 0) pobj += c; else rp[O->con[q]] += c;
        }
        for (int o = 0; o < nbo; o++) {
            const int no = abs(O->bs[o]);
            const size_t len = bsize(O->bs[o]);
            for (size_t q = 0; q < len; q++) xz += Xo[o][q] * Zo[o][q];
            if (O->bs[o] < 0) { for (int i = 0; i < no; i++) lamx = fmin(lamx, Xo[o][i]); }
            else lamx = fmin(lamx, ps_lammin_tol(no, Xo[o], work, 1e-12 * (1.0 + O->nb1)));
        }
        double r2 = 0;
        for (int i = 0; i < mo; i++) {
            double v = rp[i] - O->b[i];
            if (O->tbR != 0 && O->b[i] == O->tbR) v *= (1.0 + O->nb1) / O->tbR;
            r2 += v * v;
        }
        const double den = 1 + fabs(pobj + O->obj_off) + fabs(dobj + O->obj_off);
        res->err[1] = sqrt(r2) / (1 + O->nb1);
        res->err[2] = fmax(0.0, -lamx) / (1 + O->nb1);
        res->err[5] = (pobj - dobj) / den;
        res->err[6] = xz / den;
        res->have_x = 1;
        free(rp);
    }
    res->err[3] = 0;
    res->err[4] = fmax(0.0, -lamz) / (1 + O->nC1);
    res->pobj = pobj; res->dobj = dobj;
    free(work);
    for (int o = 0; o < nbo; o++) free(Zo[o]);
    free(Zo);
    res->t = ps_now() - t0;
    return 0;
}
