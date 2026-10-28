/* socp.c - BRISK's solver for linear and second-order cone programs (5.4).
 *
 * Independent of the semidefinite solver: its own data structures, linear algebra and iteration.
 *
 *   (P)  min c'x   s.t.  A x = b,  x in K          (D)  max b'y   s.t.  z = c - A'y in K*
 *   K = R^nf (free) x R^nl_+ x Q^{q_1} x ... x Q^{q_nq} x Qr^{r_1} x ... (rotated cones)
 *
 * the SeDuMi form with K.f, K.l, K.q, K.r. A is given by columns (CSC).
 *
 * Method (IMPLEMENTATION.md, section 25):
 *  - presolve: two nonnegative columns (a, -a) with costs (c, -c) are one free variable; rotated
 *    cones are turned into second-order cones by an orthogonal map of two columns;
 *  - Ruiz equilibration of A with one column scale per cone, norm scaling of b and c;
 *  - the simplified homogeneous self-dual model (x, y, z, tau, kappa): optimal solutions and
 *    infeasibility certificates from one iteration; a least-squares starting point;
 *  - Nesterov-Todd scaling, Mehrotra predictor-corrector with up to three centrality correctors,
 *    one step length (0.99 of the way to the boundary), steps on the equations alone at the end;
 *  - the reduced system  [ -H  A' ; A  0 ]  with a static regularization (1e-9 on the cone and dual
 *    blocks, 1e-6 on the free variables) is quasi-definite and is factored as L D L' in a
 *    fill-reducing order fixed once (AMD, variants around dense rows and free variables, the least
 *    work kept); the scaling matrix of a large cone is not formed: H = eta^2 (D + u u' - v v')
 *    enters through two extra unknowns per cone, so that a cone of dimension q costs O(q) nonzeros;
 *  - two factorizations with the same order: by columns for small and very sparse factors,
 *    supernodal (sparsechol.c) otherwise; the pivots' signs are known in advance and a pivot of the
 *    wrong sign is replaced; with replaced pivots and a solve that does not refine, the column
 *    factorization is repeated in double-double arithmetic (small problems) or the regularization
 *    of the factored matrix is raised (large ones);
 *  - with no (or a few) free variables, small cones and no dense column: the normal equations
 *    A H^-1 A' + delta I by sparse Cholesky when they cost less, for as long as they are accurate;
 *  - every solve is GMRES (at most 10 steps) on the system with H exact and the base dual
 *    regularization only, preconditioned by the factorization;
 *  - the direction is refined against the dual equation when the rounding of H dx shows there;
 *    a solve that stalls short of the tolerance is repeated with shorter steps;
 *  - termination on the errors of the problem as given (after undoing the scaling), x'z included.
 */
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#if defined(__x86_64__) || defined(__i386__)
#include <xmmintrin.h>
#include <pmmintrin.h>
#endif
#include <time.h>
#include "socp.h"
#include "lpsolve.h"
#ifdef _OPENMP
#include <omp.h>
#endif   /* without OpenMP the runtime's queries come from omp/ompstub.h (ISSUES 713) */
extern double brisk_mem_scale(void);   /* (main.c) max(1, memory / 8 GB) */
#include "amd/amd.h"
#include "blas_one.h"
#ifndef BLAS_PREFIX
#define BLAS_PREFIX
#endif
#define SC_CAT2(a, b) a##b
#define SC_CAT(a, b)  SC_CAT2(a, b)
#define SC_BL(name) SC_CAT(BLAS_PREFIX, name)
void SC_BL(dsyrk_)(const char *, const char *, const int *, const int *, const double *, const double *, const int *, const double *, double *, const int *);
void SC_BL(dgemm_)(const char *, const char *, const int *, const int *, const int *, const double *, const double *, const int *, const double *, const int *, const double *, double *, const int *);

/* the supernodal sparse factorization of quasi-definite matrices (sparsechol.c): L S L' with the
 * signs S prescribed, fill-reducing orders, BLAS panels, threaded over the elimination tree */
typedef struct SChol SChol;
SChol *schol_analyze_adj(int m, int *deg, int **nbr, size_t fillcap);
double schol_set_amalg(double f);
/* (21 October 2026, LP/SOCP) the relaxed amalgamation of the sparse factor's supernodes for the SOCP solver's matrices
 * (the full system and the normal equations), as the LP method has used since 5.7: a supernode merged into its parent
 * when the explicit zeros stay below 30 % of the merged panel (64 columns at most). The exact supernodes of a full
 * system are narrow (nonnegative least squares on a 120 x 120 image: 38,465 nodes for 43,205 unknowns), so the
 * factorization ran column by column at about 5 GF/s. BRISK_CONEAMALG=f: the fraction for every factor (0: exact supernodes). */
static double cone_amalg(double per_unknown) {
    static double v = -2; if (v < -1) { const char *e_ = getenv("BRISK_CONEAMALG"); v = e_ ? fmax(0.0, atof(e_)) : -1; }
    if (v >= 0) return v;
    /* only when the columns of L are long (at least 100 entries per unknown): there the narrow nodes cost the dense
     * kernels' rate; where they are short (TV denoising: 9 entries, DIMACS nql180: 30) the merge saves little and the
     * changed rounding moved knife-edge end phases (tv_300 17 -> 43, nql180 28 -> 49 iterations at 0.3 for all) */
    return per_unknown >= 100.0 ? 0.3 : 0.0;
}
void   schol_free(SChol *S);
double *schol_values(SChol *S);
size_t schol_offset(const SChol *S, size_t r, size_t c);
void   schol_set_signs(SChol *S, const signed char *sgn_orig);
void   schol_set_tinypiv(SChol *S, double t);
void   schol_set_tinyreg(double eps);
void   schol_set_tinyflags(SChol *S, int on);
int    schol_tiny_rows(const SChol *S, int *rows, int kmax);
int    schol_factor(SChol *S, double shift);
void   schol_solve(const SChol *S, double *b, double *work);
size_t schol_nnz(const SChol *S);
int    schol_set_amd(int mode);
const int *schol_perm(const SChol *S);
void   schol_set_amd_dense(double d);
void   schol_set_solve_seq(int on);
void   schol_set_perm(const int *perm);
void   schol_zero(SChol *S);
void   schol_set_inplace(SChol *S, int on);
int    schol_solve_part(SChol *S, double *b, double *work, int nt);
/* (23 October 2026, the SOCP developer) the supernodal solve by independent subtrees (sparsechol.c schol_solve_part: the same
 * bits at any thread count); BRISK_CONESOLVEPART=1 */
/* (23 October 2026, the SOCP developer) the triangular solves of the supernodal factor by independent subtrees
 * (schol_solve_part), by default: the same bits at any thread count, one thread inside an outer parallel region; on the
 * 47 SOCP files where the partition applies (the top of the tree at most half of the solve's work) 0.94 of the total time
 * at two threads, 1.007 at one. BRISK_CONESOLVEPART=0: the sequential solve */
static int cone_part(void) { static int v = -1; if (v < 0) { const char *e_ = getenv("BRISK_CONESOLVEPART"); v = e_ ? atoi(e_) : 1; } return v; }
double schol_flops(const SChol *S);
int    schol_ntiny(const SChol *S);

/* printing and interruption are the caller's: socp_printf (default: printf), *socp_stop (default: never) */
int (*socp_printf)(const char *fmt, ...) = printf;
void (*socp_note)(const char *fmt, ...) = NULL;    /* 5.8: a line for the caller's summary at the end of the run (brisk_note in main.c) */
#define SNOTE(...) do { if (socp_note) socp_note(__VA_ARGS__); } while (0)
static _Thread_local const char *g_stat_prefix = "";     /* (19 October 2026: "the dual: " while the dual form is solved) */
static volatile int socp_never = 0;
volatile int *socp_stop = &socp_never;
volatile int *socp_stop_never(void) { return &socp_never; }     /* (28 October 2026: the default flag, so that a caller's own flag is not replaced) */
/* (28 October 2026, the night) a second flag, the calling thread's own for its solves (NULL: none): the LP route's fallback runs
 * beside the dual simplex method, which raises it when it has its result (neos-5041822-cockle, LP: the SOCP solver ran 3.4 s
 * past the simplex method's result, whose run it then returned) */
_Thread_local volatile int *socp_stop_call = NULL;
static double sc_time(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + 1e-9 * t.tv_nsec; }

static _Thread_local double *g_ract = NULL; static _Thread_local int g_ract_m = 0;     /* (the rows' activities for the row-wise primal error; per thread: SOCP solves on several threads at once, ISSUES 291) */
static int dbg_on(void) { static int v = -1; if (v < 0) { const char *e = getenv("BRISK_CONEDBG"); v = e ? (atoi(e) > 1 ? atoi(e) : 1) : 0; } return v; }     /* (2: also the dual residual by kind) */

#define NE_MAXFREE 8
/* (20 October 2026, LP/SOCP, a test: BRISK_CONENEMAXFREE=k raises the number of free variables the normal equations take by
 * their dense Schur complement (k solves per factorization, an m x k work array); the FIR filters have 151 dense free
 * columns and their full system's factor has 40 million entries where the normal equations' has 4 million) */
static int ne_maxfree(void) { static int v = -1; if (v < 0) { const char *e_ = getenv("BRISK_CONENEMAXFREE"); v = e_ ? atoi(e_) : NE_MAXFREE; if (v < 1) v = NE_MAXFREE; } return v; }
/* (20 October 2026, LP/SOCP) the normal equations' free columns are bounded by the cost rule of ne_setup (their solves and
 * their Schur complement against the full system's factorization) and the memory of W, not by NE_MAXFREE (the dual route
 * keeps it); this is the hard cap. CBLIB's FIR filters: 151 free columns over 14,100 rows. */
#define NE_MAXFREE_NE 1024
#define NE_TINYMAX 32           /* (the tiny pivots of the normal equations corrected exactly through the Woodbury update, at most) */
#define NE_MAXWOODRANK 256       /* (the rank of the Woodbury update: the sparse cones' columns and the dense blocks') */
static int wood_on(void) { static int v = -1; if (v < 0) { const char *e_ = getenv("BRISK_CONEWOOD"); v = e_ ? atoi(e_) : 0; } return v; }   /* (a test, off: the dense blocks by the Woodbury update and the free columns by the cost rule; TRICKS 8) */
static int ne_maxfree_ne(void) { static int v = -1; if (v < 0) { const char *e_ = getenv("BRISK_CONENEMAXFREE"); v = e_ ? atoi(e_) : (wood_on() ? NE_MAXFREE_NE : NE_MAXFREE); if (v < 1) v = NE_MAXFREE; } return v; }
#define SOCP_DENSE_MAX 4          /* cones up to this dimension: H as a dense block; larger: the sparse form */

/* ------------------------------------------------------------------ sparse L D L' (up-looking) */
typedef struct {
    int n;
    int *Ap, *Ai; double *Ax;     /* the permuted matrix, upper triangle by columns, diagonal last in each column */
    int *etree, *Lnz, *Lp, *Li; double *Lx, *D, *Dinv;
    int *sign;                    /* expected sign of each pivot (in the permuted order) */
    int *iw; unsigned char *bw; double *fw;
    int *perm, *iperm;
    double *Lxl, *Dl, *Dinvl, *fwl;   /* double-double factorization: the low parts */
    int *lvp, *lvn, nlv; int *Tp, *Ti; double *Tx, *Txl;   /* the level sets of the elimination tree and L by rows, for the solves on the threads */
    int tr_ok;                        /* the transpose holds the current factor's values */
    int dd;                           /* the current factor is the double-double one */
    long cap;                         /* ldl_symbolic gives up beyond this many entries (0: no bound) */
    double dyn_delta; int ndyn; int pivdbg, pivnf, pivn, pivnaux; double pivmin[4]; long pivcnt[4];
    long lnz;
    /* (27 October 2026) the elimination tree in LDL_NPART parts and a top (ldl_part_build): part q's nodes, ascending, in
     * pnl[pnp[q] .. pnp[q+1]); the top's in ptop[0 .. ntop); ptpos: a node's index in the top or -1; ptbuf, ptbufl: the forward
     * solve's contributions of each part to the top rows; the threads' workspaces beyond the first */
    int pt_built, npart, ntop, *pnp, *pnl, *ptop, *ptpos, *ptsplit, pt_fac, pt_sol; double *ptbuf, *ptbufl;
    int nws; int *wsi[16]; unsigned char *wsb[16]; double *wsf[16], *wsfl[16];
} Ldl;

static void ldl_part_free(Ldl *F);
static int ldl_symbolic(Ldl *F) {
    const int n = F->n;
    int *work = F->iw;
    long tot = 0;
    ldl_part_free(F);
    for (int i = 0; i < n; i++) { work[i] = 0; F->Lnz[i] = 0; F->etree[i] = -1; }
    for (int j = 0; j < n; j++) {
        work[j] = j;
        for (int p = F->Ap[j]; p < F->Ap[j + 1]; p++) {
            int i = F->Ai[p];
            if (i > j) return -1;
            while (work[i] != j) {
                if (F->etree[i] == -1) F->etree[i] = j;
                F->Lnz[i]++; tot++;
                work[i] = j;
                i = F->etree[i];
            }
        }
        if (F->cap > 0 && tot > F->cap) return 2;         /* more fill than the caller's bound: given up */
    }
    F->Lp[0] = 0;
    for (int i = 0; i < n; i++) F->Lp[i + 1] = F->Lp[i] + F->Lnz[i];
    F->lnz = F->Lp[n];
    /* (27 October 2026, the SOCP developer, a measurement: BRISK_CONETREEDBG=p) how much of the factorization's work lies in
     * subtrees of the elimination tree that p threads could take (Geist and Ng's subtree-to-thread mapping): the work of
     * column j about Lnz_j^2, the largest subtree split into its children while the p bins are unbalanced; the rest, the
     * top of the tree, is sequential */
    { static int tp = -1; if (tp < 0) { const char *e_ = getenv("BRISK_CONETREEDBG"); tp = e_ ? atoi(e_) : 0; }
      if (tp > 1 && n > 0) {
        double *sw = (double *)calloc((size_t)n, sizeof(double)); int *head = (int *)malloc(sizeof(int) * (size_t)n), *nxt = (int *)malloc(sizeof(int) * (size_t)n);
        for (int j = 0; j < n; j++) head[j] = -1;
        double W = 0; for (int j = 0; j < n; j++) { const double w = (double)F->Lnz[j] * F->Lnz[j] + F->Lnz[j] + 1; sw[j] += w; W += w; if (F->etree[j] >= 0) sw[F->etree[j]] += sw[j]; }
        for (int j = n - 1; j >= 0; j--) if (F->etree[j] >= 0) { nxt[j] = head[F->etree[j]]; head[F->etree[j]] = j; }
        int cap = 1024, nl = 0, *L = (int *)malloc(sizeof(int) * (size_t)cap);
        for (int j = 0; j < n; j++) if (F->etree[j] < 0) { if (nl == cap) { cap *= 2; L = (int *)realloc(L, sizeof(int) * (size_t)cap); } L[nl++] = j; }
        double top = 0, best_ms = 1e300, best_top = 0; int iters = 0;
        for (;;) {
            /* the bins: the subtrees in L by size, each to the least loaded bin */
            double bins[64] = { 0 }; const int pb = tp < 64 ? tp : 64;
            /* (the largest first: a partial selection is enough for the measurement) */
            int imax = -1; for (int t = 0; t < nl; t++) if (imax < 0 || sw[L[t]] > sw[L[imax]]) imax = t;
            double tot = 0; for (int t = 0; t < nl; t++) tot += sw[L[t]];
            /* greedy packing in the list's order after the largest: an upper bound of the makespan */
            for (int t = 0; t < nl; t++) { int b = 0; for (int q = 1; q < pb; q++) if (bins[q] < bins[b]) b = q; bins[b] += sw[L[t]]; }
            double ms = 0; for (int q = 0; q < pb; q++) ms = fmax(ms, bins[q]);
            if (ms + top < best_ms) { best_ms = ms + top; best_top = top; }
            if (imax < 0 || ++iters > 200000 || head[L[imax]] < 0 || sw[L[imax]] <= 1.05 * tot / pb) break;
            const int sj = L[imax]; double cs = 0; for (int c = head[sj]; c >= 0; c = nxt[c]) cs += sw[c];
            top += sw[sj] - cs; L[imax] = L[--nl];
            for (int c = head[sj]; c >= 0; c = nxt[c]) { if (nl == cap) { cap *= 2; L = (int *)realloc(L, sizeof(int) * (size_t)cap); } L[nl++] = c; }
        }
        socp_printf("  [elimination tree on %d threads: work %.2e, the sequential top %.1f %%, the best time %.1f %% of one thread's (%d subtrees)]\n", tp, W, 100.0 * best_top / W, 100.0 * best_ms / W, nl);
        free(sw); free(head); free(nxt); free(L);
      } }
    return 0;
}

/* numeric factorization; returns the number of pivots that were replaced */
/* (23 October 2026, the SOCP developer) the column factorization's storage, allocated at its first use: next to the
 * supernodal factor it serves only the fallback and the double-double end phase, and allocated with the analysis it was a
 * reserve of the factor's size (socp_svmsparse_200kx20k (SOCP): 2.4 GB against the 10 GB cap of the comparison) */
static void ldl_alloc(Ldl *F) {
    if (F->Li) return;
    F->Li = (int *)malloc(sizeof(int) * (size_t)(F->lnz + 1)); F->Lx = (double *)malloc(sizeof(double) * (size_t)(F->lnz + 1));
    if (!F->Li || !F->Lx) { fprintf(stderr, "brisk: out of memory (the column factorization, %.1f GB)\n", 12.0 * (double)F->lnz / 1e9); exit(1); }
}
/* (27 October 2026, the SOCP developer) the column factorization and its solves by subtrees of the elimination tree on the
 * threads. Row k of L reads the columns of its descendants and writes column c's entry for each such c; the rows of two
 * disjoint subtrees therefore touch disjoint columns, and each column still receives its entries in increasing row order
 * (its own subtree's rows, then the top's, which are its ancestors): the factor is the same bit for bit whatever the
 * threads. The tree is cut by Geist and Ng's mapping (the largest subtree split into its children while it outweighs a
 * part's share; the split nodes form the top) into LDL_NPART parts by the largest-first rule, a partition that does not
 * depend on the number of threads, so the solves, which sum the parts' contributions to the top rows in part order, give
 * the same bits on any number of threads too. The structural-design SOCPs, factored by columns throughout (joint_FC_5:
 * 104 of 144 s in its factorizations and solves): about 9 % of the work in the top for two threads; 135 -> 102 s at two
 * threads, the same iterations; the 39 mezo SOCPs: the same bits, time 0.974 (db-shear-wall 5.9 -> 4.5 s). BRISK_CONEPARTREE=0:
 * the sequential factorization and solves (the same bits) */
#define LDL_NPART 8
static void ldl_part_free(Ldl *F) {
    free(F->pnp); free(F->pnl); free(F->ptop); free(F->ptpos); free(F->ptsplit); free(F->ptbuf); free(F->ptbufl);
    F->pnp = F->pnl = F->ptop = F->ptpos = F->ptsplit = NULL; F->ptbuf = F->ptbufl = NULL; F->npart = 0; F->ntop = 0; F->pt_built = 0;
    for (int t = 0; t < F->nws; t++) { free(F->wsi[t]); free(F->wsb[t]); free(F->wsf[t]); free(F->wsfl[t]); F->wsi[t] = NULL; F->wsb[t] = NULL; F->wsf[t] = F->wsfl[t] = NULL; }
    F->nws = 0;
}
static void ldl_part_build(Ldl *F) {
    const int n = F->n;
    F->pt_built = 1; F->npart = 0;
    static int on = -1; if (on < 0) { const char *e_ = getenv("BRISK_CONEPARTREE"); on = e_ ? atoi(e_) : 1; }
    if (!on || n < 20000) return;
    double *sw = (double *)calloc((size_t)n, sizeof(double)); int *head = (int *)malloc(sizeof(int) * (size_t)n), *nxt = (int *)malloc(sizeof(int) * (size_t)n);
    int *rbin = (int *)malloc(sizeof(int) * (size_t)n);
    if (!sw || !head || !nxt || !rbin) { free(sw); free(head); free(nxt); free(rbin); return; }
    for (int j = 0; j < n; j++) { head[j] = -1; rbin[j] = -1; }
    double W = 0; for (int j = 0; j < n; j++) { const double w = (double)F->Lnz[j] * F->Lnz[j] + F->Lnz[j] + 1; sw[j] += w; W += w; if (F->etree[j] >= 0) sw[F->etree[j]] += sw[j]; }
    for (int j = n - 1; j >= 0; j--) if (F->etree[j] >= 0) { nxt[j] = head[F->etree[j]]; head[F->etree[j]] = j; }
    /* the subtrees by a max-heap on their work (each split takes one node into the top: at most n splits) */
    int nl = 0, *L = (int *)malloc(sizeof(int) * ((size_t)n + 1));
    double WL = 0;
    #define LDL_HUP(i_) do { int c_ = (i_); while (c_ > 0) { const int p_ = (c_ - 1) / 2; if (sw[L[p_]] > sw[L[c_]] || (sw[L[p_]] == sw[L[c_]] && L[p_] < L[c_])) break; const int t_ = L[p_]; L[p_] = L[c_]; L[c_] = t_; c_ = p_; } } while (0)
    #define LDL_HDOWN(i_) do { int c_ = (i_); for (;;) { int l_ = 2 * c_ + 1, b_ = c_; if (l_ < nl && (sw[L[l_]] > sw[L[b_]] || (sw[L[l_]] == sw[L[b_]] && L[l_] < L[b_]))) b_ = l_; if (l_ + 1 < nl && (sw[L[l_ + 1]] > sw[L[b_]] || (sw[L[l_ + 1]] == sw[L[b_]] && L[l_ + 1] < L[b_]))) b_ = l_ + 1; if (b_ == c_) break; const int t_ = L[b_]; L[b_] = L[c_]; L[c_] = t_; c_ = b_; } } while (0)
    for (int j = 0; j < n; j++) if (F->etree[j] < 0) { L[nl++] = j; LDL_HUP(nl - 1); WL += sw[j]; }
    double top = 0;
    while (nl > 0) {
        const int sj = L[0];
        if (head[sj] < 0 || sw[sj] <= WL / LDL_NPART || top > 0.5 * W) break;
        double cs = 0; for (int c = head[sj]; c >= 0; c = nxt[c]) cs += sw[c];
        top += sw[sj] - cs; WL -= sw[sj] - cs;
        L[0] = L[--nl]; LDL_HDOWN(0);
        for (int c = head[sj]; c >= 0; c = nxt[c]) { L[nl++] = c; LDL_HUP(nl - 1); }
    }
    #undef LDL_HUP
    #undef LDL_HDOWN
    if (top <= 0.5 * W && nl >= 2) {
        /* the subtrees to the parts, the largest first, each to the least loaded part (ties: the lower index) */
        for (int a = 1; a < nl; a++) { const int v = L[a]; int b = a; while (b > 0 && (sw[L[b - 1]] < sw[v] || (sw[L[b - 1]] == sw[v] && L[b - 1] > v))) { L[b] = L[b - 1]; b--; } L[b] = v; }
        double ld[LDL_NPART] = { 0 };
        for (int t = 0; t < nl; t++) { int b = 0; for (int q = 1; q < LDL_NPART; q++) if (ld[q] < ld[b]) b = q; ld[b] += sw[L[t]]; rbin[L[t]] = b; }
        int *pn = (int *)malloc(sizeof(int) * (size_t)n);
        for (int j = n - 1; j >= 0; j--) pn[j] = rbin[j] >= 0 ? rbin[j] : (F->etree[j] >= 0 && pn[F->etree[j]] >= 0 ? pn[F->etree[j]] : -1);
        F->pnp = (int *)calloc(LDL_NPART + 1, sizeof(int)); F->pnl = (int *)malloc(sizeof(int) * (size_t)n); F->ptpos = (int *)malloc(sizeof(int) * (size_t)n);
        int ntop = 0; for (int j = 0; j < n; j++) { if (pn[j] >= 0) F->pnp[pn[j] + 1]++; else ntop++; }
        for (int q = 0; q < LDL_NPART; q++) F->pnp[q + 1] += F->pnp[q];
        F->ptop = (int *)malloc(sizeof(int) * ((size_t)ntop + 1));
        int fill[LDL_NPART]; for (int q = 0; q < LDL_NPART; q++) fill[q] = F->pnp[q];
        int nt_ = 0; for (int j = 0; j < n; j++) { if (pn[j] >= 0) { F->pnl[fill[pn[j]]++] = j; F->ptpos[j] = -1; } else { F->ptpos[j] = nt_; F->ptop[nt_++] = j; } }
        F->ntop = ntop;
        F->npart = LDL_NPART;
        free(pn);
        if (getenv("BRISK_CONEDBG")) socp_printf("  [the column factorization in %d parts and a top of %d nodes: %.1f %% of the work in the top, the largest part %.1f %%]\n", LDL_NPART, ntop, 100.0 * top / W, 100.0 * ld[0] / W > 0 ? 100.0 * fmax(fmax(fmax(ld[0], ld[1]), fmax(ld[2], ld[3])), fmax(fmax(ld[4], ld[5]), fmax(ld[6], ld[7]))) / W : 0.0);
    }
    free(sw); free(head); free(nxt); free(rbin); free(L);
}
/* each column's first entry in a top row (the entries of a column are in increasing row order, and its top rows, the
 * ancestors above its part's subtree, come last) */
static void ldl_part_split(Ldl *F) {
    if (F->npart <= 0 || F->ptsplit) return;      /* (once per analysis: the pattern is the same at every factorization) */
    F->ptsplit = (int *)malloc(sizeof(int) * (size_t)F->n); if (!F->ptsplit) { F->npart = 0; return; }
    /* the shares the parts take, from the pattern (deterministic, the same on any number of threads): of the
     * factorization, row k's work is that of its entries L(k, c), each the entries of column c above it (its position);
     * of the forward solve, the entries of the parts' columns in their own rows; the parts are used for each only where
     * they take at least half (socp_enclose_20000x20, SOCP: 21 top rows that every column reaches, the solves 0.59 ->
     * 0.98 s by the parts) */
    double fw_all = 0, fw_top = 0, sv_all = 0, sv_par = 0;
    for (int j = 0; j < F->n; j++) {
        int p = F->Lp[j];
        if (F->ptpos[j] < 0) while (p < F->Lp[j + 1] && F->ptpos[F->Li[p]] < 0) p++;
        F->ptsplit[j] = p;
        for (int q = F->Lp[j]; q < F->Lp[j + 1]; q++) { const double w = (double)(q - F->Lp[j]) + 1.0; fw_all += w; if (F->ptpos[F->Li[q]] >= 0) fw_top += w; }
        sv_all += F->Lp[j + 1] - F->Lp[j]; sv_par += p - F->Lp[j];
    }
    F->pt_fac = fw_all > 0 && fw_top <= 0.85 * fw_all;     /* (the factorization's parts cost nothing extra: joint_FC_5, 72.6 % in the top, 31.6 -> 25.0 s) */ F->pt_sol = sv_all > 0 && sv_par >= 0.5 * sv_all;
    if (getenv("BRISK_CONEDBG")) socp_printf("  [the column factorization's parts: %.1f %% of its work in the top rows%s, %.1f %% of the forward solve in the parts%s]\n", 100.0 * fw_top / fmax(fw_all, 1.0), F->pt_fac ? "" : " (the factorization sequential)", 100.0 * sv_par / fmax(sv_all, 1.0), F->pt_sol ? "" : " (the solves sequential)");
}
/* the threads for the parts: the solve's own (g_cone_nt), at most the parts; one inside a parallel region */
static _Thread_local int g_cone_nt_ldl = 1;
static int ldl_part_nt(Ldl *F) {
    int nt = g_cone_nt_ldl; if (nt > LDL_NPART) nt = LDL_NPART; if (nt > 16) nt = 16;
    if (nt < 1 || omp_in_parallel()) nt = 1;
    const int n = F->n;
    while (F->nws < nt - 1) {      /* (the threads' workspaces beyond the first, zero: each row leaves them zero) */
        const int t = F->nws;
        F->wsi[t] = (int *)malloc(sizeof(int) * 2 * (size_t)n); F->wsb[t] = (unsigned char *)calloc((size_t)n, 1);
        F->wsf[t] = (double *)calloc((size_t)n, sizeof(double)); F->wsfl[t] = (double *)calloc((size_t)n, sizeof(double));
        if (!F->wsi[t] || !F->wsb[t] || !F->wsf[t] || !F->wsfl[t]) { free(F->wsi[t]); free(F->wsb[t]); free(F->wsf[t]); free(F->wsfl[t]); F->wsi[t] = NULL; F->wsb[t] = NULL; F->wsf[t] = F->wsfl[t] = NULL; break; }
        F->nws++;
    }
    if (nt - 1 > F->nws) nt = F->nws + 1;
    return nt;
}
/* row k of the factorization in double (the body of the sequential loop) */
static inline int ldl_row(Ldl *F, int k, int *yIdx, int *elim, int *next, unsigned char *mark, double *y) {
    int nny = 0, ndyn = 0;
    double dk = 0.0;
    for (int p = F->Ap[k]; p < F->Ap[k + 1]; p++) {
        int i = F->Ai[p];
        if (i == k) { dk = F->Ax[p]; continue; }
        y[i] = F->Ax[p];
        if (!mark[i]) {
            int ne = 0, j = i;
            mark[j] = 1; elim[ne++] = j;
            j = F->etree[j];
            while (j != -1 && j < k && !mark[j]) { mark[j] = 1; elim[ne++] = j; j = F->etree[j]; }
            while (ne) yIdx[nny++] = elim[--ne];
        }
    }
    for (int t = nny - 1; t >= 0; t--) {
        const int c = yIdx[t];
        const double yc = y[c];
        const int pe = next[c];
        for (int p = F->Lp[c]; p < pe; p++) y[F->Li[p]] -= F->Lx[p] * yc;
        const double l = yc * F->Dinv[c];
        dk -= yc * l;
        F->Li[pe] = k; F->Lx[pe] = l; next[c]++;
        y[c] = 0.0; mark[c] = 0;
    }
    if (!(F->sign[k] * dk > 0)) {   /* (BRISK_CONEPIVDBG=1: each replaced pivot printed, 22 October 2026) */
        if (getenv("BRISK_CONEPIVDBG")) { const int o = F->perm[k]; double d0 = 0; for (int p = F->Ap[k]; p < F->Ap[k + 1]; p++) if (F->Ai[p] == k) d0 = F->Ax[p];
            socp_printf("        [pivot %d (original %d, %s, sign %+d): %.3e, its diagonal %.3e]\n", k, o, o < F->pivnf ? "free" : o < F->pivn ? "cone" : o < F->pivn + F->pivnaux ? "aux" : (F->sign[k] < 0 ? "linear" : "row"), F->sign[k], dk, d0); }
        dk = F->sign[k] * F->dyn_delta; ndyn++; }       /* a pivot of the wrong sign: replaced (the caller turns to double-double) */
    F->D[k] = dk; F->Dinv[k] = 1.0 / dk;
    if (F->pivdbg) { const int o = F->perm[k]; const int cl = o < F->pivnf ? 0 : o < F->pivn ? 1 : o < F->pivn + F->pivnaux ? 2 : 3; if (fabs(dk) < F->pivmin[cl]) F->pivmin[cl] = fabs(dk); if (fabs(dk) < 1e-6) F->pivcnt[cl]++; }
    return ndyn;
}
static int ldl_numeric(Ldl *F) {
    const int n = F->n;
    ldl_alloc(F);
    int *yIdx = F->iw, *elim = F->iw + n, *next = F->iw + 2 * n;
    unsigned char *mark = F->bw;
    double *y = F->fw;
    F->ndyn = 0;
    for (int i = 0; i < n; i++) { mark[i] = 0; y[i] = 0.0; next[i] = F->Lp[i]; }
    if (!F->pt_built) ldl_part_build(F);
    const int nt = F->npart > 0 && !F->pivdbg && F->ptsplit && F->pt_fac ? ldl_part_nt(F) : 1;     /* (the first factorization sequential: the shares need its pattern) */
    if (nt > 1) {
        int nd[LDL_NPART] = { 0 };
        #pragma omp parallel for schedule(dynamic, 1) num_threads(nt)
        for (int q = 0; q < F->npart; q++) {
            const int tid = omp_get_thread_num();
            int *yI = tid ? F->wsi[tid - 1] : yIdx, *el = tid ? F->wsi[tid - 1] + n : elim; unsigned char *mk = tid ? F->wsb[tid - 1] : mark; double *yy = tid ? F->wsf[tid - 1] : y;
            for (int a = F->pnp[q]; a < F->pnp[q + 1]; a++) nd[q] += ldl_row(F, F->pnl[a], yI, el, next, mk, yy);
        }
        for (int q = 0; q < F->npart; q++) F->ndyn += nd[q];
        for (int a = 0; a < F->ntop; a++) F->ndyn += ldl_row(F, F->ptop[a], yIdx, elim, next, mark, y);
    } else
        for (int k = 0; k < n; k++) F->ndyn += ldl_row(F, k, yIdx, elim, next, mark, y);
    ldl_part_split(F);
    return F->ndyn;
}

/* ---- the same factorization in double-double arithmetic (about 32 digits): used when the
 * double one meets pivots of the wrong sign or its solves stop refining, near the end of a
 * solve on a badly conditioned problem. The matrix is the double one. */
typedef struct { double h, l; } sdd;
static inline sdd dd_fast(double a, double b) { sdd r; r.h = a + b; r.l = b - (r.h - a); return r; }
static inline sdd dd_mul(sdd a, sdd b) {
    const double p = a.h * b.h;
    const double e = fma(a.h, b.h, -p) + (a.h * b.l + a.l * b.h);
    return dd_fast(p, e);
}
static inline sdd dd_sub(sdd a, sdd b) {
    const double s = a.h - b.h, bb = s - a.h;
    const double e = (a.h - (s - bb)) + (-b.h - bb) + (a.l - b.l);
    return dd_fast(s, e);
}
static inline sdd dd_inv(sdd d) {
    const double q1 = 1.0 / d.h;
    sdd one = { 1.0, 0.0 }, q = { q1, 0.0 };
    const sdd r = dd_sub(one, dd_mul(q, d));
    return dd_fast(q1, r.h / d.h);
}
/* row k of the factorization in double-double (the body of the sequential loop) */
static inline int ldl_row_dd(Ldl *F, int k, int *yIdx, int *elim, int *next, unsigned char *mark, double *y, double *yl) {
    int ndyn = 0;
    int nny = 0;
    sdd dk = { 0.0, 0.0 };
    for (int p = F->Ap[k]; p < F->Ap[k + 1]; p++) {
        int i = F->Ai[p];
        if (i == k) { dk.h = F->Ax[p]; continue; }
        y[i] = F->Ax[p]; yl[i] = 0.0;
        if (!mark[i]) {
            int ne = 0, j = i;
            mark[j] = 1; elim[ne++] = j;
            j = F->etree[j];
            while (j != -1 && j < k && !mark[j]) { mark[j] = 1; elim[ne++] = j; j = F->etree[j]; }
            while (ne) yIdx[nny++] = elim[--ne];
        }
    }
    for (int t = nny - 1; t >= 0; t--) {
        const int c = yIdx[t];
        const sdd yc = { y[c], yl[c] };
        const int pe = next[c];
        for (int p = F->Lp[c]; p < pe; p++) {
            const int r = F->Li[p];
            const sdd lv = { F->Lx[p], F->Lxl[p] }, yr = { y[r], yl[r] };
            const sdd v = dd_sub(yr, dd_mul(lv, yc));
            y[r] = v.h; yl[r] = v.l;
        }
        const sdd di = { F->Dinv[c], F->Dinvl[c] };
        const sdd l = dd_mul(yc, di);
        dk = dd_sub(dk, dd_mul(yc, l));
        F->Li[pe] = k; F->Lx[pe] = l.h; F->Lxl[pe] = l.l; next[c]++;
        y[c] = 0.0; yl[c] = 0.0; mark[c] = 0;
    }
    if (!(F->sign[k] * dk.h > 0)) {
        if (getenv("BRISK_CONEPIVDBG")) { const int o = F->perm[k]; double d0 = 0; for (int p = F->Ap[k]; p < F->Ap[k + 1]; p++) if (F->Ai[p] == k) d0 = F->Ax[p];
            socp_printf("        [dd pivot %d (original %d, %s, sign %+d): %.3e, its diagonal %.3e]\n", k, o, o < F->pivnf ? "free" : o < F->pivn ? "cone" : o < F->pivn + F->pivnaux ? "aux" : (F->sign[k] < 0 ? "linear" : "row"), F->sign[k], dk.h, d0); }
        dk.h = F->sign[k] * F->dyn_delta; dk.l = 0.0; ndyn++; }
    const sdd di = dd_inv(dk);
    F->D[k] = dk.h; F->Dl[k] = dk.l; F->Dinv[k] = di.h; F->Dinvl[k] = di.l;
    return ndyn;
}
static int ldl_numeric_dd(Ldl *F) {
    const int n = F->n;
    ldl_alloc(F);
    if (!F->Lxl) {
        F->Lxl = (double *)malloc(sizeof(double) * (size_t)(F->lnz + 1)); F->Dl = (double *)malloc(sizeof(double) * (size_t)n);
        F->Dinvl = (double *)malloc(sizeof(double) * (size_t)n); F->fwl = (double *)malloc(sizeof(double) * (size_t)n);
    }
    int *yIdx = F->iw, *elim = F->iw + n, *next = F->iw + 2 * n;
    unsigned char *mark = F->bw;
    double *y = F->fw, *yl = F->fwl;
    F->ndyn = 0;
    for (int i = 0; i < n; i++) { mark[i] = 0; y[i] = 0.0; yl[i] = 0.0; next[i] = F->Lp[i]; }
    if (!F->pt_built) ldl_part_build(F);
    const int nt = F->npart > 0 && F->ptsplit && F->pt_fac ? ldl_part_nt(F) : 1;
    if (nt > 1) {
        int nd[LDL_NPART] = { 0 };
        #pragma omp parallel for schedule(dynamic, 1) num_threads(nt)
        for (int q = 0; q < F->npart; q++) {
            const int tid = omp_get_thread_num();
            int *yI = tid ? F->wsi[tid - 1] : yIdx, *el = tid ? F->wsi[tid - 1] + n : elim; unsigned char *mk = tid ? F->wsb[tid - 1] : mark;
            double *yy = tid ? F->wsf[tid - 1] : y, *yyl = tid ? F->wsfl[tid - 1] : yl;
            for (int a = F->pnp[q]; a < F->pnp[q + 1]; a++) nd[q] += ldl_row_dd(F, F->pnl[a], yI, el, next, mk, yy, yyl);
        }
        for (int q = 0; q < F->npart; q++) F->ndyn += nd[q];
        for (int a = 0; a < F->ntop; a++) F->ndyn += ldl_row_dd(F, F->ptop[a], yIdx, elim, next, mark, y, yl);
    } else
        for (int k = 0; k < n; k++) F->ndyn += ldl_row_dd(F, k, yIdx, elim, next, mark, y, yl);
    ldl_part_split(F);
    F->dd = 1; F->tr_ok = 0;
    return F->ndyn;
}
static void ldl_solve_perm_dd_part(const Ldl *F, double *x);
static void ldl_solve_perm_dd(const Ldl *F, double *x) {
    const int n = F->n;
    if (F->npart > 0 && F->ptsplit && F->pt_sol && g_cone_nt_ldl > 1 && !omp_in_parallel()) { ldl_solve_perm_dd_part(F, x); return; }
    double *xl = F->fwl;
    for (int j = 0; j < n; j++) xl[j] = 0.0;
    for (int j = 0; j < n; j++) {
        const sdd xj = { x[j], xl[j] };
        if (xj.h != 0.0 || xj.l != 0.0)
            for (int p = F->Lp[j]; p < F->Lp[j + 1]; p++) {
                const int r = F->Li[p];
                const sdd lv = { F->Lx[p], F->Lxl[p] }, xr = { x[r], xl[r] };
                const sdd v = dd_sub(xr, dd_mul(lv, xj));
                x[r] = v.h; xl[r] = v.l;
            }
    }
    for (int j = 0; j < n; j++) { const sdd xj = { x[j], xl[j] }, di = { F->Dinv[j], F->Dinvl[j] }; const sdd v = dd_mul(xj, di); x[j] = v.h; xl[j] = v.l; }
    for (int j = n - 1; j >= 0; j--) {
        sdd sacc = { x[j], xl[j] };
        for (int p = F->Lp[j]; p < F->Lp[j + 1]; p++) {
            const int r = F->Li[p];
            const sdd lv = { F->Lx[p], F->Lxl[p] }, xr = { x[r], xl[r] };
            sacc = dd_sub(sacc, dd_mul(lv, xr));
        }
        x[j] = sacc.h; xl[j] = sacc.l;
    }
}

/* (20 October 2026, LP/SOCP) the double-double solve on the threads: the rows of L (its transpose, built once per
 * factorization) gathered level by level of the elimination tree in the forward solve, the columns of L gathered level
 * by level from the top in the backward one; every unknown is finished by one thread with the same sequence of
 * operations as the sequential solve, so the bits are the same at any thread count. On joint_FC_7 the double-double
 * solves of the last twelve iterations were 2.2-4.4 s each of 4-7 s. Measured alone on the machine, four threads:
 * joint_FC_7's 1,360 solves 59.5 -> 55.7 s, db-shear-wall 1.57 -> 1.46 s, db-plane-strain-prism 10.6 -> 11.2 s: the
 * trees are tall (thousands of levels, most of them a few nodes), and a barrier per level eats the gain. Off by
 * default (BRISK_CONEDDPAR=1). */
static void ldl_levels(Ldl *F) {
    const int n = F->n;
    if (F->lvp) return;
    int *lv = (int *)calloc((size_t)n + 1, sizeof(int));
    int nlv = 0;
    for (int i = 0; i < n; i++) { const int p = F->etree[i]; if (p >= 0 && lv[i] + 1 > lv[p]) lv[p] = lv[i] + 1; if (lv[i] + 1 > nlv) nlv = lv[i] + 1; }
    F->nlv = nlv; F->lvp = (int *)calloc((size_t)nlv + 2, sizeof(int)); F->lvn = (int *)malloc(sizeof(int) * ((size_t)n + 1));
    for (int i = 0; i < n; i++) F->lvp[lv[i] + 2]++;
    for (int l = 0; l < nlv; l++) F->lvp[l + 2] += F->lvp[l + 1];
    for (int i = 0; i < n; i++) F->lvn[F->lvp[lv[i] + 1]++] = i;      /* (ascending indices within a level) */
    free(lv);
    F->Tp = (int *)calloc((size_t)n + 2, sizeof(int)); F->Ti = (int *)malloc(sizeof(int) * (size_t)(F->lnz + 1));
    F->Tx = (double *)malloc(sizeof(double) * (size_t)(F->lnz + 1)); F->Txl = (double *)malloc(sizeof(double) * (size_t)(F->lnz + 1));
}
static void ldl_transpose_dd(Ldl *F) {           /* L by rows, the entries of a row in ascending column order */
    const int n = F->n;
    for (int i = 0; i <= n + 1; i++) F->Tp[i] = 0;
    for (int j = 0; j < n; j++) for (int p = F->Lp[j]; p < F->Lp[j + 1]; p++) F->Tp[F->Li[p] + 2]++;
    for (int i = 0; i < n; i++) F->Tp[i + 2] += F->Tp[i + 1];
    for (int j = 0; j < n; j++) for (int p = F->Lp[j]; p < F->Lp[j + 1]; p++) { const int q = F->Tp[F->Li[p] + 1]++; F->Ti[q] = j; F->Tx[q] = F->Lx[p]; F->Txl[q] = F->Lxl[p]; }
}
static void ldl_solve_perm_dd_par(const Ldl *F, double *x, int nt) {
    const int n = F->n;
    double *xl = F->fwl;
    for (int j = 0; j < n; j++) xl[j] = 0.0;
    #pragma omp parallel num_threads(nt)
    {
        for (int l = 0; l < F->nlv; l++) {
            #pragma omp for schedule(static)
            for (int q = F->lvp[l]; q < F->lvp[l + 1]; q++) {
                const int i = F->lvn[q];
                sdd acc = { x[i], xl[i] };
                for (int p = F->Tp[i]; p < F->Tp[i + 1]; p++) {
                    const int j = F->Ti[p];
                    const sdd lv = { F->Tx[p], F->Txl[p] }, xj = { x[j], xl[j] };
                    if (xj.h != 0.0 || xj.l != 0.0) acc = dd_sub(acc, dd_mul(lv, xj));
                }
                x[i] = acc.h; xl[i] = acc.l;
            }
        }
        #pragma omp for schedule(static)
        for (int j = 0; j < n; j++) { const sdd xj = { x[j], xl[j] }, di = { F->Dinv[j], F->Dinvl[j] }; const sdd v = dd_mul(xj, di); x[j] = v.h; xl[j] = v.l; }
        for (int l = F->nlv - 1; l >= 0; l--) {
            #pragma omp for schedule(static)
            for (int q = F->lvp[l]; q < F->lvp[l + 1]; q++) {
                const int j = F->lvn[q];
                sdd sacc = { x[j], xl[j] };
                for (int p = F->Lp[j]; p < F->Lp[j + 1]; p++) {
                    const int r = F->Li[p];
                    const sdd lv = { F->Lx[p], F->Lxl[p] }, xr = { x[r], xl[r] };
                    sacc = dd_sub(sacc, dd_mul(lv, xr));
                }
                x[j] = sacc.h; xl[j] = sacc.l;
            }
        }
    }
}

/* (27 October 2026) the solves by the parts (ldl_part_build): forward, each part's columns on its thread for its own
 * rows, then every column's entries in top rows in increasing column order; backward, the top's columns, then each part's
 * on its thread (a column's rows are its ancestors: its own part's or the top's). Every entry of x receives the same
 * operations in the same order as in the sequential solve: the same bits on any number of threads, and as before */
static void ldl_solve_perm_part(const Ldl *F, double *x) {
    const int P = F->npart, T = F->ntop, nt = g_cone_nt_ldl > 1 && !omp_in_parallel() ? (g_cone_nt_ldl < P ? g_cone_nt_ldl : P) : 1;
    (void)T;
    /* forward: a part row's updates come from its own part's columns, in increasing order, on the part's thread; the top
     * rows' from all columns (a column's entries in top rows are its last ones, ptsplit), in increasing column order in one
     * sequential pass: the same order as the sequential solve's, so the same bits */
    #pragma omp parallel for schedule(dynamic, 1) num_threads(nt) if (nt > 1)
    for (int q = 0; q < P; q++)
        for (int a = F->pnp[q]; a < F->pnp[q + 1]; a++) {
            const int j = F->pnl[a]; const double xj = x[j];
            if (xj != 0.0) for (int p = F->Lp[j]; p < F->ptsplit[j]; p++) x[F->Li[p]] -= F->Lx[p] * xj;
        }
    for (int j = 0; j < F->n; j++) {
        const double xj = x[j];
        if (xj != 0.0) for (int p = F->ptsplit[j]; p < F->Lp[j + 1]; p++) x[F->Li[p]] -= F->Lx[p] * xj;
    }
    for (int j = 0; j < F->n; j++) x[j] *= F->Dinv[j];
    for (int t = T - 1; t >= 0; t--) {
        const int j = F->ptop[t]; double s = x[j];
        for (int p = F->Lp[j]; p < F->Lp[j + 1]; p++) s -= F->Lx[p] * x[F->Li[p]];
        x[j] = s;
    }
    #pragma omp parallel for schedule(dynamic, 1) num_threads(nt) if (nt > 1)
    for (int q = 0; q < P; q++)
        for (int a = F->pnp[q + 1] - 1; a >= F->pnp[q]; a--) {
            const int j = F->pnl[a]; double s = x[j];
            for (int p = F->Lp[j]; p < F->Lp[j + 1]; p++) s -= F->Lx[p] * x[F->Li[p]];
            x[j] = s;
        }
}
static void ldl_solve_perm_dd_part(const Ldl *F, double *x) {
    const int P = F->npart, T = F->ntop, nt = g_cone_nt_ldl > 1 && !omp_in_parallel() ? (g_cone_nt_ldl < P ? g_cone_nt_ldl : P) : 1;
    double *xl = F->fwl;
    for (int j = 0; j < F->n; j++) xl[j] = 0.0;
    #pragma omp parallel for schedule(dynamic, 1) num_threads(nt) if (nt > 1)
    for (int q = 0; q < P; q++)
        for (int a = F->pnp[q]; a < F->pnp[q + 1]; a++) {
            const int j = F->pnl[a]; const sdd xj = { x[j], xl[j] };
            if (xj.h != 0.0 || xj.l != 0.0)
                for (int p = F->Lp[j]; p < F->ptsplit[j]; p++) {
                    const int r = F->Li[p]; const sdd lv = { F->Lx[p], F->Lxl[p] }, xr = { x[r], xl[r] }, v = dd_sub(xr, dd_mul(lv, xj));
                    x[r] = v.h; xl[r] = v.l;
                }
        }
    for (int j = 0; j < F->n; j++) {
        const sdd xj = { x[j], xl[j] };
        if (xj.h != 0.0 || xj.l != 0.0)
            for (int p = F->ptsplit[j]; p < F->Lp[j + 1]; p++) {
                const int r = F->Li[p]; const sdd lv = { F->Lx[p], F->Lxl[p] }, xr = { x[r], xl[r] }, v = dd_sub(xr, dd_mul(lv, xj));
                x[r] = v.h; xl[r] = v.l;
            }
    }
    (void)T;
    for (int j = 0; j < F->n; j++) { const sdd xj = { x[j], xl[j] }, di = { F->Dinv[j], F->Dinvl[j] }; const sdd v = dd_mul(xj, di); x[j] = v.h; xl[j] = v.l; }
    for (int t = T - 1; t >= 0; t--) {
        const int j = F->ptop[t]; sdd sacc = { x[j], xl[j] };
        for (int p = F->Lp[j]; p < F->Lp[j + 1]; p++) { const int r = F->Li[p]; const sdd lv = { F->Lx[p], F->Lxl[p] }, xr = { x[r], xl[r] }; sacc = dd_sub(sacc, dd_mul(lv, xr)); }
        x[j] = sacc.h; xl[j] = sacc.l;
    }
    #pragma omp parallel for schedule(dynamic, 1) num_threads(nt) if (nt > 1)
    for (int q = 0; q < P; q++)
        for (int a = F->pnp[q + 1] - 1; a >= F->pnp[q]; a--) {
            const int j = F->pnl[a]; sdd sacc = { x[j], xl[j] };
            for (int p = F->Lp[j]; p < F->Lp[j + 1]; p++) { const int r = F->Li[p]; const sdd lv = { F->Lx[p], F->Lxl[p] }, xr = { x[r], xl[r] }; sacc = dd_sub(sacc, dd_mul(lv, xr)); }
            x[j] = sacc.h; xl[j] = sacc.l;
        }
}
/* x <- (L D L')^{-1} x in the permuted order */
static void ldl_solve_perm(const Ldl *F, double *x) {
    const int n = F->n;
    if (F->npart > 0 && F->ptsplit && F->pt_sol && g_cone_nt_ldl > 1 && !omp_in_parallel()) { ldl_solve_perm_part(F, x); return; }   /* (the same bits either way) */
    for (int j = 0; j < n; j++) {
        const double xj = x[j];
        if (xj != 0.0) for (int p = F->Lp[j]; p < F->Lp[j + 1]; p++) x[F->Li[p]] -= F->Lx[p] * xj;
    }
    for (int j = 0; j < n; j++) x[j] *= F->Dinv[j];
    for (int j = n - 1; j >= 0; j--) {
        double s = x[j];
        for (int p = F->Lp[j]; p < F->Lp[j + 1]; p++) s -= F->Lx[p] * x[F->Li[p]];
        x[j] = s;
    }
}

/* ------------------------------------------------------------------ the solver's state */
typedef struct {
    int m, n;                      /* rows, variables (after the rotated cones became second-order cones) */
    int nf, nl, nq; int *q, *qs;   /* free, linear, cones: dimensions and first index */
    int *Ap, *Ai; double *Ax;      /* A scaled, by columns */
    int *Rp, *Ri; double *Rx, *R0x;      /* A scaled, by rows (column indices); A as given, by rows (the same pattern) */
    /* (23 October 2026, the SOCP developer) A's dense rows as one dense block Dm (nd x ncd, by rows, over the union dcol of
     * their columns; dmap: row -> its index or -1, dcm: column -> its index or -1), the rest of A by columns (Bp, Bi, Bx;
     * Bq: a position of A -> its position in B, -1 on a dense row); dxs and dpart: work of the products */
    int nd, ncd, dch, *drow, *dmap, *dcol, *dcm, *Bp, *Bi, *Bq; double *Dm, *Bx, *dxs, *dacc, *dpart;
    double *b, *c;                 /* scaled */
    double *Dr, *Ec, sb, scl;      /* row and column scales, the two scalars */
    /* iterate */
    double *x, *y, *z, tau, kap;
    /* scaling: lp: h = z/x; cones: eta, wbar = (a, q) */
    double *h, *eta, *wbar;        /* h (nl), eta (nq), wbar (n: cone parts used) */
    double *lam;                   /* scaled variable lambda = W x (lp: sqrt(x z)) */
    /* KKT */
    int nk, naux; int *auxu, *auxv;        /* unknowns of the factored system; index of the two extra unknowns of cone k (-1: dense block) */
    Ldl F;
    int *kdiag;                    /* position in F.Ax of the diagonal entry of each unknown (original numbering) */
    int **kblk;                    /* dense cones: positions of the upper triangle (column by column) */
    /* rotated cones (dense cones in the eigenbasis of their scaling: diagonal block, columns A Q) */
    unsigned char *rot; int **rrows, *rnr; double **rG; long *rpos0; double *Qr; int **rapos;
    int **ku, **kv;                /* sparse cones: positions of the u and v entries */
    double *du, *dv, *dd0;         /* sparse cones: u0, u1, v1, d0 per cone packed: du[2k], du[2k+1]; dv[k]; dd0[k] */
    double sreg, sregx, sregf, sreg0, bump; int ref_pred, bump_on, nbump, bump_hold;
    int use_dd, dd_on, ndd, want_dd; long ncor_total;          /* the factorizations are in double-double from now on; allowed; how many */
    int dd_polish, nfailit; double it_maxres;                            /* double-double asked by the end-phase rule (its own work limit); worst GMRES residual of the iteration */
    double *w1, *w2, *w3, *w4, *rk, *sk;   /* work: n+m sized */
    double *lastx, *lasty, *swork2;
    double *gs, *gV, *gZ, *gH, *gcs; int gm_max; double gm_tol; double gm_abs; double tg_d, tg_p, tg_g, tg_f, tg_fl, tg_fd, tg_fp; long tg_skip; int tg_w;   /* GMRES on the KKT system (gm_abs > 0: an absolute target that also stops it, BRISK_CONEREFMU) */
    int rc_k, rc_kmax, rc_N; long rc_fact; long rc_used; double rc_gain;
    int nd_used;                     /* the nested dissection order was taken for the full system */   /* the last solve's Krylov space, reused for the next right-hand side of the same factorization */
    double rr_est; double t_mul; double cflops; double last_res; double t_setup, t_factor, t_solve, t_prec; long nfact; long nback, slnz, nfallback; int sc_ok;
    /* the normal equations A H^-1 A' + delta I (no free variable, small cones, no dense column) */
    double *fpr, *fpr_si, *fp_dxp, fp_rd;   /* (test, BRISK_CONEFREEPROXREL) the free variables' proximal weights; work over the rows; the last step of the free variables and the present dual residual's largest entry */
    SChol *ne; size_t *ne_off, *ne_doff; double *ne_G; int *ne_r, **ne_rows; int ne_ok, ne_fallback, ne_tiny, ne_left, ne_tinykept, free_last, dz_refine; double *wdz, *wdx, *wdy, *wdy2; long dz_count, zproj_used; int dd_force; double *ne_W, *ne_T, *ne_xf, *ne_Af; long ndyn_total; long ne_nfact, ne_lnz; double *ne_C; int nr_rot; int **bpos; double *bth; int *Rq; long nboost; int anyrot; double ne_dl;
    int ne_nd, ne_ndc, ne_md, *ne_dk, *ne_drow, *ne_dpos; size_t *ne_doffd; char ne_dense_msg[96]; int ne_nwl, *ne_wl, ne_wrank; int kkt_defer; long nrot_tr; int rot_mode, nsmallc, rot_switch, rot_back; double it_rs2, it_rt2;   /* (the eigenbasis of the small cones: 0 never, 1 from the start, 2 when the solves need it; the residual's share in the small cones this iteration) */
    /* (23 October 2026) rows that fix one entry of a small cone (a row with a single entry, in that cone's column), eliminated
     * from the normal equations: ne_fixrow[kc] the row (-1: none), ne_fixpos[kc] the entry's place in the cone, ne_fixa[kc]
     * its coefficient, ne_fixt[kc] the row's right-hand side during a solve; ne_elim[i] 1 for such a row */
    int ne_nfix, *ne_fixrow, *ne_fixpos; double *ne_fixa, *ne_fixt; char *ne_elim;
    /* (23 October 2026) dense linear columns kept as unknowns of the normal equations: [S A_d; A_d' -H_d] (dy; dx_d), the
     * unknowns m .. m + ne_nkd - 1; ne_kdcol the columns, ne_kdoff their diagonals' places, ne_kaoff the places of A_d's
     * entries (by A's index) */
    int ne_nkd, *ne_kdcol; size_t *ne_kdoff, *ne_kaoff; int ne_mm;   /* (ne_mm: the system's unknowns, m + ne_nkd) */
    int ne_ndt, *ne_dtcol, *ne_dtk; char *ne_xden;   /* (25 October 2026) the dense tail columns of the sparse-form cones, in the Woodbury update */
    int ne_nsp, *ne_spk, ne_nex, *ne_excol, *ne_exk, **ne_exrows, *ne_exr, ne_nu, ne_nu0, ne_wcap, ne_ntf, ne_depmode, ne_tfrow[NE_TINYMAX]; double ne_tfd[NE_TINYMAX], *ne_dg; double *ne_wU, *ne_wW, *ne_wC; int *ne_wpiv; unsigned char *fcone;   /* (large cones by their sparse low-rank form: the tail columns as pairs, [v a_0] by a Woodbury update) */     /* (the dense linear columns of the normal equations: their blocks, rows, offsets) */
    SChol *sc; size_t *soff; long knz; double *swork;   /* the supernodal factorization, when it is used */
    int refine_max, verbose;
    long refine_total, solves;
} Socp;

/* y += alpha A x ; y += alpha A' x.
 * 18 October 2026 (LP/SOCP): on the threads the SOCP solver keeps for its large blocks (g_cone_nt), when A has 1e5
 * entries or more: A'x by columns (each entry its own sum), A x as a gather by rows from a row copy whose entries
 * are in ascending column order, each y_i starting from its value: the additions into y_i in the order of the
 * scatter by columns, so the same bits at any thread count (BRISK_CONEAPAR=0: the sequential products). */
static _Thread_local int g_cone_nt = 1;     /* (per thread: the calling thread's solve; ISSUES 291) */
static int cone_part_nt(void) { return omp_in_parallel() ? 1 : g_cone_nt; }   /* (the partitioned solve's threads) */
static int cone_apar(const Socp *S) {
    static int on = -1; if (on < 0) { const char *e = getenv("BRISK_CONEAPAR"); on = !(e && atoi(e) == 0); }
    return on && g_cone_nt > 1 && S->Rp && !omp_in_parallel();
}
/* the dense block's products, by chunks of S->dch columns fixed by the block alone (its columns / 32, from 256 to 2048:
 * enough chunks for the threads' balance), each chunk's sums in a fixed order and
 * the chunks' partial sums of a row added in their order: the same bits at any thread count. wy (rows) += al D x,
 * wx (columns) += al D' yv; either may be NULL */
static _Thread_local long g_dn_calls[3]; static _Thread_local double g_dn_time;      /* (the debug line's counts: D x, D'y, both; their time) */
static void dn_mul(const Socp *S, double al, const double *x, double *wy, const double *yv, double *wx) {
    g_dn_calls[wy && wx ? 2 : wy ? 0 : 1]++; const double tdn = omp_get_wtime();
    const int nd = S->nd, nc = S->ncd, ck = S->dch, nch = (nc + ck - 1) / ck;
    double *xs = S->dxs, *acc = S->dacc, *part = S->dpart;
    if (wy) for (int c = 0; c < nc; c++) xs[c] = x[S->dcol[c]];
    const int par = g_cone_nt > 1 && !omp_in_parallel();
    #pragma omp parallel for schedule(static) num_threads(g_cone_nt) if (par)
    for (int ch = 0; ch < nch; ch++) {
        const int c0 = ch * ck, c1 = c0 + ck < nc ? c0 + ck : nc;
        double *a = acc + c0;
        if (wx) for (int c = c0; c < c1; c++) a[c - c0] = 0.0;
        for (int i = 0; i < nd; i++) {
            const double *Di = S->Dm + (size_t)i * nc;
            if (wy) {
                double s0 = 0, s1 = 0, s2 = 0, s3 = 0; int c = c0;
                for (; c + 3 < c1; c += 4) { s0 += Di[c] * xs[c]; s1 += Di[c + 1] * xs[c + 1]; s2 += Di[c + 2] * xs[c + 2]; s3 += Di[c + 3] * xs[c + 3]; }
                for (; c < c1; c++) s0 += Di[c] * xs[c];
                part[(size_t)ch * nd + i] = (s0 + s1) + (s2 + s3);
            }
            if (wx) { const double yi = yv[S->drow[i]]; if (yi != 0.0) for (int c = c0; c < c1; c++) a[c - c0] += Di[c] * yi; }
        }
    }
    if (wy) for (int i = 0; i < nd; i++) { double s = 0; for (int ch = 0; ch < nch; ch++) s += part[(size_t)ch * nd + i]; wy[S->drow[i]] += al * s; }
    if (wx) for (int c = 0; c < nc; c++) wx[S->dcol[c]] += al * acc[c];
    g_dn_time += omp_get_wtime() - tdn;
}
/* A x and A'y of the rest of A, the dense block apart (S->nd > 0) */
static void B_mul(const Socp *S, double al, const double *x, double *y) {
    if (cone_apar(S)) {
        #pragma omp parallel for schedule(static) num_threads(g_cone_nt)
        for (int i = 0; i < S->m; i++) {
            if (S->dmap[i] >= 0) continue;
            double v = y[i];
            for (int q = S->Rp[i]; q < S->Rp[i + 1]; q++) { const double xj = al * x[S->Ri[q]]; if (xj != 0.0) v += S->Rx[q] * xj; }
            y[i] = v;
        }
        return;
    }
    for (int j = 0; j < S->n; j++) { const double xj = al * x[j]; if (xj != 0.0) for (int p = S->Bp[j]; p < S->Bp[j + 1]; p++) y[S->Bi[p]] += S->Bx[p] * xj; }
}
static void Bt_mul(const Socp *S, double al, const double *x, double *y) {
    #pragma omp parallel for schedule(static) num_threads(g_cone_nt) if (cone_apar(S))
    for (int j = 0; j < S->n; j++) { double s = 0.0; for (int p = S->Bp[j]; p < S->Bp[j + 1]; p++) s += S->Bx[p] * x[S->Bi[p]]; y[j] += al * s; }
}
static void A_mul(const Socp *S, double al, const double *x, double *y) {
    if (S->nd > 0) { B_mul(S, al, x, y); dn_mul(S, al, x, y, NULL, NULL); return; }
    if (cone_apar(S)) {
        #pragma omp parallel for schedule(static) num_threads(g_cone_nt)
        for (int i = 0; i < S->m; i++) {
            double v = y[i];
            for (int q = S->Rp[i]; q < S->Rp[i + 1]; q++) { const double xj = al * x[S->Ri[q]]; if (xj != 0.0) v += S->Rx[q] * xj; }
            y[i] = v;
        }
        return;
    }
    for (int j = 0; j < S->n; j++) { const double xj = al * x[j]; if (xj != 0.0) for (int p = S->Ap[j]; p < S->Ap[j + 1]; p++) y[S->Ai[p]] += S->Ax[p] * xj; }
}
static void At_mul(const Socp *S, double al, const double *x, double *y) {
    if (S->nd > 0) { Bt_mul(S, al, x, y); dn_mul(S, al, NULL, NULL, x, y); return; }
    #pragma omp parallel for schedule(static) num_threads(g_cone_nt) if (cone_apar(S))
    for (int j = 0; j < S->n; j++) { double s = 0.0; for (int p = S->Ap[j]; p < S->Ap[j + 1]; p++) s += S->Ax[p] * x[S->Ai[p]]; y[j] += al * s; }
}
/* (27 October 2026, the SOCP developer) the loops over the cones on the threads: each cone writes only its own entries, and a
 * minimum is the same in any order, so the results are the same bits at any thread count (the SDP developer and
 * integrator's finding on 400,000 small cones). Worth a parallel region (a few microseconds) from about 32,768 entries on. */
static int cone_vpar(const Socp *S) {
    static int on = -1; if (on < 0) { const char *e = getenv("BRISK_CONEVPAR"); on = !(e && atoi(e) == 0); }
    return on && g_cone_nt > 1 && S->n >= 32768 && !omp_in_parallel();
}
static double dotv(int n, const double *a, const double *b) { double s = 0; for (int i = 0; i < n; i++) s += a[i] * b[i]; return s; }
static double nrm2(int n, const double *a) { return sqrt(dotv(n, a, a)); }
static double nrmi(int n, const double *a) { double s = 0; for (int i = 0; i < n; i++) if (fabs(a[i]) > s) s = fabs(a[i]); return s; }
/* (27 October 2026, the SOCP developer) the scaling of Curtis and Reid (1972): row-group and column-group factors e^rho, e^gamma
 * that minimize the sum over the entries of (log|a_ij| + rho_g(i) + gamma_h(j))^2, by conjugate gradients on its normal
 * equations with their diagonal (the entries' counts) as the preconditioner. Its scaled matrix does not depend on any row and
 * column scaling of the input (the minimizer moves by the logarithms of the scaling, exactly), unlike the passes that divide by
 * the largest entries, whose fixed point depends on where they start. A column with cg[j] < 0 is left out. */
static void cr_scale(int m, int n, const int *Ap, const int *Ai, const double *Ax, const int *rg, int nrg, const int *cg, int ncg, int maxit, double *rho, double *gam, const double *bv, const double *cv) {
    /* (bv, cv not NULL: b a column and c a row of the matrix, [A b; c' 0], each with a factor of its own, group nrg - 1 the row
     * of c and ncg - 1 the column of b: the costs and the right-hand side take part in the balance) */
    const int N = nrg + ncg;
    double *d = (double *)calloc((size_t)N + 1, sizeof(double)), *r = (double *)calloc((size_t)N + 1, sizeof(double)), *z = (double *)malloc(sizeof(double) * ((size_t)N + 1));
    double *p = (double *)calloc((size_t)N + 1, sizeof(double)), *q = (double *)malloc(sizeof(double) * ((size_t)N + 1)), *u = (double *)calloc((size_t)N + 1, sizeof(double));
    (void)m;
    for (int j = 0; j < n; j++) { if (cg[j] < 0) continue; const int h = nrg + cg[j];
        for (int t = Ap[j]; t < Ap[j + 1]; t++) { if (Ax[t] == 0.0) continue; const int g = rg[Ai[t]]; const double l = log(fabs(Ax[t])); d[g] += 1; d[h] += 1; r[g] -= l; r[h] -= l; } }
    if (bv) for (int i = 0; i < m; i++) if (bv[i] != 0.0) { const int g = rg[i], h = N - 1; const double l = log(fabs(bv[i])); d[g] += 1; d[h] += 1; r[g] -= l; r[h] -= l; }
    if (cv) for (int j = 0; j < n; j++) if (cv[j] != 0.0 && cg[j] >= 0) { const int g = nrg - 1, h = nrg + cg[j]; const double l = log(fabs(cv[j])); d[g] += 1; d[h] += 1; r[g] -= l; r[h] -= l; }
    static double lam = -1; if (lam < 0) lam = getenv("BRISK_CONEEQCRL") ? atof(getenv("BRISK_CONEEQCRL")) : 1e-2;
    double r0 = 0; for (int k = 0; k < N; k++) r0 += r[k] * r[k];
    double rz = 0; for (int k = 0; k < N; k++) { z[k] = d[k] > 0 ? r[k] / ((1 + lam) * d[k]) : 0.0; rz += r[k] * z[k]; p[k] = z[k]; }
    for (int it = 0; it < maxit && rz > 0; it++) {
        for (int k = 0; k < N; k++) q[k] = lam * d[k] * p[k];
        for (int j = 0; j < n; j++) { if (cg[j] < 0) continue; const int h = nrg + cg[j];
            for (int t = Ap[j]; t < Ap[j + 1]; t++) { if (Ax[t] == 0.0) continue; const int g = rg[Ai[t]]; const double v = p[g] + p[h]; q[g] += v; q[h] += v; } }
        if (bv) for (int i = 0; i < m; i++) if (bv[i] != 0.0) { const int g = rg[i], h = N - 1; const double v = p[g] + p[h]; q[g] += v; q[h] += v; }
        if (cv) for (int j = 0; j < n; j++) if (cv[j] != 0.0 && cg[j] >= 0) { const int g = nrg - 1, h = nrg + cg[j]; const double v = p[g] + p[h]; q[g] += v; q[h] += v; }
        double pq = 0; for (int k = 0; k < N; k++) pq += p[k] * q[k];
        if (!(pq > 0)) break;
        const double al = rz / pq; double rr = 0, rzn = 0;
        for (int k = 0; k < N; k++) { u[k] += al * p[k]; r[k] -= al * q[k]; rr += r[k] * r[k]; }
        if (rr <= 1e-16 * r0) break;
        for (int k = 0; k < N; k++) { z[k] = d[k] > 0 ? r[k] / ((1 + lam) * d[k]) : 0.0; rzn += r[k] * z[k]; }
        const double be = rzn / rz; rz = rzn;
        for (int k = 0; k < N; k++) p[k] = z[k] + be * p[k];
    }
    /* the minimizer is fixed up to rho + t, gamma - t on each connected part of the matrix (the same scaled matrix, b times
     * e^t and c times e^-t), a direction the preconditioned iterations do not keep out: each part's t taken so that its mean
     * rho and its mean gamma are equal (b and c are divided by their largest entries afterwards) */
    int *par = (int *)malloc(sizeof(int) * ((size_t)N + 1));
    for (int k = 0; k < N; k++) par[k] = k;
    for (int j = 0; j < n; j++) { if (cg[j] < 0) continue; int h = nrg + cg[j]; while (par[h] != h) h = par[h] = par[par[h]];
        for (int t = Ap[j]; t < Ap[j + 1]; t++) { if (Ax[t] == 0.0) continue; int g = rg[Ai[t]]; while (par[g] != g) g = par[g] = par[par[g]]; if (g != h) { par[g] = h; } } }
    #define CR_UNION(g0, h0) do { int g_ = (g0), h_ = (h0); while (par[g_] != g_) g_ = par[g_] = par[par[g_]]; while (par[h_] != h_) h_ = par[h_] = par[par[h_]]; if (g_ != h_) par[g_] = h_; } while (0)
    if (bv) for (int i = 0; i < m; i++) if (bv[i] != 0.0) CR_UNION(rg[i], N - 1);
    if (cv) for (int j = 0; j < n; j++) if (cv[j] != 0.0 && cg[j] >= 0) CR_UNION(nrg - 1, nrg + cg[j]);
    #undef CR_UNION
    double *sr = (double *)calloc((size_t)N + 1, sizeof(double)), *sg = (double *)calloc((size_t)N + 1, sizeof(double));
    int *nr_ = (int *)calloc((size_t)N + 1, sizeof(int)), *ng_ = (int *)calloc((size_t)N + 1, sizeof(int));
    for (int k = 0; k < N; k++) { int a = k; while (par[a] != a) a = par[a]; par[k] = a; if (k < nrg) { sr[a] += u[k]; nr_[a]++; } else { sg[a] += u[k]; ng_[a]++; } }
    for (int k = 0; k < N; k++) { const int a = par[k]; const double t = nr_[a] && ng_[a] ? 0.5 * (sg[a] / ng_[a] - sr[a] / nr_[a]) : 0.0; u[k] += k < nrg ? t : -t; }
    free(par); free(sr); free(sg); free(nr_); free(ng_);
    for (int g = 0; g < nrg; g++) rho[g] = u[g];
    for (int h = 0; h < ncg; h++) gam[h] = u[nrg + h];
    free(d); free(r); free(z); free(p); free(q); free(u);
}

/* cone k: out = W in (inv = 0) or W^{-1} in (inv = 1); in and out may not overlap */
static void W_apply(const Socp *S, int k, int inv, const double *in, double *out) {
    const int d = S->q[k]; const double *w = S->wbar + S->qs[k];
    const double a = w[0], eta = S->eta[k];
    double qx = 0.0;
    for (int i = 1; i < d; i++) qx += w[i] * in[i];
    if (!inv) {
        out[0] = eta * (a * in[0] + qx);
        const double f = in[0] + qx / (1.0 + a);
        for (int i = 1; i < d; i++) out[i] = eta * (in[i] + f * w[i]);
    } else {
        out[0] = (a * in[0] - qx) / eta;
        const double f = -in[0] + qx / (1.0 + a);
        for (int i = 1; i < d; i++) out[i] = (in[i] + f * w[i]) / eta;
    }
}
/* out = H in over all variables (free: 0; lp: h; cones: eta^2 (2 w w' - J)) */
static void H_apply(const Socp *S, const double *in, double *out) {
    for (int i = 0; i < S->nf; i++) out[i] = 0.0;
    const int par = cone_vpar(S);
    #pragma omp parallel for schedule(static) num_threads(g_cone_nt) if (par)
    for (int i = 0; i < S->nl; i++) out[S->nf + i] = S->h[i] * in[S->nf + i];
    #pragma omp parallel for schedule(static) num_threads(g_cone_nt) if (par)
    for (int k = 0; k < S->nq; k++) {
        const int d = S->q[k], s = S->qs[k]; const double *w = S->wbar + s; const double e2 = S->eta[k] * S->eta[k];
        double wx = 0.0;
        for (int i = 0; i < d; i++) wx += w[i] * in[s + i];
        out[s] = e2 * (2.0 * wx * w[0] - in[s]);
        for (int i = 1; i < d; i++) out[s + i] = e2 * (2.0 * wx * w[i] + in[s + i]);
    }
}

/* the values of the factored matrix from the current scaling */
static int kkt_analyse(Socp *S);
static void kkt_fill(Socp *S) {
    Ldl *F = &S->F;
    if (S->ne && !S->ne_fallback) return;            /* (the normal equations are assembled by ne_factor) */
    if (S->kkt_defer) {                               /* (the full system needed for the first time: analysed now) */
        const double ts = sc_time(); const int rc = kkt_analyse(S); S->t_setup += sc_time() - ts;
        if (dbg_on()) socp_printf("  [the full system analysed when first needed: %.3fs%s]\n", sc_time() - ts, rc ? ", failed" : "");
    }
    const double sr = S->sreg;
    for (int i = 0; i < S->nf; i++) F->Ax[S->kdiag[i]] = -(S->use_dd ? sr : S->sregf);
    const double sx = S->sregx;
    /* (20 October 2026, LP/SOCP, a test: BRISK_CONEREGSMALL=r, the regularization of the cones held as dense blocks (no
     * auxiliary rows) alone; the linear block and the large cones keep sregx) */
    static double sxs_ = -1; if (sxs_ < 0) { const char *e_ = getenv("BRISK_CONEREGSMALL"); sxs_ = e_ ? atof(e_) : -2; }
    const double sxs = S->use_dd || sxs_ < 0 ? sx : sxs_;
    /* (27 October 2026, the SOCP developer, a test: BRISK_CONEREGREL=rho[,floor]) the linear and cone blocks' regularization
     * relative to each diagonal entry, max(rho |H_ii|, floor), instead of the fixed sregx: late in a degenerate solve the
     * entries of H fall to 1e-9 and below (joint_FC_7, SOCP: 290,000 cone pivots below 1e-6, the smallest at 1.1e-9), the
     * fixed 1e-9 then dominates them, the factor is a poor preconditioner there and every solve takes up to 40 GMRES
     * steps (ISSUES 217: "a regularization only where a pivot is dangerous") */
    static double rgr = -1, rgf = 1e-14; if (rgr < 0) { const char *e_ = getenv("BRISK_CONEREGREL"); rgr = 0.0; if (e_) sscanf(e_, "%lf,%lf", &rgr, &rgf); }
    const int rgon = rgr > 0 && !S->use_dd && S->bump == 1.0;
    for (int i = 0; i < S->nl; i++) F->Ax[S->kdiag[S->nf + i]] = -S->h[i] - (rgon ? fmax(rgr * S->h[i], rgf) : sx);
    for (int k = 0; k < S->nq; k++) {
        const int d = S->q[k], s = S->qs[k]; const double *w = S->wbar + s; const double e2 = S->eta[k] * S->eta[k];
        if (S->auxu[k] < 0 && S->rot[k]) {
            /* Q = [e+ e- t_1 ..], e+- = (1, +-r)/sqrt2, r = w1/|w1|, t_j an orthonormal basis of r's
             * complement; eigenvalues e2 (w0+|w1|)^2, e2 / (w0+|w1|)^2, e2 */
            double *Q = S->Qr + 16 * (size_t)k, r[4] = { 0, 0, 0, 0 };
            double nq = 0.0; for (int i = 1; i < d; i++) nq += w[i] * w[i];
            nq = sqrt(nq);
            if (nq > 0) for (int i = 1; i < d; i++) r[i - 1] = w[i] / nq; else r[0] = 1.0;
            const double is2 = sqrt(0.5);
            Q[0] = is2; Q[d] = is2;
            for (int i = 1; i < d; i++) { Q[i] = is2 * r[i - 1]; Q[d + i] = -is2 * r[i - 1]; }
            if (d == 3) { Q[6] = 0; Q[7] = -r[1]; Q[8] = r[0]; }
            else if (d == 4) {
                int mi = 0; for (int i = 1; i < 3; i++) if (fabs(r[i]) < fabs(r[mi])) mi = i;
                double t1[3] = { 0, 0, 0 }; t1[mi] = 1.0;
                for (int i = 0; i < 3; i++) t1[i] -= r[mi] * r[i];
                double nt = sqrt(t1[0] * t1[0] + t1[1] * t1[1] + t1[2] * t1[2]);
                for (int i = 0; i < 3; i++) t1[i] /= nt;
                const double t2[3] = { r[1] * t1[2] - r[2] * t1[1], r[2] * t1[0] - r[0] * t1[2], r[0] * t1[1] - r[1] * t1[0] };
                Q[8] = 0; Q[9] = t1[0]; Q[10] = t1[1]; Q[11] = t1[2];
                Q[12] = 0; Q[13] = t2[0]; Q[14] = t2[1]; Q[15] = t2[2];
            }
            const double lp = (w[0] + nq) * (w[0] + nq);
            const int *pos = S->kblk[k];
            int t = 0;
            for (int j = 0; j < d; j++)
                for (int i = 0; i <= j; i++) {
                    const double lam = j == 0 ? lp : j == 1 ? 1.0 / lp : 1.0;
                    F->Ax[pos[t++]] = i == j ? -e2 * lam - (rgon ? fmax(rgr * e2 * lam, rgf) : sxs) : 0.0;
                }
            /* the columns A Q */
            const int nr = S->rnr[k]; const double *G = S->rG[k]; const int *ap = S->rapos[k];
            for (int i = 0; i < d; i++)
                for (int rr = 0; rr < nr; rr++) {
                    double v = 0; for (int j = 0; j < d; j++) v += G[rr + (size_t)j * nr] * Q[j + i * d];
                    F->Ax[ap[(long)i * nr + rr]] = v;
                }
        } else if (S->auxu[k] < 0) {
            const int *pos = S->kblk[k];
            int t = 0;
            for (int j = 0; j < d; j++)
                for (int i = 0; i <= j; i++) {
                    double v = 2.0 * w[i] * w[j];
                    if (i == j) v += i == 0 ? -1.0 : 1.0;
                    F->Ax[pos[t++]] = -e2 * v - (i == j ? (rgon ? fmax(rgr * e2 * fabs(v), rgf) : sxs) : 0.0);
                }
        } else {
            /* H / eta^2 = I + (l+ - 1) p+ p+' - (1 - l-) p- p-',  l+- = (a +- |q|)^2,  p+- = (1, +-q/|q|) / sqrt 2:
             * the spectral form. Its entries are computed without cancellation (l- = 1 / l+), the
             * diagonal is the identity, and I - v v' has the smallest eigenvalue l- > 0. */
            const double a = w[0], eta = S->eta[k];
            double nq = 0.0; for (int i = 1; i < d; i++) nq += w[i] * w[i];
            nq = sqrt(nq);
            const double lp = (a + nq) * (a + nq), lm = 1.0 / lp;
            const double su = sqrt(0.5 * (lp - 1.0)), sv = sqrt(0.5 * (1.0 - lm));
            const double iq = nq > 0 ? 1.0 / nq : 0.0;
            for (int i = 0; i < d; i++) F->Ax[S->kdiag[s + i]] = -e2 - (rgon ? fmax(rgr * e2, rgf) : sx);
            F->Ax[S->ku[k][0]] = eta * su; F->Ax[S->kv[k][0]] = eta * sv;
            for (int i = 1; i < d; i++) { F->Ax[S->ku[k][i]] = eta * su * iq * w[i]; F->Ax[S->kv[k][i]] = -eta * sv * iq * w[i]; }
            F->Ax[S->kdiag[S->auxu[k]]] = 1.0;
            F->Ax[S->kdiag[S->auxv[k]]] = -1.0;
        }
    }
    /* (20 October 2026, LP/SOCP, a test: BRISK_CONERELREG=rho[,floor]) the rows' regularization relative to the row's own scale in
     * the Schur complement, s_i = sum_j a_ij^2 (H^-1)_jj (free columns at 1/delta_f): delta_i = max(rho s_i, floor). A fixed
     * delta is a 100 % perturbation of a row whose variables have all gone to zero (its pivot falls like mu), and the
     * preconditioned GMRES then contracts by 0.5-0.9 a step on every such row; relative to the row's scale, the
     * preconditioner stays within rho of the matrix. The rounding of a pivot is relative to its summands, so the
     * protection a fixed delta gives is also relative. */
    static double relreg = -1, relfl = 1e-14; if (relreg < 0) { const char *e_ = getenv("BRISK_CONERELREG"); relreg = 0.0; if (e_) sscanf(e_, "%lf,%lf", &relreg, &relfl); }
    static int reldd = -1; if (reldd < 0) { const char *e_ = getenv("BRISK_CONERELREGDD"); reldd = e_ ? atoi(e_) : 0; }   /* (in double-double too, with the floor at its regularization) */
    if (relreg > 0 && (!S->use_dd || reldd)) {
        const double relfl_ = S->use_dd ? fmax(relfl, sr) : relfl;
        double *si = S->swork2;
        for (int i = 0; i < S->m; i++) si[i] = 0.0;
        static double relf = -1; if (relf < 0) { const char *e_ = getenv("BRISK_CONERELREGF"); relf = e_ ? atof(e_) : 0.0; }   /* (the free columns' share: relf / delta_f; 0 none) */
        if (relf > 0) for (int j = 0; j < S->nf; j++) { const double hi = relf / S->sregf; for (int p = S->Ap[j]; p < S->Ap[j + 1]; p++) si[S->Ai[p]] += S->Ax[p] * S->Ax[p] * hi; }
        for (int i = 0; i < S->nl; i++) { const int j = S->nf + i; const double hi = 1.0 / S->h[i]; for (int p = S->Ap[j]; p < S->Ap[j + 1]; p++) si[S->Ai[p]] += S->Ax[p] * S->Ax[p] * hi; }
        for (int k = 0; k < S->nq; k++) {
            const int d = S->q[k], s0 = S->qs[k]; const double *w = S->wbar + s0; const double e2 = S->eta[k] * S->eta[k];
            double nq = 0.0; for (int i = 1; i < d; i++) nq += w[i] * w[i];
            const double hmax = (w[0] + sqrt(nq)) * (w[0] + sqrt(nq)) / e2;      /* the largest eigenvalue of H^-1 */
            for (int i = 0; i < d; i++) { const int j = s0 + i; const double hi = S->rot && S->rot[k] ? hmax : (2.0 * w[i] * w[i] + (i ? 1.0 : -1.0)) / e2;
                for (int p = S->Ap[j]; p < S->Ap[j + 1]; p++) si[S->Ai[p]] += S->Ax[p] * S->Ax[p] * hi; }
        }
        for (int i = 0; i < S->m; i++) F->Ax[S->kdiag[S->n + S->naux + i]] = fmax(relreg * si[i], relfl_);
    } else
    for (int i = 0; i < S->m; i++) F->Ax[S->kdiag[S->n + S->naux + i]] = sr;
    /* (27 October 2026, the SOCP developer, a test: BRISK_CONEFREEPROXREL=kappa) the free variables' regularization kept in
     * the solved system as a proximal term, each free variable's weight capped at kappa times its curvature through the
     * rows: rho_j = min(delta_f, kappa s_j), s_j = sum_i a_ij^2 / M_ii, M_ii = sum over the other columns of a_ik^2 (H^-1)_kk
     * (the diagonal of the normal equations without the free columns; a Jacobi estimate). The term (rho_j / 2)(x_j - x_j^k)^2
     * damps the variable's step by about rho_j / s_j <= kappa, so its dual residual still falls each iteration (the whole
     * term, BRISK_CONEFREEPROX=1, left socp_riskinf_1000x20xnear's one free variable with a dual residual stalled at the
     * term's size: 19 -> 195 iterations); where rho_j = delta_f the factor is exact on that variable and the refinement has
     * nothing to remove there (the structural-design SOCPs' late solves: their error is in the free variables' rows) */
    /* (the second bound, kappa,theta: rho_j |dx_j| of the last step at most theta times the present dual residual's largest
     * entry, so the term's own error, the weight times the step, stays below what the next step must remove: on
     * socp_beamform_200x3000, SOCP, all 400 free variables took the whole weight and the dual residual stalled at about
     * 2e-8, 8 -> 11 iterations) */
    { static double fpk = -1, fpt = 0.0; if (fpk < 0) { const char *e_ = getenv("BRISK_CONEFREEPROXREL"); fpk = 0.0; if (e_) sscanf(e_, "%lf,%lf", &fpk, &fpt); }
      if (fpk > 0 && !S->use_dd && S->nf > 0) {
        const int m = S->m;
        if (!S->fpr) { S->fpr = (double *)calloc((size_t)S->nf + 1, sizeof(double)); S->fpr_si = (double *)calloc((size_t)m + 1, sizeof(double)); }
        double *si = S->fpr_si;
        for (int i = 0; i < m; i++) si[i] = sr;
        for (int i = 0; i < S->nl; i++) { const int j = S->nf + i; const double hi = 1.0 / S->h[i]; for (int p = S->Ap[j]; p < S->Ap[j + 1]; p++) si[S->Ai[p]] += S->Ax[p] * S->Ax[p] * hi; }
        for (int k = 0; k < S->nq; k++) {
            const int d = S->q[k], s0 = S->qs[k]; const double *w = S->wbar + s0; const double e2 = S->eta[k] * S->eta[k];
            double nq = 0.0; for (int i = 1; i < d; i++) nq += w[i] * w[i];
            const double hmax = (w[0] + sqrt(nq)) * (w[0] + sqrt(nq)) / e2;
            for (int i = 0; i < d; i++) { const int j = s0 + i; const double hi = S->rot && S->rot[k] ? hmax : (2.0 * w[i] * w[i] + (i ? 1.0 : -1.0)) / e2;
                for (int p = S->Ap[j]; p < S->Ap[j + 1]; p++) si[S->Ai[p]] += S->Ax[p] * S->Ax[p] * hi; }
        }
        long nfull = 0;
        for (int j = 0; j < S->nf; j++) {
            double sj = 0.0; for (int p = S->Ap[j]; p < S->Ap[j + 1]; p++) sj += S->Ax[p] * S->Ax[p] / si[S->Ai[p]];
            S->fpr[j] = fmin(S->sregf, fpk * sj);
            if (fpt > 0 && S->fp_dxp && fabs(S->fp_dxp[j]) > 0) S->fpr[j] = fmin(S->fpr[j], fpt * S->fp_rd / fabs(S->fp_dxp[j]));
            if (S->fpr[j] >= S->sregf) nfull++;
        }
        if (dbg_on()) socp_printf("      [the free variables' proximal weights: %ld of %d at the factor's %.1e]\n", nfull, S->nf, S->sregf);
      } }
}

/* factors the matrix of kkt_fill; returns the number of pivots replaced (own factorization) or 1 when the
 * supernodal factorization met a pivot of the wrong sign */
/* out = H^-1 in on the cone part (free variables: 0); H^-1 = eta^-2 (2 u u' - J), u = J w */
static void Hinv_apply(const Socp *S, const double *in, double *out) {
    for (int i = 0; i < S->nf; i++) out[i] = 0.0;
    const int par = cone_vpar(S);
    #pragma omp parallel for schedule(static) num_threads(g_cone_nt) if (par)
    for (int i = 0; i < S->nl; i++) out[S->nf + i] = in[S->nf + i] / S->h[i];
    #pragma omp parallel for schedule(static) num_threads(g_cone_nt) if (par)
    for (int k = 0; k < S->nq; k++) {
        const int d = S->q[k], s = S->qs[k]; const double *w = S->wbar + s; const double ie2 = 1.0 / (S->eta[k] * S->eta[k]);
        double ux = w[0] * in[s];
        for (int i = 1; i < d; i++) ux -= w[i] * in[s + i];
        out[s] = ie2 * (2.0 * ux * w[0] - in[s]);
        for (int i = 1; i < d; i++) out[s + i] = ie2 * (-2.0 * ux * w[i] + in[s + i]);
    }
}

/* ---- the normal equations. With no free variable, small cones and no dense column the system is
 * reduced to  (A H^-1 A' + delta I) dy = ry + A H^-1 rx,  dx = H^-1 (A' dy - rx):  a positive definite
 * matrix, factored by the sparse Cholesky factorization with its treatment of tiny pivots (the
 * unknown is dropped), where the quasi-definite factorization of the full system meets pivots of the
 * wrong sign late in the solve of a degenerate problem. A block is a linear variable or a cone; its rows
 * are those of its columns of A. */
#define NE_MAXCONE 16
/* (NE_MAXFREE, above: a few free variables are treated by their Schur complement, one more solve each per factorization) */
static void ne_wood_apply(const Socp *S, double *t);
static int ne_col_in(const Socp *S, int j, const char *blk) { for (int p = S->Ap[j]; p < S->Ap[j + 1]; p++) if (!blk[S->Ai[p]]) return 0; return 1; }
/* (20 October 2026, LP/SOCP) R, the symmetric square root of a cone's H_k^-1 = eta^-2 (2 u u' - J), u = (u0, u1) with
 * u' J u = 1, in closed form: Q^(1/2) = [[u0, u1'], [u1, I + (u0 - 1) u1 u1' / |u1|^2]] (the eigenvalues of Q are
 * (u0 +- |u1|)^2 and 1, those of its root u0 +- |u1| and 1), so R = Q^(1/2) / eta and R R' = H_k^-1. A Cholesky
 * factor of Q in double fails late in a solve (the condition number of Q reaches 1e28: the group lasso's last
 * iteration left the normal equations for the full system); the root's entries are of the order of u0. d x d. */
static int ne_cone_sqrt(const Socp *S, int kc, double *R) {
    const int d = S->q[kc]; const double *w = S->wbar + S->qs[kc]; const double ie = 1.0 / S->eta[kc];
    const double u0 = w[0]; double n2 = 0; for (int i = 1; i < d; i++) n2 += w[i] * w[i];
    if (!(u0 > 0) || !isfinite(u0) || !isfinite(n2)) return 1;
    const double g = n2 > 0 ? (u0 - 1.0) / n2 : 0.0;       /* (u1 = -w1: the signs cancel in u1 u1'; the off-diagonal takes -w) */
    R[0] = ie * u0;
    for (int i = 1; i < d; i++) { R[i] = R[i * d] = ie * (-w[i]); for (int j = 1; j < d; j++) R[i * d + j] = ie * ((i == j ? 1.0 : 0.0) + g * w[i] * w[j]); }
    return 0;
}
static void ne_free(Socp *S) {
    free(S->ne_fixrow); free(S->ne_fixpos); free(S->ne_fixa); free(S->ne_fixt); free(S->ne_elim);
    S->ne_fixrow = S->ne_fixpos = NULL; S->ne_fixa = S->ne_fixt = NULL; S->ne_elim = NULL; S->ne_nfix = 0;
    free(S->ne_kdcol); free(S->ne_kdoff); free(S->ne_kaoff); S->ne_kdcol = NULL; S->ne_kdoff = S->ne_kaoff = NULL; S->ne_nkd = 0;
    if (S->ne) schol_free(S->ne);
    S->ne = NULL;
    free(S->ne_off); free(S->ne_doff); free(S->ne_G); free(S->ne_r); free(S->ne_C); S->ne_C = NULL;
    if (S->ne_rows) { free(S->ne_rows[0]); free(S->ne_rows); }
    S->ne_off = S->ne_doff = NULL; S->ne_G = NULL; S->ne_r = NULL; S->ne_rows = NULL;
    free(S->ne_W); S->ne_W = NULL; free(S->ne_T); S->ne_T = NULL; free(S->ne_xf); S->ne_xf = NULL; free(S->ne_Af); S->ne_Af = NULL;
    free(S->ne_wl); S->ne_wl = NULL; S->ne_nwl = 0; S->ne_wrank = 0;
    free(S->ne_dk); free(S->ne_drow); free(S->ne_dpos); free(S->ne_doffd); S->ne_dk = S->ne_drow = S->ne_dpos = NULL; S->ne_doffd = NULL; S->ne_nd = S->ne_ndc = S->ne_md = 0;
    free(S->ne_spk); free(S->ne_excol); free(S->ne_exk); free(S->ne_exr); if (S->ne_exrows) { free(S->ne_exrows[0]); free(S->ne_exrows); }
    free(S->ne_wU); free(S->ne_wW); free(S->ne_wC); free(S->ne_wpiv); free(S->ne_dg); S->ne_dg = NULL; S->ne_ntf = 0;
    S->ne_spk = S->ne_excol = S->ne_exk = S->ne_exr = NULL; S->ne_exrows = NULL; S->ne_wU = S->ne_wW = S->ne_wC = NULL; S->ne_wpiv = NULL; S->ne_nsp = S->ne_nex = S->ne_nu = 0; S->ne_wcap = 0;
    free(S->ne_dtcol); free(S->ne_dtk); S->ne_dtcol = S->ne_dtk = NULL; S->ne_ndt = 0; free(S->ne_xden); S->ne_xden = NULL;
}
/* (20 October 2026, LP/SOCP) C += alpha W W' (lower triangle, C r x r column-major, W r x k), by row blocks of 256 on
 * the threads: each block's diagonal tile by dsyrk, the rest of its rows by dgemm (disjoint parts of C: the same bits at
 * any thread count) */
static void syrk_blocks(int rr, int kk, double alpha, const double *W, double beta, double *C) {
    const int nbk = (rr + 255) / 256;
    #pragma omp parallel for schedule(dynamic, 1) num_threads(g_cone_nt) if (g_cone_nt > 1 && !omp_in_parallel())
    for (int bk = nbk - 1; bk >= 0; bk--) {
        const int i0 = bk * 256, ib = rr - i0 < 256 ? rr - i0 : 256;
        SC_BL(dsyrk_)("L", "N", &ib, &kk, &alpha, W + i0, &rr, &beta, C + (size_t)i0 * rr + i0, &rr);
        if (i0 > 0) SC_BL(dgemm_)("N", "T", &ib, &i0, &kk, &alpha, W + i0, &rr, W, &rr, &beta, C + i0, &rr);
    }
}
static int cmp_dbl(const void *a, const void *b) { const double x = *(const double *)a, y = *(const double *)b; return x < y ? -1 : x > y; }
static int cmp_ll(const void *a, const void *b) { const long long x = *(const long long *)a, y = *(const long long *)b; return x < y ? -1 : x > y; }
/* (23 October 2026) the rows that fix one entry of a small cone: a row with a single entry, in a column of a cone of at
 * most NE_MAXCONE entries; one such row a cone (a second stays a row). Such a row touches no other block, so the normal
 * equations can do without it: its unknown is eliminated by its own equation, and the cone enters the rows that remain
 * through the Schur complement of its fixed entry, H_k^-1 - h h' / h_t (h = H_k^-1 e_t). CBLIB's 2013_firL1 and
 * 2013_firL2L1eps (SOCPs): 19,902 cones (t, u, v) with t = 1 by such a row, next to 1,000 dense rows; MOSEK's presolve
 * removes the rows likewise (20,902 constraints -> 1,000). With elim NULL: only counted. BRISK_CONENOFIX=1: none. */
static int ne_find_fixed(const Socp *S, char *elim, int *frow, int *fpos, double *fa) {
    if (S->nf > 0 || getenv("BRISK_CONENOFIX")) return 0;
    const int m = S->m;
    int *cnt = (int *)calloc((size_t)m + 1, sizeof(int)); if (!cnt) return 0;
    for (int p = 0; p < S->Ap[S->n]; p++) cnt[S->Ai[p]]++;
    char *own = elim ? elim : (char *)calloc((size_t)m + 1, 1);
    int nfix = 0;
    for (int kc = 0; kc < S->nq; kc++) {
        if (frow) frow[kc] = -1;
        const int d = S->q[kc]; if (d > NE_MAXCONE || d < 2) continue;
        int got = 0;
        for (int t = 0; t < d && !got; t++) { const int j = S->qs[kc] + t;
            for (int p = S->Ap[j]; p < S->Ap[j + 1]; p++) { const int i = S->Ai[p];
                if (cnt[i] == 1 && S->Ax[p] != 0.0 && !own[i]) { own[i] = 1; if (frow) { frow[kc] = i; fpos[kc] = t; fa[kc] = S->Ax[p]; } nfix++; got = 1; break; } } }
    }
    free(cnt); if (!elim) free(own);
    return nfix;
}
/* (20 October 2026, LP/SOCP) the work of the full system's factorization when more of its linear columns are dense
 * (by the rule of ne_setup) than they have rows: eliminated first, each costs l^2, and the rest is the dense block;
 * 0 when the case does not hold. The full system's analysis is then deferred and the normal equations judged
 * against this work (CBLIB 2013_wbNRL: 4.0e10 against the analysis' 4.07e10; the analysis took 15 s there). */
static double kkt_dense_work(const Socp *S) {
    if (S->nf > ne_maxfree_ne() || getenv("BRISK_CONENODLIN") || getenv("BRISK_CONENODEFER") || getenv("BRISK_CONENONE")) return 0;
    const int m = S->m; char *mk = (char *)calloc((size_t)m + 1, 1); int md = 0;
    for (int k = 0; k < S->nl; k++) for (int p = S->Ap[S->nf + k]; p < S->Ap[S->nf + k + 1]; p++) if (!mk[S->Ai[p]]) { mk[S->Ai[p]] = 1; md++; }
    /* (27 October 2026, the SOCP developer) a large cone's columns count as the linear ones (the normal equations take the
     * cone by its low-rank form, the full system eliminates its columns by its diagonal): CBLIB's 2013_firL2a (SOCP), one cone
     * of 10,002 entries over 10,001 rows of half-dense data, analysed its full system for 25 s (6.7e11 flops) and then took
     * the normal equations (3.3e11). BRISK_CONENODEFERBIG=1: the large cones not counted, as before */
    static int dbig = -1; if (dbig < 0) dbig = getenv("BRISK_CONENODEFERBIG") == NULL;
    if (dbig) for (int kc = 0; kc < S->nq; kc++) if (S->q[kc] > NE_MAXCONE)
        for (int p = S->Ap[S->qs[kc]]; p < S->Ap[S->qs[kc] + S->q[kc]]; p++) if (!mk[S->Ai[p]]) { mk[S->Ai[p]] = 1; md++; }
    double w = 0; int nd = 0;
    for (int k = 0; k < S->nl; k++) { const double l = S->Ap[S->nf + k + 1] - S->Ap[S->nf + k]; if (l > 1 && 10.0 * l * l >= (double)md * md) { w += l * l; nd++; } }
    if (dbig) for (int kc = 0; kc < S->nq; kc++) if (S->q[kc] > NE_MAXCONE)
        for (int t = 0; t < S->q[kc]; t++) { const double l = S->Ap[S->qs[kc] + t + 1] - S->Ap[S->qs[kc] + t]; if (l > 1 && 10.0 * l * l >= (double)md * md) { w += l * l; nd++; } }
    /* (20 October 2026, LP/SOCP) the small cones with dense columns (the rule of ne_setup: a cone of at most NE_MAXCONE
     * entries over r rows with 10 r^2 >= m^2) count with their columns; the dense rows are the union */
    int mdc = 0; char *mc = (char *)calloc((size_t)m + 1, 1); int *rows = (int *)malloc(sizeof(int) * ((size_t)m + 1));
    /* (23 October 2026) the rows that fix a cone's entry leave the normal equations (ne_find_fixed): the rule against the rest */
    char *fx = (char *)calloc((size_t)m + 1, 1); const int nfx = ne_find_fixed(S, fx, NULL, NULL, NULL); const double me = (double)(m - nfx);
    if (!getenv("BRISK_CONENODCONE")) for (int kc = 0; kc < S->nq; kc++) {
        const int d = S->q[kc]; if (d > NE_MAXCONE) continue;
        const int c0 = S->qs[kc]; int r = 0, hasfx = 0;
        for (int p = S->Ap[c0]; p < S->Ap[c0 + d]; p++) { if (fx[S->Ai[p]]) { hasfx = 1; continue; } if (!mc[S->Ai[p]]) { mc[S->Ai[p]] = 1; rows[r++] = S->Ai[p]; } }
        if (r > 1 && 10.0 * r * (double)r >= me * me) { for (int t = 0; t < d; t++) { const double l = S->Ap[c0 + t + 1] - S->Ap[c0 + t]; w += l * l; } nd += d - hasfx; for (int a = 0; a < r; a++) if (!mk[rows[a]]) { mk[rows[a]] = 1; mdc++; } }
        for (int a = 0; a < r; a++) mc[rows[a]] = 0;
    }
    free(fx);
    free(mk); free(mc); free(rows);
    const int mda = md + mdc;      /* (the linear columns' rows and the dense cones' further rows) */
    return nd > 1 && nd >= mda ? w + (double)mda * mda * mda / 3.0 : 0;
}
/* returns 1 when the normal equations are set up (S->ne), 0 when they are not used, -1 (fixon only) when they are to be set up
 * again without the elimination of fixed rows */
static int ne_setup_fix(Socp *S, int fixon) {
    const int n = S->n, m = S->m, nb = S->nl + S->nq;
    if (S->nf > ne_maxfree_ne() || (double)m * S->nf * 8.0 > 2e9 * brisk_mem_scale() || nb <= 0 || getenv("BRISK_CONENONE")) return 0;     /* (18 October 2026: m < 64 no longer excluded, the cost decides: one cone of 50,001 in 50 rows) */
    /* (18 October 2026, LP/SOCP: a cone of more than NE_MAXCONE entries enters by its low-rank form,
     * A_k H_k^-1 A_k' = eta^-2 (2 v v' - A_k J A_k') with v = A_k u: A_k J A_k' (C_k) once, v every iteration. One
     * cone of 50,001 entries in 50 dense rows, after the free variables' substitution: the full system had 50,053
     * unknowns and nnz(L) 2.6e6, the normal equations are 50 x 50.) */
    /* (23 October 2026) the rows that fix a small cone's entry, eliminated (ne_find_fixed); none with the Woodbury update or
     * the sparse cones (their solves would need the elimination too: ne_setup_fix 0 then) */
    S->ne_nfix = 0;
    if (fixon) {
        S->ne_elim = (char *)calloc((size_t)m + 1, 1); S->ne_fixrow = (int *)malloc(sizeof(int) * ((size_t)S->nq + 1));
        S->ne_fixpos = (int *)malloc(sizeof(int) * ((size_t)S->nq + 1)); S->ne_fixa = (double *)malloc(sizeof(double) * ((size_t)S->nq + 1));
        S->ne_fixt = (double *)malloc(sizeof(double) * ((size_t)S->nq + 1));
        S->ne_nfix = ne_find_fixed(S, S->ne_elim, S->ne_fixrow, S->ne_fixpos, S->ne_fixa);
        if (S->ne_nfix == 0) { free(S->ne_elim); free(S->ne_fixrow); free(S->ne_fixpos); free(S->ne_fixa); free(S->ne_fixt); S->ne_elim = NULL; S->ne_fixrow = S->ne_fixpos = NULL; S->ne_fixa = S->ne_fixt = NULL; }
    }
    const char *elim = S->ne_elim;
#define NE_ELIM(i) (elim && elim[i])
    /* the rows of every block; the number of pairs */
    int *r = (int *)malloc(sizeof(int) * (size_t)nb), *mark = (int *)malloc(sizeof(int) * (size_t)m);
    for (int i = 0; i < m; i++) mark[i] = -1;
    long tot = 0; double pairs = 0;
    for (int k = 0; k < nb; k++) {
        const int c0 = k < S->nl ? S->nf + k : S->qs[k - S->nl], c1 = k < S->nl ? c0 + 1 : c0 + S->q[k - S->nl];
        int cnt = 0;
        for (int p = S->Ap[c0]; p < S->Ap[c1]; p++) if (mark[S->Ai[p]] != k && !NE_ELIM(S->Ai[p])) { mark[S->Ai[p]] = k; cnt++; }
        r[k] = cnt; tot += cnt; pairs += 0.5 * (double)cnt * (cnt + 1);
    }
    /* (20 October 2026, LP/SOCP) a large cone whose columns are sparse against its rows enters by its sparse low-rank form:
     * A_k H_k^-1 A_k' = eta^-2 (2 v v' - a_0 a_0' + sum_{t>=1} a_t a_t'), the tail columns as pairs of the pattern (as
     * linear columns) and the rank-two part [v a_0] by a Woodbury update of the factored matrix (two solves per
     * factorization); the dense C_k of r_k (r_k + 1) / 2 entries would make its rows one dense block. Taken when the tail
     * columns' pairs are below a quarter of that block. The free variables written as one cone (t, x_f) are the case
     * that matters: t has no rows, so the update has rank one; nql180 has 32,399 of them over 60,000 rows. */
    S->ne_nsp = 0; S->ne_nex = 0; S->ne_nu = 0; S->ne_ndt = 0;
    double exp_ = 0; long extot = 0; int nexc = 0;
    /* (25 October 2026, the SOCP developer) a tail column dense against the cone's rows (10 l^2 >= r_k^2, the rule of the
     * dense linear columns' block) leaves the pattern for the Woodbury update as well: its term eta^-2 a_t a_t' is one more
     * column of U with D = eta^-2 > 0. A factor-model portfolio's risk cone (s, F'x, D^1/2 x) has 50 tail columns over all
     * 10,000 rows and 10,000 of one entry: with the dense ones the cone was refused (2.5e9 pairs) and the full system
     * solved, 11.6 solves of its factor an iteration (socp_riskinf_10000x50xnear, SOCP; MOSEK 3.5 times faster); without
     * them the matrix is diagonal and the update of rank 51. BRISK_CONENODTAIL=1: the dense tail columns kept in the
     * pattern, as before */
    const int dtail = !getenv("BRISK_CONENODTAIL");
    free(S->ne_xden); S->ne_xden = dtail ? (char *)calloc((size_t)n + 1, 1) : NULL; char *xden = S->ne_xden; int ndt = 0;
    /* (26 October 2026, the SOCP developer) a dense tail column moves into the update only to save fill: where its rows lie
     * in the dense block that the dense linear columns form (the rule below, 10 l^2 >= md^2), it stays in the factored
     * matrix at no fill, and the factored matrix keeps its terms. Moved anyway, it left that matrix nearly singular late in
     * a solve, where the update's terms cover rows that the linear columns' shrinking weights no longer do: 2013_wbNRL
     * (SOCP; 1,042 rows made one dense block by 37,081 linear columns) met tiny pivots and then the update's loss of
     * accuracy over a nearly singular matrix (18 refinement steps for 5 solves), left the normal equations (3.8e8 flops a
     * factorization) for the full system (4e10, 28 s of analysis): 47 -> 79 s against the build of 19 October. The risk
     * cone of a factor model (its rows its own diagonal block) keeps the move. BRISK_CONEDTAILALL=1: every dense tail column
     * moved, as on 25 October */
    char *inblk = NULL;
    if (dtail && !getenv("BRISK_CONEDTAILALL") && !getenv("BRISK_CONENODLIN")) {
        int md0 = 0; for (int i = 0; i < m; i++) mark[i] = -1;
        for (int k = 0; k < S->nl; k++) for (int p = S->Ap[S->nf + k]; p < S->Ap[S->nf + k + 1]; p++) if (mark[S->Ai[p]] < 0) { mark[S->Ai[p]] = 0; md0++; }
        inblk = (char *)calloc((size_t)m + 1, 1);
        for (int k = 0; k < S->nl; k++) if (r[k] > 1 && 10.0 * r[k] * (double)r[k] >= (double)md0 * md0) for (int p = S->Ap[S->nf + k]; p < S->Ap[S->nf + k + 1]; p++) inblk[S->Ai[p]] = 1;
        for (int i = 0; i < m; i++) mark[i] = -1;
    }
    #define NE_TAIL_INBLK(j) (inblk && ne_col_in(S, j, inblk))
    { int nsp = 0, usp = 0; int *spk = (int *)malloc(sizeof(int) * ((size_t)S->nq + 1));
      for (int kc = 0; kc < S->nq; kc++) {
        const int d = S->q[kc], k = S->nl + kc; if (d <= NE_MAXCONE || r[k] <= 64 || getenv("BRISK_CONENOSPCONE")) continue;
        { static int sph = -1; if (sph < 0) sph = getenv("BRISK_CONESPHEAD") != NULL;   /* (27 October 2026, a test: also a cone whose head has rows) */
          if (!sph && S->Ap[S->qs[kc] + 1] > S->Ap[S->qs[kc]]) continue; }     /* (only a cone whose first entry has no rows, the free cone: its update is rank one and positive; with a_0 the rank-two update is indefinite and TV denoising's cone lost accuracy under a GMRES cap; measured again 25 October 2026 with the update's solve corrected: socp_tv_150 12 -> 30 iterations, socp_svm_10k OPTIMAL -> reduced accuracy in 51) */
        double ps = 0; long cnt = 0; int nc = 0, nden = 0;
        for (int t = 1; t < d; t++) { const double l = S->Ap[S->qs[kc] + t + 1] - S->Ap[S->qs[kc] + t];
            if (xden && l > 1 && 10.0 * l * l >= (double)r[k] * r[k] && !NE_TAIL_INBLK(S->qs[kc] + t)) { nden++; continue; }
            ps += 0.5 * l * (l + 1); cnt += (long)l; if (l > 0) nc++; }
        const double rk0 = r[k];
        /* (28 October 2026, the SOCP developer; BRISK_CONEWOODFRAC=0: no such bound) the update's rank at most a quarter of the rows
         * of the system: an update of rank comparable to its order saves nothing over the dense matrix of that order and loses
         * accuracy. A dense-margin classifier (SOCP; a cone of 193 entries over all 384 rows after the substitutions, 192 of its
         * columns dense): the update of rank 193 on 384 rows, 10 iterations as written but 33 and 24 with its rows or its
         * variables rescaled, a stall near 1e-7 and a second attempt; without the update 12 and 10 */
        static double wfrac = -1; if (wfrac < 0) wfrac = getenv("BRISK_CONEWOODFRAC") ? atof(getenv("BRISK_CONEWOODFRAC")) : 0.25;
        if (getenv("BRISK_CONEWOODSCAN") && ps < 0.25 * 0.5 * (double)r[k] * (r[k] + 1) && nsp < 8 && usp + 1 + nden <= NE_MAXWOODRANK) { printf("woodscan %s rank %d rows %d\n", usp + 1 + nden <= wfrac * m ? "kept" : "refused", usp + 1 + nden, m); fflush(stdout); if (!(usp + 1 + nden <= wfrac * m)) exit(0); }   /* (experiments) */
        if (ps < 0.25 * 0.5 * (double)r[k] * (r[k] + 1) && nsp < 8 && usp + 1 + nden <= NE_MAXWOODRANK && (wfrac <= 0 || usp + 1 + nden <= wfrac * m)) {
            spk[nsp++] = kc; usp += 1 + nden; pairs -= 0.5 * (double)r[k] * (r[k] + 1); tot -= r[k]; r[k] = 0;
            pairs += ps; exp_ += ps; extot += cnt; nexc += nc;
            if (nden) for (int t = 1; t < d; t++) { const double l = S->Ap[S->qs[kc] + t + 1] - S->Ap[S->qs[kc] + t]; if (l > 1 && 10.0 * l * l >= rk0 * rk0 && !NE_TAIL_INBLK(S->qs[kc] + t)) { xden[S->qs[kc] + t] = 1; ndt++; } }
        }
      }
      S->ne_nsp = nsp; S->ne_spk = spk; }
    #undef NE_TAIL_INBLK
    free(inblk);
    if (ndt > 0) {
        S->ne_ndt = ndt; S->ne_dtcol = (int *)malloc(sizeof(int) * (size_t)ndt); S->ne_dtk = (int *)malloc(sizeof(int) * (size_t)ndt);
        int a = 0; for (int t_ = 0; t_ < S->ne_nsp; t_++) { const int kc = S->ne_spk[t_]; for (int t = 1; t < S->q[kc]; t++) if (xden[S->qs[kc] + t]) { S->ne_dtcol[a] = S->qs[kc] + t; S->ne_dtk[a] = kc; a++; } }
    }
    /* (20 October 2026, LP/SOCP) the dense linear columns by BLAS-3: a linear column of l rows whose pairs cost, indexed,
     * at least a tenth of the dense work over all md rows of such columns (10 l^2 >= md^2, the rule of the large cones'
     * C_k) enters the matrix through dsyrk on dense panels, and its rows make one dense block of the pattern. CBLIB
     * 2013_wbNRL: 37,081 linear columns over 1,042 rows at 93 % density, 2e10 pairs. */
    double dflops = 0;
    S->ne_nd = S->ne_ndc = S->ne_md = 0; S->ne_nwl = 0;
    if (!getenv("BRISK_CONENODLIN")) {
        int md0 = 0; for (int i = 0; i < m; i++) mark[i] = -1;
        for (int k = 0; k < S->nl; k++) for (int p = S->Ap[S->nf + k]; p < S->Ap[S->nf + k + 1]; p++) if (mark[S->Ai[p]] < 0) { mark[S->Ai[p]] = 0; md0++; }
        /* (20 October 2026, LP/SOCP) small cones with dense columns join the dense block: a cone of at most NE_MAXCONE
         * entries over r rows with 10 r^2 >= m^2 (the linear columns' rule, against all rows), its columns scaled by the
         * square root R of H_k^-1 (ne_cone_sqrt; G_k H_k^-1 G_k' = (G_k R)(G_k R)'). The group lasso: 300 cones of 11 entries over
         * all 1,501 rows made 339 million pairs for a 1,501 x 1,501 matrix, so the full system (nnz(L) 5.7 million,
         * 7.9e9 flops) did the work; MOSEK's factor there is that matrix (1.13e6 entries). BRISK_CONENODCONE=1: off. */
        const int dcone = !getenv("BRISK_CONENODCONE");
        const double mrem = (double)(m - S->ne_nfix);      /* (the rows that remain: the fixed rows are not in the matrix) */
        #define NE_DENSE_BLK(k) ((k) < S->nl ? (r[k] > 1 && 10.0 * r[k] * (double)r[k] >= (double)md0 * md0) : (dcone && S->q[(k) - S->nl] <= NE_MAXCONE && r[k] > 1 && 10.0 * r[k] * (double)r[k] >= mrem * mrem))
        #define NE_BLK_COLS(k) ((k) < S->nl ? 1 : S->q[(k) - S->nl] - (S->ne_fixrow && S->ne_fixrow[(k) - S->nl] >= 0))
        int ndb = 0, ndc = 0;
        for (int k = 0; k < nb; k++) if (NE_DENSE_BLK(k)) { ndb++; ndc += NE_BLK_COLS(k); }
        if (ndc > 1) {
            for (int i = 0; i < m; i++) mark[i] = -1;
            int *dk = (int *)malloc(sizeof(int) * (size_t)ndb), *drow = (int *)malloc(sizeof(int) * (size_t)m); int md = 0, nd = 0;
            for (int k = 0; k < nb; k++) if (NE_DENSE_BLK(k)) {
                dk[nd++] = k;
                const int c0 = k < S->nl ? S->nf + k : S->qs[k - S->nl], c1 = k < S->nl ? c0 + 1 : c0 + S->q[k - S->nl];
                for (int p = S->Ap[c0]; p < S->Ap[c1]; p++) if (mark[S->Ai[p]] < 0 && !NE_ELIM(S->Ai[p])) { mark[S->Ai[p]] = md; drow[md++] = S->Ai[p]; }
            }
            if ((double)md * md * 8.0 <= 2e9 * brisk_mem_scale()) {
                for (int t = 0; t < nd; t++) { const int k = dk[t]; pairs -= 0.5 * (double)r[k] * (r[k] + 1); tot -= r[k]; r[k] = 0; }
                pairs += 0.5 * (double)md * (md + 1);
                S->ne_nd = nd; S->ne_ndc = ndc; S->ne_md = md; S->ne_dk = dk; S->ne_drow = drow;
                S->ne_dpos = (int *)malloc(sizeof(int) * (size_t)m); for (int i = 0; i < m; i++) S->ne_dpos[i] = mark[i];
                dflops = (double)md * md * ndc;
                snprintf(S->ne_dense_msg, sizeof S->ne_dense_msg, "; %d dense columns (%d blocks) over %d rows by dsyrk", ndc, nd, md);
            } else { free(dk); free(drow); }
        }
        #undef NE_DENSE_BLK
        #undef NE_BLK_COLS
        for (int i = 0; i < m; i++) mark[i] = -1;
    }
    /* (20 October 2026, LP/SOCP) when the pairs exceed the limits below, the blocks with the most rows (a linear column,
     * or a cone of at most NE_MAXCONE entries: A_k H_k^-1 A_k' = (A_k R)(A_k R)' with H_k^-1 = R R') leave the pattern
     * for the Woodbury update of the factored matrix (one solve per column of U a factorization), the most rows first,
     * while the pairs exceed the limit and the update's rank stays at most NE_MAXWOODRANK. CBLIB's FIR filters: after the presolve's
     * substitution one cone of 3 entries has a column over 14,100 rows (99.5 million pairs for a rank-3 term); the rest
     * of the normal equations are 14,100 cones of 3 rows next to the 150 free columns' Schur complement, where the full
     * system's factor has 4.5 million entries. */
    S->ne_wrank = 0;
    {
        double lim = fmin(40.0 * (double)S->Ap[n] + 1e5, 2e8 * brisk_mem_scale());
        if (!getenv("BRISK_CONENE") && !S->kkt_defer) lim = fmin(lim, 0.75 * (double)S->F.lnz);
        int nusp = 0; for (int t_ = 0; t_ < S->ne_nsp; t_++) { const int kc = S->ne_spk[t_]; nusp += 1 + (S->Ap[S->qs[kc] + 1] > S->Ap[S->qs[kc]]); }
        int rank = 0, nwl = 0; int *wl = NULL;
        while (pairs > lim && wood_on()) {
            int kb = -1;
            for (int k = 0; k < nb; k++) { if (r[k] <= 1) continue; if (k >= S->nl && S->q[k - S->nl] > NE_MAXWOODRANK) continue; if (kb < 0 || r[k] > r[kb]) kb = k; }
            if (kb < 0) break;
            const int dk = kb < S->nl ? 1 : S->q[kb - S->nl];      /* (a cone of any size up to the rank cap: the SVM's cone of 102 with 100 dense columns over 50,001 rows) */
            if (nusp + rank + dk > NE_MAXWOODRANK) break;
            if (!wl) wl = (int *)malloc(sizeof(int) * (size_t)nb);
            wl[nwl++] = kb; rank += dk; pairs -= 0.5 * (double)r[kb] * (r[kb] + 1); tot -= r[kb]; r[kb] = 0;
        }
        S->ne_nwl = nwl; S->ne_wl = wl; S->ne_wrank = rank;
        if (S->ne_nfix > 0 && (nwl > 0 || S->ne_nsp > 0)) { free(r); free(mark); ne_free(S); return -1; }   /* (again without the elimination) */
        if (nwl > 0) snprintf(S->ne_dense_msg, sizeof S->ne_dense_msg, "; %d dense block%s by a rank-%d update", nwl, nwl > 1 ? "s" : "", rank);
    }
    double cpre = 0;                  /* the work of the C_k: the products of the entries of each column (the cones of the update excluded) */
    for (int k = 0; k < S->nq; k++) if (S->q[k] > NE_MAXCONE && r[S->nl + k] > 0)
        for (int t = 0; t < S->q[k]; t++) { const double l = S->Ap[S->qs[k] + t + 1] - S->Ap[S->qs[k] + t]; cpre += l * l; }
    if (cpre > 0 && !getenv("BRISK_CONENE") && cpre > 1.5 * S->cflops + 1e6) { free(r); free(mark); ne_free(S); return 0; }
    /* (23 October 2026) the dense linear columns kept as unknowns: while the pairs exceed the route's limits, the linear
     * column with the most rows leaves the pattern and becomes an unknown of the system [S A_d; A_d' -H_d] (quasi-definite:
     * the sparse factorization with the signs; its fill is the dense columns' own, by the analysis), as MOSEK's normal
     * equations set dense columns apart and the QP method keeps them (qpipm.c). CBLIB's beam7 (SOCP): 467 columns of about
     * 31,000 entries over 218,855 rows made 6.3e10 pairs, 6e5 without them. Not with the free variables, the Woodbury
     * update or the dense block (the fixed rows are cones', the kept columns linear; the Woodbury update of the sparse cones
     * runs over the whole system). A test, off (BRISK_CONEKEEP=1): on beam7 the system's factor has nnz(L) 3.2e7 and 5.2e9
     * flops against the full system's 2.3e7 and 4.3e9 (the rows that share sparse columns carry the dense columns' entries
     * to each other); MOSEK's presolve first halves the rows (218,855 -> 111,787), then its factor has 1.5e7 entries. */
    S->ne_nkd = 0;
    {   const double limk = fmin(fmin(40.0 * (double)S->Ap[n] + 1e5, 2e8 * brisk_mem_scale()), (!getenv("BRISK_CONENE") && !S->kkt_defer) ? 0.75 * (double)S->F.lnz : 1e300);
        if (pairs > limk && S->nf == 0 && S->ne_nwl == 0 && S->ne_md == 0 && S->nl > 0 && getenv("BRISK_CONEKEEP") && atoi(getenv("BRISK_CONEKEEP")) == 1) {
            int *ord = (int *)malloc(sizeof(int) * (size_t)S->nl), no = 0;
            for (int k = 0; k < S->nl; k++) if (r[k] > 1) ord[no++] = k;
            /* (the most rows first: a counting sort by r) */
            int rmax = 0; for (int a = 0; a < no; a++) if (r[ord[a]] > rmax) rmax = r[ord[a]];
            int *cnt = (int *)calloc((size_t)rmax + 2, sizeof(int)), *srt = (int *)malloc(sizeof(int) * ((size_t)no + 1));
            for (int a = 0; a < no; a++) cnt[rmax - r[ord[a]] + 1]++;
            for (int v = 0; v <= rmax; v++) cnt[v + 1] += cnt[v];
            for (int a = 0; a < no; a++) srt[cnt[rmax - r[ord[a]]]++] = ord[a];
            free(cnt);
            double pk = pairs; long tk = tot; int nk_ = 0;
            while (pk > limk && nk_ < no && nk_ < 1000) { const int k = srt[nk_]; pk -= 0.5 * (double)r[k] * (r[k] + 1); tk -= r[k]; nk_++; }
            if (pk <= limk && nk_ > 0) {
                S->ne_kdcol = (int *)malloc(sizeof(int) * (size_t)nk_);
                for (int a = 0; a < nk_; a++) { const int k = srt[a]; S->ne_kdcol[a] = S->nf + k; r[k] = 0; }
                S->ne_nkd = nk_; pairs = pk; tot = tk;
                const size_t l_ = strlen(S->ne_dense_msg); snprintf(S->ne_dense_msg + l_, sizeof S->ne_dense_msg - l_, "; %d dense linear columns kept as unknowns", nk_);
            }
            free(ord); free(srt);
        }
    }
    if (getenv("BRISK_CONEDBG")) socp_printf("  [normal equations: %.0f pairs, nnz(A) %d, m %d%s]\n", pairs, S->Ap[n], m, S->ne_nwl > 0 || S->ne_nkd > 0 ? S->ne_dense_msg : "");
    /* (no gain to expect: no analysis). (27 October 2026, the SOCP developer) refused by work, not by entries: a factor with P
     * entries over m columns does at least P^2 / m operations (the column counts sum to P at least), so the normal
     * equations are analysed unless that bound exceeds the full system's work by half. Dense blocks have as many entries
     * either way and different work: a column-scaled least-squares SOCP of the owner's independent benchmark (1,027 rows
     * against 2,571 unknowns) had the full system's 9.0e8 flops a factorization against the normal equations' 3.6e8,
     * and the time ratio to MOSEK grew with the size (2.5 at 1,024 columns, 6.6 at 6,144). BRISK_CONENEBYNNZ=1: as before */
    { static int bynnz = -1; if (bynnz < 0) bynnz = getenv("BRISK_CONENEBYNNZ") != NULL;
      /* (only for a pattern at least a quarter dense, where entries say nothing about work: for a sparse one the bound is
       * weak and the analysis of normal equations that then lose cost more than it saves; a feeder relaxation, SOCP: 0.065
       * -> 0.092 s with the bound for every pattern) */
      const int dense_ = pairs >= 0.125 * (double)m * ((double)m + 1.0);
      if (!getenv("BRISK_CONENE") && !S->kkt_defer && pairs > 0.75 * (double)S->F.lnz && (bynnz || !dense_ || pairs * pairs / fmax((double)m, 1.0) > 1.5 * S->cflops + 1e6)) { free(r); free(mark); ne_free(S); return 0; } }
    if (pairs > 40.0 * (double)S->Ap[n] + 1e5 || pairs > 2e8 * brisk_mem_scale()) { free(r); free(mark); ne_free(S); return 0; }        /* dense columns */
    if (S->ne_nsp > 0) {           /* the sparse cones' tail columns, each with its sorted distinct rows */
        S->ne_excol = (int *)malloc(sizeof(int) * ((size_t)nexc + 1)); S->ne_exk = (int *)malloc(sizeof(int) * ((size_t)nexc + 1));
        S->ne_exr = (int *)malloc(sizeof(int) * ((size_t)nexc + 1)); S->ne_exrows = (int **)malloc(sizeof(int *) * ((size_t)nexc + 1));
        int *xpool = (int *)malloc(sizeof(int) * (size_t)(extot + 1)); long o = 0; int e = 0;
        for (int i = 0; i < m; i++) mark[i] = -1;
        for (int t_ = 0; t_ < S->ne_nsp; t_++) {
            const int kc = S->ne_spk[t_], d = S->q[kc];
            for (int t = 1; t < d; t++) {
                const int j = S->qs[kc] + t; if (S->Ap[j + 1] == S->Ap[j] || (xden && xden[j])) continue;
                S->ne_excol[e] = j; S->ne_exk[e] = kc; S->ne_exrows[e] = xpool + o; int cnt = 0;
                for (int p = S->Ap[j]; p < S->Ap[j + 1]; p++) if (mark[S->Ai[p]] != e) { mark[S->Ai[p]] = e; S->ne_exrows[e][cnt++] = S->Ai[p]; }
                int *rw = S->ne_exrows[e];
                for (int a = 1; a < cnt; a++) { const int v = rw[a]; int b_ = a; while (b_ > 0 && rw[b_ - 1] > v) { rw[b_] = rw[b_ - 1]; b_--; } rw[b_] = v; }
                S->ne_exr[e] = cnt; o += cnt; e++;
            }
        }
        S->ne_nex = e;
        if (e == 0) { free(xpool); S->ne_exrows[0] = NULL; }     /* (ne_free frees the pool through the first entry) */
    }
    int **rows = (int **)malloc(sizeof(int *) * (size_t)nb), *pool = (int *)malloc(sizeof(int) * (size_t)(tot + 1));
    for (int i = 0; i < m; i++) mark[i] = -1;
    { long o = 0;
      for (int k = 0; k < nb; k++) {
        const int c0 = k < S->nl ? S->nf + k : S->qs[k - S->nl], c1 = k < S->nl ? c0 + 1 : c0 + S->q[k - S->nl];
        rows[k] = pool + o; int cnt = 0;
        if (r[k] == 0) continue;                   /* (an empty column, or a dense one: its rows are in the dense block) */
        for (int p = S->Ap[c0]; p < S->Ap[c1]; p++) if (mark[S->Ai[p]] != k && !NE_ELIM(S->Ai[p])) { mark[S->Ai[p]] = k; rows[k][cnt++] = S->Ai[p]; }
        for (int a = 1; a < cnt; a++) { const int v = rows[k][a]; int b_ = a; while (b_ > 0 && rows[k][b_ - 1] > v) { rows[k][b_] = rows[k][b_ - 1]; b_--; } rows[k][b_] = v; }
        o += cnt;
      } }
    /* the pattern: the distinct pairs */
    const size_t np = pairs > (double)tot ? (size_t)(pairs - (double)tot) : 0;   /* off-diagonal pairs with repetitions */
    long long *key = (long long *)malloc(sizeof(long long) * (np + 1));
    long q = 0;
    for (int k = 0; k < nb; k++) for (int a = 1; a < r[k]; a++) for (int b_ = 0; b_ < a; b_++) key[q++] = ((long long)rows[k][a] << 32) | (unsigned)rows[k][b_];
    for (int e_ = 0; e_ < S->ne_nex; e_++) { const int *rw = S->ne_exrows[e_]; for (int a = 1; a < S->ne_exr[e_]; a++) for (int b_ = 0; b_ < a; b_++) key[q++] = ((long long)rw[a] << 32) | (unsigned)rw[b_]; }
    for (int a = 1; a < S->ne_md; a++) for (int b_ = 0; b_ < a; b_++) {
        const int i = S->ne_drow[a], j = S->ne_drow[b_];
        key[q++] = ((long long)(i > j ? i : j) << 32) | (unsigned)(i > j ? j : i);
    }
    qsort(key, (size_t)q, sizeof(long long), cmp_ll);
    long u = 0;
    for (long e = 0; e < q; e++) if (e == 0 || key[e] != key[e - 1]) key[u++] = key[e];
    const int mm = m + S->ne_nkd;          /* (the rows, then the dense linear columns kept as unknowns) */
    int *deg = (int *)calloc((size_t)mm + 1, sizeof(int)), **nbr = (int **)malloc(sizeof(int *) * ((size_t)mm + 1));
    for (long e = 0; e < u; e++) { deg[(int)(key[e] >> 32)]++; deg[(int)(key[e] & 0xffffffffLL)]++; }
    long kdn = 0;
    for (int a = 0; a < S->ne_nkd; a++) { const int j = S->ne_kdcol[a]; for (int p = S->Ap[j]; p < S->Ap[j + 1]; p++) { deg[S->Ai[p]]++; deg[m + a]++; kdn++; } }
    int *npool = (int *)malloc(sizeof(int) * (size_t)(2 * u + 2 * kdn + 1));
    { long o = 0; for (int i = 0; i < mm; i++) { nbr[i] = npool + o; o += deg[i]; deg[i] = 0; } }
    for (long e = 0; e < u; e++) { const int i = (int)(key[e] >> 32), j = (int)(key[e] & 0xffffffffLL); nbr[i][deg[i]++] = j; nbr[j][deg[j]++] = i; }
    for (int a = 0; a < S->ne_nkd; a++) { const int j = S->ne_kdcol[a], d_ = m + a; for (int p = S->Ap[j]; p < S->Ap[j + 1]; p++) { const int i = S->Ai[p]; nbr[d_][deg[d_]++] = i; nbr[i][deg[i]++] = d_; } }
    free(key);
    long npair_ = 0; for (int i = 0; i < mm; i++) npair_ += deg[i];
    const int om = schol_set_amd(2); const double oam = schol_set_amalg(cone_amalg(mm > 0 ? 0.5 * (double)npair_ / mm : 0.0));
    SChol *sc = schol_analyze_adj(mm, deg, nbr, (size_t)1 << 31);
    schol_set_amd(om); schol_set_amalg(oam);
    free(deg); free(nbr); free(npool); free(mark);
    if (!sc) { free(r); free(pool); free(rows); ne_free(S); return 0; }
    /* worth it against the factorization of the full system? (work of the two, the same unit; the dense columns' panels
     * counted as the factorization's flops are, both BLAS-3) */
    /* (20 October 2026, LP/SOCP) the free columns' and the Woodbury columns' solves, and the Schur complement's products */
    const double fw = (wood_on() ? ((double)S->nf + S->ne_wrank) * 2.0 * (double)schol_nnz(sc) + (double)S->nf * (double)S->Ap[S->nf] + (double)S->ne_wrank * S->ne_wrank * (double)m : 0.0)
                      + (double)S->ne_ndt * 2.0 * (double)schol_nnz(sc) + (double)S->ne_ndt * S->ne_ndt * (double)m;   /* (the dense tail columns' solves and U'W) */
    if (!getenv("BRISK_CONENE") && schol_flops(sc) + dflops + fw > 1.5 * S->cflops + 1e6) { schol_free(sc); free(r); free(pool); free(rows); ne_free(S); return 0; }
    S->ne = sc; S->ne_r = r; S->ne_rows = rows; S->ne_lnz = (long)schol_nnz(sc); S->ne_mm = mm;
    if (dbg_on()) socp_printf("  [normal equations' factor: %d unknowns, nnz(L) %.3g, %.3g flops (the full system's %.3g)]\n", mm, (double)schol_nnz(sc), schol_flops(sc), S->cflops);
    /* (20 October 2026, LP/SOCP) the free columns as a dense m x nf matrix when they are dense enough, for the Schur
     * complement's product T = A_f' W by dgemm (the FIR filters: 151 columns of 14,100 rows, 1.6e8 products a factorization) */
    free(S->ne_Af); S->ne_Af = NULL;
    if (S->nf > NE_MAXFREE && (double)S->Ap[S->nf] >= 0.25 * (double)m * S->nf && !getenv("BRISK_CONENEAFLOOP")) {
        S->ne_Af = (double *)calloc((size_t)m * S->nf, sizeof(double));
        if (S->ne_Af) for (int f = 0; f < S->nf; f++) for (int p = S->Ap[f]; p < S->Ap[f + 1]; p++) S->ne_Af[(size_t)f * m + S->Ai[p]] += S->Ax[p];
    }
    S->ne_off = (size_t *)malloc(sizeof(size_t) * (size_t)(pairs + 1)); S->ne_doff = (size_t *)malloc(sizeof(size_t) * (size_t)m);
    size_t o = 0; int bad = 0; long gsz = 0;
    for (int k = 0; k < nb; k++) {
        for (int a = 0; a < r[k]; a++) for (int b_ = 0; b_ <= a; b_++) { const size_t f = schol_offset(sc, (size_t)rows[k][a], (size_t)rows[k][b_]); if (f == (size_t)-1) bad = 1; S->ne_off[o++] = f; }
        if (k >= S->nl && S->q[k - S->nl] <= NE_MAXCONE) gsz += (long)r[k] * S->q[k - S->nl];
    }
    for (int e_ = 0; e_ < S->ne_nex; e_++) { const int *rw = S->ne_exrows[e_]; for (int a = 0; a < S->ne_exr[e_]; a++) for (int b_ = 0; b_ <= a; b_++) { const size_t f = schol_offset(sc, (size_t)rw[a], (size_t)rw[b_]); if (f == (size_t)-1) bad = 1; S->ne_off[o++] = f; } }
    for (int i = 0; i < m; i++) { S->ne_doff[i] = schol_offset(sc, (size_t)i, (size_t)i); if (S->ne_doff[i] == (size_t)-1) bad = 1; }
    if (S->ne_nkd > 0) {         /* the kept columns: signs (rows +, columns -), their diagonals' and A_d's places */
        signed char *sg = (signed char *)malloc((size_t)mm + 1); for (int i = 0; i < mm; i++) sg[i] = i < m ? 1 : -1;
        schol_set_signs(sc, sg); free(sg);
        S->ne_kdoff = (size_t *)malloc(sizeof(size_t) * (size_t)S->ne_nkd); S->ne_kaoff = (size_t *)malloc(sizeof(size_t) * ((size_t)S->Ap[n] + 1));
        for (int a = 0; a < S->ne_nkd; a++) {
            const int j = S->ne_kdcol[a]; S->ne_kdoff[a] = schol_offset(sc, (size_t)(m + a), (size_t)(m + a)); if (S->ne_kdoff[a] == (size_t)-1) bad = 1;
            for (int p = S->Ap[j]; p < S->Ap[j + 1]; p++) { S->ne_kaoff[p] = schol_offset(sc, (size_t)(m + a), (size_t)S->Ai[p]); if (S->ne_kaoff[p] == (size_t)-1) bad = 1; }
        }
        double *sw = (double *)realloc(S->swork2, sizeof(double) * ((size_t)mm + 8)); if (sw) S->swork2 = sw; else bad = 1;
    }
    S->ne_nu0 = 0; S->ne_ntf = 0; S->ne_wcap = 0;
    if (S->ne_nsp > 0 || S->ne_nwl > 0) {         /* the Woodbury columns: v and a_0 of each sparse cone (a_0 only when it has rows), the dense linear columns */
        int nu = S->ne_wrank + S->ne_ndt; for (int t_ = 0; t_ < S->ne_nsp; t_++) { const int kc = S->ne_spk[t_]; nu += 1 + (S->Ap[S->qs[kc] + 1] > S->Ap[S->qs[kc]]); }
        S->ne_nu = S->ne_nu0 = nu; const int nuc = nu; S->ne_wcap = nuc;     /* (the tiny pivots' columns get room when they occur, ne_factor) */
        S->ne_wU = (double *)malloc(sizeof(double) * (size_t)mm * nuc); S->ne_wW = (double *)malloc(sizeof(double) * (size_t)mm * nuc);
        S->ne_wC = (double *)malloc(sizeof(double) * (size_t)nuc * nuc); S->ne_wpiv = (int *)malloc(sizeof(int) * (size_t)nuc);
        if (dbg_on()) socp_printf("  [normal equations: %d large cones by their sparse form, %d tail columns (%d dense ones in the update), %d dense blocks, a Woodbury update of rank %d]\n", S->ne_nsp, S->ne_nex, S->ne_ndt, S->ne_nwl, nu);
    }
    if (S->ne_md > 0) {          /* the dense block's lower triangle, by its columns */
        const int md = S->ne_md; S->ne_doffd = (size_t *)malloc(sizeof(size_t) * ((size_t)md * (md + 1) / 2));
        size_t e = 0;
        for (int b_ = 0; b_ < md; b_++) for (int a = b_; a < md; a++) {
            const int i = S->ne_drow[a], j = S->ne_drow[b_];
            const size_t f = schol_offset(sc, (size_t)(i > j ? i : j), (size_t)(i > j ? j : i)); if (f == (size_t)-1) bad = 1; S->ne_doffd[e++] = f;
        }
    }
    if (bad) { ne_free(S); return 0; }
    /* the cones' columns of A as dense r x d matrices (by rows) */
    S->ne_G = (double *)calloc((size_t)gsz + 1, sizeof(double));
    int *posr = (int *)malloc(sizeof(int) * (size_t)m);
    long g = 0;
    long csz = 0;
    for (int k = S->nl; k < nb; k++) if (S->q[k - S->nl] > NE_MAXCONE && r[k] > 0) csz += (long)r[k] * (r[k] + 1) / 2;
    if (csz) S->ne_C = (double *)calloc((size_t)csz, sizeof(double));
    long cpos = 0;
    for (int k = S->nl; k < nb; k++) {
        const int d = S->q[k - S->nl], c0 = S->qs[k - S->nl];
        for (int a = 0; a < r[k]; a++) posr[rows[k][a]] = a;
        if (r[k] == 0) continue;     /* (a sparse-mode cone: no C_k; a dense block in the Woodbury update) */
        if (d > NE_MAXCONE) {          /* C_k = sum_t J_tt a_t a_t', packed as the pairs (a >= b) */
            double *C = S->ne_C + cpos;
            /* 19 October 2026 (LP/SOCP): by dsyrk on dense panels of the columns when the dense work r^2 d is at most ten
             * times the scattered loop's sum of l_t^2 (BLAS-3 against an indexed update) and the r x r buffer fits: a cone
             * of 10,002 entries over 10,001 dense rows (CBLIB 2013_firL2a) took 417 s by the loop */
            double lsum = 0; for (int t = 0; t < d; t++) { const double l = S->Ap[c0 + t + 1] - S->Ap[c0 + t]; lsum += l * l; }
            const double rk = r[k];
            double *Cf = NULL, *Wp = NULL; const int wp = 256;
            if (!getenv("BRISK_CONENOSYRK") && rk * rk * (double)d <= 10.0 * lsum && rk * rk * 8.0 <= 2e9 * brisk_mem_scale()) {
                Cf = (double *)calloc((size_t)r[k] * (size_t)r[k], sizeof(double)); Wp = (double *)malloc(sizeof(double) * (size_t)r[k] * wp);
            }
            if (Cf && Wp) {
                const int rr = r[k]; const double one = 1.0, mone = -1.0, zero = 0.0; int first = 1;
                for (int t0 = 0; t0 < d; ) {
                    const int t1 = t0 == 0 ? 1 : (d - t0 < wp ? d : t0 + wp), kk = t1 - t0;       /* (the first column apart: J_00 = +1) */
                    memset(Wp, 0, sizeof(double) * (size_t)rr * kk);
                    for (int t = t0; t < t1; t++) { double *wc = Wp + (size_t)(t - t0) * rr; for (int p = S->Ap[c0 + t]; p < S->Ap[c0 + t + 1]; p++) wc[posr[S->Ai[p]]] += S->Ax[p]; }
                    /* the lower triangle by row blocks of 256 (disjoint parts of Cf, the same bits at any thread count):
                     * the block's diagonal tile by dsyrk, the rest of its rows by dgemm */
                    syrk_blocks(rr, kk, t0 == 0 ? one : mone, Wp, first ? zero : one, Cf);
                    first = 0; t0 = t1;
                }
                for (int a = 0; a < rr; a++) for (int b_ = 0; b_ <= a; b_++) C[(long)a * (a + 1) / 2 + b_] = Cf[(size_t)b_ * rr + a];
                free(Cf); free(Wp);
                cpos += (long)r[k] * (r[k] + 1) / 2;
                continue;
            }
            free(Cf); free(Wp);
            for (int t = 0; t < d; t++) {
                const double jt = t == 0 ? 1.0 : -1.0;
                for (int p = S->Ap[c0 + t]; p < S->Ap[c0 + t + 1]; p++) {
                    const long ra = posr[S->Ai[p]]; const double xa = jt * S->Ax[p];
                    for (int p2 = S->Ap[c0 + t]; p2 < S->Ap[c0 + t + 1]; p2++) { const int rb = posr[S->Ai[p2]]; if (rb <= ra) C[ra * (ra + 1) / 2 + rb] += xa * S->Ax[p2]; }
                }
            }
            cpos += (long)r[k] * (r[k] + 1) / 2;
            continue;
        }
        for (int t = 0; t < d; t++) for (int p = S->Ap[c0 + t]; p < S->Ap[c0 + t + 1]; p++) if (!NE_ELIM(S->Ai[p])) S->ne_G[g + (long)posr[S->Ai[p]] * d + t] += S->Ax[p];
        g += (long)r[k] * d;
    }
    free(posr);
    if (S->ne_nfix > 0) { const size_t l_ = strlen(S->ne_dense_msg); snprintf(S->ne_dense_msg + l_, sizeof S->ne_dense_msg - l_, "; %d rows fixing a cone's entry eliminated", S->ne_nfix); }
#undef NE_ELIM
    return 1;
}
/* (23 October 2026) the normal equations with the rows that fix a cone's entry eliminated, else (the Woodbury update or the
 * sparse cones needed) without */
static int ne_setup(Socp *S) {
    const int rc = ne_setup_fix(S, 1);
    return rc >= 0 ? rc : ne_setup_fix(S, 0);
}
/* assembles and factors; 0: done, 1: failed */
static int ne_factor(Socp *S) {
    const int m = S->m, nb = S->nl + S->nq;
    schol_zero(S->ne);
    double *pm = schol_values(S->ne);
    const size_t *off = S->ne_off;
    for (int k = 0; k < S->nl; k++) {
        const int j = S->nf + k, p0 = S->Ap[j], rk = S->ne_r[k];
        const double ih = 1.0 / S->h[k];
        if (rk == S->Ap[j + 1] - p0) {                    /* (rows sorted and distinct: the column itself) */
            for (int a = 0; a < rk; a++) { const double va = S->Ax[p0 + a] * ih; for (int b_ = 0; b_ <= a; b_++) pm[*off++] += va * S->Ax[p0 + b_]; }
        } else off += (size_t)rk * (rk + 1) / 2;
    }
    if (S->ne_md > 0) {          /* the dense blocks: linear columns a_j a_j' / h_j and small cones (G_k R)(G_k R)', by dsyrk on panels of 1,024 columns */
        const int md = S->ne_md, nd = S->ne_nd, wp = 1024;
        double *Cf = (double *)malloc(sizeof(double) * (size_t)md * md), *Wp = (double *)malloc(sizeof(double) * (size_t)md * wp);
        if (!Cf || !Wp) { free(Cf); free(Wp); return 1; }
        int c = 0, first = 1;
        for (int t = 0; t < nd; t++) {
            const int k = S->ne_dk[t], fxk = k >= S->nl && S->ne_fixrow && S->ne_fixrow[k - S->nl] >= 0, dk_ = k < S->nl ? 1 : S->q[k - S->nl] - fxk;
            if (c + dk_ > wp) { syrk_blocks(md, c, 1.0, Wp, first ? 0.0 : 1.0, Cf); first = 0; c = 0; }
            if (k < S->nl) {
                const int j = S->nf + k; const double sh = 1.0 / sqrt(S->h[k]); double *wc = Wp + (size_t)c * md;
                for (int i = 0; i < md; i++) wc[i] = 0.0;
                for (int p = S->Ap[j]; p < S->Ap[j + 1]; p++) wc[S->ne_dpos[S->Ai[p]]] += S->Ax[p] * sh;
                c++; continue;
            }
            const int kc = k - S->nl, d = S->q[kc], s_ = S->qs[kc];
            double R[NE_MAXCONE * NE_MAXCONE]; if (ne_cone_sqrt(S, kc, R)) { free(Cf); free(Wp); return 1; }
            if (fxk) {
                /* (23 October 2026) the fixed entry t eliminated: H_k^-1 - h h' / h_t = R (I - q q' / q'q) R, q = R e_t; with the
                 * reflection Q = I - 2 v v' / v'v (v = q + sign(q_0) |q| e_0, Q q = -sign(q_0) |q| e_0), the columns 1 .. d-1
                 * of R Q span it: R Q e_c = R e_c - (2 v_c / v'v) R v. The fixed row's entries are not in the block. */
                const int tf = S->ne_fixpos[kc]; double qv[NE_MAXCONE], Rv[NE_MAXCONE], nq = 0, vv = 0, Bq[NE_MAXCONE * NE_MAXCONE];
                for (int a = 0; a < d; a++) { qv[a] = R[a * d + tf]; nq += qv[a] * qv[a]; }
                nq = sqrt(nq); qv[0] += qv[0] >= 0 ? nq : -nq;
                for (int a = 0; a < d; a++) vv += qv[a] * qv[a];
                for (int a = 0; a < d; a++) { double v_ = 0; for (int b_ = 0; b_ < d; b_++) v_ += R[a * d + b_] * qv[b_]; Rv[a] = v_; }
                for (int cc = 1; cc < d; cc++) { const double f = vv > 0 ? 2.0 * qv[cc] / vv : 0.0; for (int a = 0; a < d; a++) Bq[a * d + cc] = R[a * d + cc] - f * Rv[a]; }
                for (int cc = 1; cc < d; cc++) {
                    double *wc = Wp + (size_t)(c + cc - 1) * md; for (int i = 0; i < md; i++) wc[i] = 0.0;
                    for (int t2 = 0; t2 < d; t2++) { const double rt = Bq[t2 * d + cc]; if (rt == 0.0) continue; for (int p = S->Ap[s_ + t2]; p < S->Ap[s_ + t2 + 1]; p++) if (!S->ne_elim[S->Ai[p]]) wc[S->ne_dpos[S->Ai[p]]] += rt * S->Ax[p]; }
                }
                c += d - 1;
                continue;
            }
            for (int cc = 0; cc < d; cc++) {           /* column cc of G_k R = sum_t R[t][cc] a_t (R symmetric) */
                double *wc = Wp + (size_t)(c + cc) * md; for (int i = 0; i < md; i++) wc[i] = 0.0;
                for (int t2 = 0; t2 < d; t2++) { const double rt = R[t2 * d + cc]; if (rt == 0.0) continue; for (int p = S->Ap[s_ + t2]; p < S->Ap[s_ + t2 + 1]; p++) wc[S->ne_dpos[S->Ai[p]]] += rt * S->Ax[p]; }
            }
            c += d;
        }
        if (c > 0) syrk_blocks(md, c, 1.0, Wp, first ? 0.0 : 1.0, Cf);
        size_t e = 0;
        for (int b_ = 0; b_ < md; b_++) for (int a = b_; a < md; a++) pm[S->ne_doffd[e++]] += Cf[(size_t)b_ * md + a];
        free(Cf); free(Wp);
    }
    const double *G = S->ne_G;
    double Hi[NE_MAXCONE * NE_MAXCONE], T[NE_MAXCONE];
    const double *Cb = S->ne_C;
    double *vb = NULL; int *posb = NULL;
    for (int k = S->nl; k < nb; k++) {
        const int kc = k - S->nl, d = S->q[kc], s = S->qs[kc], rk = S->ne_r[k];
        const double *w = S->wbar + s; const double ie2 = 1.0 / (S->eta[kc] * S->eta[kc]);
        if (rk == 0) continue;       /* (a sparse-mode cone: its tail columns and the update below; a dense block in the update) */
        if (d > NE_MAXCONE) {          /* eta^-2 (2 v v' - C_k), v = A_k u, u = (w0, -w1) */
            if (!vb) { vb = (double *)malloc(sizeof(double) * (size_t)m); posb = (int *)malloc(sizeof(int) * (size_t)m); }
            for (int a = 0; a < rk; a++) { posb[S->ne_rows[k][a]] = a; vb[a] = 0.0; }
            for (int t = 0; t < d; t++) {
                const double ut = t == 0 ? w[0] : -w[t];
                for (int p = S->Ap[s + t]; p < S->Ap[s + t + 1]; p++) vb[posb[S->Ai[p]]] += ut * S->Ax[p];
            }
            for (int a = 0; a < rk; a++) { const double va = 2.0 * vb[a]; for (int b_ = 0; b_ <= a; b_++) pm[*off++] += ie2 * (va * vb[b_] - *Cb++); }
            continue;
        }
        for (int i = 0; i < d; i++) for (int j = 0; j < d; j++) {
            const double ui = i == 0 ? w[0] : -w[i], uj = j == 0 ? w[0] : -w[j];
            Hi[i * d + j] = ie2 * (2.0 * ui * uj - (i == j ? (i == 0 ? 1.0 : -1.0) : 0.0));
        }
        if (S->ne_fixrow && S->ne_fixrow[kc] >= 0) {   /* (23 October 2026) the fixed entry t eliminated: H^-1 - h h' / h_t, h = H^-1 e_t */
            const int tf = S->ne_fixpos[kc]; double hc[NE_MAXCONE]; const double htt = Hi[tf * d + tf];
            for (int i = 0; i < d; i++) hc[i] = Hi[i * d + tf];
            for (int i = 0; i < d; i++) for (int j = 0; j < d; j++) Hi[i * d + j] -= hc[i] * hc[j] / htt;
        }
        for (int a = 0; a < rk; a++) {
            const double *ga = G + (size_t)a * d;
            for (int j = 0; j < d; j++) { double t = 0; for (int i = 0; i < d; i++) t += ga[i] * Hi[i * d + j]; T[j] = t; }
            for (int b_ = 0; b_ <= a; b_++) { const double *gb = G + (size_t)b_ * d; double v = 0; for (int j = 0; j < d; j++) v += T[j] * gb[j]; pm[*off++] += v; }
        }
        G += (size_t)rk * d;
    }
    free(vb); free(posb);
    for (int e_ = 0; e_ < S->ne_nex; e_++) {         /* a sparse cone's tail column: eta^-2 a_t a_t' */
        const int j = S->ne_excol[e_], kc = S->ne_exk[e_], rk = S->ne_exr[e_]; const double ie2 = 1.0 / (S->eta[kc] * S->eta[kc]);
        if (rk == S->Ap[j + 1] - S->Ap[j]) {
            const int p0 = S->Ap[j];
            for (int a = 0; a < rk; a++) { const double va = S->Ax[p0 + a] * ie2; for (int b_ = 0; b_ <= a; b_++) pm[*off++] += va * S->Ax[p0 + b_]; }
        } else {                                      /* (repeated rows: gathered on the sorted list) */
            double *vb_ = S->w4; const int *rw = S->ne_exrows[e_];
            for (int a = 0; a < rk; a++) vb_[a] = 0.0;
            for (int p = S->Ap[j]; p < S->Ap[j + 1]; p++) { int lo = 0, hi = rk - 1; while (lo < hi) { const int mid = (lo + hi) >> 1; if (rw[mid] < S->Ai[p]) lo = mid + 1; else hi = mid; } vb_[lo] += S->Ax[p]; }
            for (int a = 0; a < rk; a++) { const double va = vb_[a] * ie2; for (int b_ = 0; b_ <= a; b_++) pm[*off++] += va * vb_[b_]; }
        }
    }
    { static int dlf = -1; if (dlf < 0) { const char *e_ = getenv("BRISK_CONENEDLF"); dlf = e_ ? atoi(e_) : 0; }   /* (test: the factor's dual regularization the operator's, min(1e-9, tol/100)) */
      const double dl_ = dlf && S->ne_dl > 0 ? fmin(S->sreg0, S->ne_dl) : S->sreg0;
      for (int i = 0; i < m; i++) pm[S->ne_doff[i]] += dl_;
      for (int a = 0; a < S->ne_nkd; a++) {          /* (23 October 2026) a kept dense column: A_d's entries and -(h_j + delta) */
          const int j = S->ne_kdcol[a];
          for (int p = S->Ap[j]; p < S->Ap[j + 1]; p++) pm[S->ne_kaoff[p]] += S->Ax[p];
          pm[S->ne_kdoff[a]] = -(S->h[j - S->nf] + S->sregx);
      }
      if (S->ne_elim) for (int i = 0; i < m; i++) if (S->ne_elim[i]) pm[S->ne_doff[i]] = 1.0; }   /* (the eliminated rows: a unit, solved apart in ne_prec) */
    { static double netp = -1; if (netp < 0) { const char *e_ = getenv("BRISK_CONENETP"); netp = e_ ? atof(e_) : 1e-14; }   /* (test: the tiny-pivot threshold; the LP method uses 1e-20) */
      schol_set_tinypiv(S->ne, netp); }
    /* (26 October 2026, the SOCP developer) a tiny pivot regularized: replaced by 1e-10 of its diagonal entry, its column kept,
     * instead of dropped (a pivot of 1e128) and counted as the failure of the normal equations, which left them for the full
     * system for good: on 2013_wbNRL (SOCP) at iteration 10, for a full system of 4e10 flops (28 s of analysis, 4 s a
     * factorization) against 3.8e8; the refinement against the unregularized system removes the change. With the leave rule
     * by cost below, 83.2 -> 59.8 s; the 209 SOCP files keep their statuses. BRISK_CONENEREG=0: the pivot dropped (eps: another
     * fraction) */
    static double nereg = -1; if (nereg < 0) { const char *e_ = getenv("BRISK_CONENEREG"); nereg = e_ ? atof(e_) : 0.0; }   /* (a test: a tiny pivot replaced by nereg times its diagonal and left to the refinement, BRISK_CONENEREG=1e-10) */
    /* (26 October 2026, the SOCP developer) the tiny pivots corrected exactly (Andersen's treatment of dense columns, 1996).
     * When the Woodbury update carries the large cones' rank-one terms and their dense tail columns (ne_setup, 25 October),
     * the factored part alone can lose rank late in a solve, where the update's terms cover rows that the linear columns'
     * shrinking weights no longer do: 2013_wbNRL (SOCP) met 4 tiny pivots at iteration 10 and left the normal equations
     * (3.8e8 flops a factorization) for the full system (4e10 and 28 s of analysis): 47 -> 79 s against the build of 19
     * October. A pivot below the threshold at row i is raised to the row's own diagonal M_ii (raising a pivot by Delta adds
     * Delta to M_ii, since later steps leave earlier pivots alone), so the factored matrix is M + Delta e_i e_i'; the column
     * e_i with weight -Delta in the Woodbury update gives back M exactly (Delta = M_ii - d, d below 1e-13 M_ii: M_ii to the
     * factorization's rounding). At most NE_TINYMAX of them; more, and the normal equations are left as before.
     * BRISK_CONENOTINYFIX=1: off (the first tiny pivot leaves them) */
    static int tfix = -1; if (tfix < 0) tfix = getenv("BRISK_CONENOTINYFIX") == NULL;
    /* (27 October 2026) also without a Woodbury update of its own: the update is then made of the corrections alone, its
     * arrays created when the first tiny pivot occurs. db-plate-yield-line-fox (SOCP): one tiny pivot at iteration 23 left
     * the normal equations (solves to 1e-13) for the full system, whose solves stopped at 1e-4 to 1e-5 with the
     * regularization raised to 1e-4; the dual residual stalled at 2e-6 and the solve at the iteration limit.
     * BRISK_CONETINYFIXWOOD=1: only with an update of its own, as before */
    static int tfw = -1; if (tfw < 0) tfw = getenv("BRISK_CONETINYFIXWOOD") != NULL;
    const int tf_on = tfix && nereg <= 0 && !S->use_dd && (S->ne_wU != NULL || !tfw) && !S->ne_depmode;
    if (tf_on) {
        const int mm_ = S->ne_mm > 0 ? S->ne_mm : m;
        if (!S->ne_dg) S->ne_dg = (double *)malloc(sizeof(double) * ((size_t)mm_ + 1));
        for (int i = 0; i < mm_; i++) S->ne_dg[i] = 0.0;
        for (int i = 0; i < m; i++) S->ne_dg[i] = pm[S->ne_doff[i]];
        for (int a = 0; a < S->ne_nkd; a++) S->ne_dg[m + a] = pm[S->ne_kdoff[a]];
        schol_set_tinyflags(S->ne, 1); schol_set_tinyreg(1.0);
    }
    if (nereg > 0) schol_set_tinyreg(nereg);
    const int bad = schol_factor(S->ne, 0.0);
    if (nereg > 0 || tf_on) schol_set_tinyreg(0.0);
    const int nt = (int)schol_ntiny(S->ne);
    S->ne_ntf = 0; if (S->ne_wU) S->ne_nu = S->ne_nu0;
    int ndrop_ = 0;     /* (the tiny pivots on rows dependent to working precision, kept raised without a correction, below) */
    if (tf_on && nt > 0 && bad == 0) {
        int rows[NE_TINYMAX + 1]; const int k_ = schol_tiny_rows(S->ne, rows, NE_TINYMAX + 1);
        int room = S->ne_nu0 + k_ <= S->ne_wcap;
        if (!room && k_ == nt && k_ <= NE_TINYMAX && S->ne_nu0 + k_ <= NE_MAXWOODRANK) {      /* (the update's arrays grown to the new rank) */
            const int nuc = S->ne_nu0 + k_; const size_t mmz = (size_t)(S->ne_mm > 0 ? S->ne_mm : m);
            double *U_ = (double *)realloc(S->ne_wU, sizeof(double) * mmz * nuc), *W_ = U_ ? (double *)realloc(S->ne_wW, sizeof(double) * mmz * nuc) : NULL;
            if (U_) S->ne_wU = U_;
            if (W_) S->ne_wW = W_;
            double *C_ = (U_ && W_) ? (double *)realloc(S->ne_wC, sizeof(double) * (size_t)nuc * nuc) : NULL; if (C_) S->ne_wC = C_;
            int *P_ = C_ ? (int *)realloc(S->ne_wpiv, sizeof(int) * (size_t)nuc) : NULL; if (P_) S->ne_wpiv = P_;
            if (P_) { S->ne_wcap = nuc; room = 1; }
        }
        if (room && k_ == nt && k_ <= NE_TINYMAX && S->ne_nu0 + k_ <= NE_MAXWOODRANK) {
            for (int a = 0; a < k_; a++) { S->ne_tfrow[a] = rows[a]; S->ne_tfd[a] = fabs(S->ne_dg[rows[a]]); }
            S->ne_ntf = k_; S->ne_nu = S->ne_nu0 + k_;
            /* (27 October 2026) a correction is exact only when M itself is not singular in its direction: its entry of
             * C = -1/Delta + e_i' (L D L')^-1 e_i is then of the order of 1/Delta. When it cancels to rounding (db-plate-yield-
             * line-fox, SOCP: C = 6e-30 against 1/Delta = 3e-14, the row's Schur complement zero to working precision: rows
             * numerically dependent), the raised pivot is kept without its correction, the treatment of a dependent row in an
             * interior-point method's Cholesky factor (Wright 1999): the step in that direction is left to the refinement
             * against the full system. BRISK_CONETINYDROP=0: as before (the correction kept, the normal equations then left) */
            static int tdrop = -1; if (tdrop < 0) { const char *e_ = getenv("BRISK_CONETINYDROP"); tdrop = e_ ? atoi(e_) : 0; }   /* (a test since 27 October, 13:00: qssp60 17 -> 19 iterations; the stall of db-plate-yield-line-fox was the raised regularization never lowered, BRISK_CONEBUMPDECAY) */
            if (tdrop) {
                const int mmz = S->ne_mm > 0 ? S->ne_mm : m; double *wv = (double *)malloc(sizeof(double) * ((size_t)mmz + 8)); int kk = 0;
                for (int a = 0; a < k_ && wv; a++) {
                    for (int i = 0; i < mmz; i++) wv[i] = 0.0;
                    const int r_ = S->ne_tfrow[a]; wv[r_] = 1.0;
                    schol_set_solve_seq(1); schol_solve(S->ne, wv, S->swork2); schol_set_solve_seq(0);
                    const double cd = -1.0 / S->ne_tfd[a] + wv[r_];
                    if (fabs(cd) > 1e-8 / S->ne_tfd[a]) { S->ne_tfrow[kk] = r_; S->ne_tfd[kk] = S->ne_tfd[a]; kk++; }
                }
                if (wv) { ndrop_ = k_ - kk; S->ne_ntf = kk; S->ne_nu = S->ne_nu0 + kk; }
                free(wv);
                if (ndrop_ > 0 && dbg_on()) socp_printf("      [the normal equations' factor: %d of %d tiny pivots on rows dependent to working precision: refactored with them dropped, and so for the rest of the solve]\n", ndrop_, k_);
                /* (a dependent row's pivot is dropped, not raised: raised to its diagonal, the column of L below it perturbs
                 * every row factored after it, and the refinement needed 20 steps for 3 solves (db-plate-yield-line-fox);
                 * dropped, the rest of the factor is that of M without the row) */
                if (ndrop_ > 0) { S->ne_depmode = 1; S->ne_ntf = 0; if (S->ne_wU) S->ne_nu = S->ne_nu0; return ne_factor(S); }
            }
            if (dbg_on()) socp_printf("      [the normal equations' factor: %d tiny pivots corrected through the Woodbury update (rank %d)]\n", S->ne_ntf, S->ne_nu);
        }
    }
    if (bad && dbg_on()) socp_printf("      [the normal equations' factor: schol_factor returned %d (%d tiny pivots in it)]\n", bad, nt);
    S->ne_tiny += nt;
    S->ne_nfact++;
    if (bad == 0 && S->ne_nu > 0) {
        /* the Woodbury update: M = S + U D U', D = diag(2 eta^-2 | -eta^-2) over the columns v | a_0 of each sparse cone;
         * W = S^-1 U, C = D^-1 + U' W (nu x nu, factored by Gaussian elimination with pivoting), M^-1 r = S^-1 r - W C^-1 U' S^-1 r */
        const int nu = S->ne_nu, mm = S->ne_mm; double *U = S->ne_wU, *W = S->ne_wW, *C = S->ne_wC; int c = 0;   /* (stride mm: zero on the kept columns' unknowns) */
        double dinv[NE_MAXWOODRANK];
        for (int t_ = 0; t_ < S->ne_nsp; t_++) {
            const int kc = S->ne_spk[t_], d = S->q[kc], s_ = S->qs[kc]; const double *w = S->wbar + s_; const double ie2 = 1.0 / (S->eta[kc] * S->eta[kc]);
            double *u = U + (size_t)c * mm; for (int i = 0; i < mm; i++) u[i] = 0.0;
            for (int t = 0; t < d; t++) { const double ut = t == 0 ? w[0] : -w[t]; for (int p = S->Ap[s_ + t]; p < S->Ap[s_ + t + 1]; p++) u[S->Ai[p]] += ut * S->Ax[p]; }
            dinv[c] = 1.0 / (2.0 * ie2); c++;
            if (S->Ap[s_ + 1] > S->Ap[s_]) { u = U + (size_t)c * mm; for (int i = 0; i < mm; i++) u[i] = 0.0; for (int p = S->Ap[s_]; p < S->Ap[s_ + 1]; p++) u[S->Ai[p]] += S->Ax[p]; dinv[c] = -1.0 / ie2; c++; }
        }
        for (int a = 0; a < S->ne_ndt; a++) {          /* (25 October 2026) the dense tail columns: eta^-2 a_t a_t' */
            const int j = S->ne_dtcol[a], kc = S->ne_dtk[a]; const double ie2 = 1.0 / (S->eta[kc] * S->eta[kc]);
            double *u = U + (size_t)c * mm; for (int i = 0; i < mm; i++) u[i] = 0.0;
            for (int p = S->Ap[j]; p < S->Ap[j + 1]; p++) u[S->Ai[p]] += S->Ax[p];
            dinv[c] = 1.0 / ie2; c++;
        }
        for (int t_ = 0; t_ < S->ne_nwl; t_++) {       /* the dense blocks: a linear column, a_j a_j' / h_j; a small cone, (A_k R)(A_k R)' with H_k^-1 = R R' */
            const int k = S->ne_wl[t_];
            if (k < S->nl) { const int j = S->nf + k; double *u = U + (size_t)c * mm; for (int i = 0; i < mm; i++) u[i] = 0.0; for (int p = S->Ap[j]; p < S->Ap[j + 1]; p++) u[S->Ai[p]] += S->Ax[p]; dinv[c] = S->h[k]; c++; continue; }
            const int kc = k - S->nl, d = S->q[kc], s_ = S->qs[kc];
            double *R = (double *)malloc(sizeof(double) * (size_t)d * d); if (!R || ne_cone_sqrt(S, kc, R)) { free(R); return 1; }
            for (int cc = 0; cc < d; cc++) {           /* u_cc = sum_t R[t][cc] a_t (R symmetric) */
                double *u = U + (size_t)c * mm; for (int i = 0; i < mm; i++) u[i] = 0.0;
                for (int t = 0; t < d; t++) { const double rt = R[t * d + cc]; if (rt == 0.0) continue; for (int p = S->Ap[s_ + t]; p < S->Ap[s_ + t + 1]; p++) u[S->Ai[p]] += rt * S->Ax[p]; }
                dinv[c] = 1.0; c++;
            }
            free(R);
        }
        for (int a = 0; a < S->ne_ntf; a++) {          /* (26 October 2026) the raised tiny pivots: -Delta e_i e_i' */
            double *u = U + (size_t)c * mm; for (int i = 0; i < mm; i++) u[i] = 0.0;
            u[S->ne_tfrow[a]] = 1.0; dinv[c] = -1.0 / S->ne_tfd[a]; c++;
        }
        for (int f = 0; f < nu; f++) { memcpy(W + (size_t)f * mm, U + (size_t)f * mm, sizeof(double) * (size_t)mm); schol_set_solve_seq(1); schol_solve(S->ne, W + (size_t)f * mm, S->swork2); schol_set_solve_seq(0); }
        if (nu > 16) {              /* C = U' W by dgemm (the sparse cones' update alone stays at most 16: the scalar sum, the same bits) (symmetric: the row-major reads below see the same entries), then D^-1 on the diagonal */
            const double one = 1.0, zero = 0.0; SC_BL(dgemm_)("T", "N", &nu, &nu, &mm, &one, U, &mm, W, &mm, &zero, C, &nu);
            for (int f = 0; f < nu; f++) C[f * nu + f] += dinv[f];
        } else
        for (int f = 0; f < nu; f++) for (int g = 0; g < nu; g++) { double v = 0; const double *uf = U + (size_t)f * mm, *wg = W + (size_t)g * mm; for (int i = 0; i < mm; i++) v += uf[i] * wg[i]; C[f * nu + g] = v + (f == g ? dinv[f] : 0.0); }
        for (int f = 0; f < nu; f++) {                 /* LU with partial pivoting, in place */
            int pv = f; for (int g = f + 1; g < nu; g++) if (fabs(C[g * nu + f]) > fabs(C[pv * nu + f])) pv = g;
            S->ne_wpiv[f] = pv; if (pv != f) for (int g = 0; g < nu; g++) { const double t = C[f * nu + g]; C[f * nu + g] = C[pv * nu + g]; C[pv * nu + g] = t; }
            if (C[f * nu + f] == 0.0) { if (dbg_on()) socp_printf("      [the Woodbury update's C: a zero pivot at %d of %d]\n", f, nu); return 1; }
            for (int g = f + 1; g < nu; g++) { const double l = C[g * nu + f] / C[f * nu + f]; C[g * nu + f] = l; for (int h = f + 1; h < nu; h++) C[g * nu + h] -= l * C[f * nu + h]; }
        }
        if (dbg_on() >= 2) { double pmin = HUGE_VAL, pmax = 0, dmin = HUGE_VAL, dmax = 0; for (int f = 0; f < nu; f++) { pmin = fmin(pmin, fabs(C[f * nu + f])); pmax = fmax(pmax, fabs(C[f * nu + f])); dmin = fmin(dmin, fabs(1.0 / dinv[f])); dmax = fmax(dmax, fabs(1.0 / dinv[f])); }
          double smin = HUGE_VAL, smax = 0; for (int i = 0; i < S->m; i++) { const double v_ = pm[S->ne_doff[i]]; smin = fmin(smin, v_); smax = fmax(smax, v_); }
          socp_printf("      [the Woodbury update: rank %d, |D| %.1e .. %.1e, C's pivots %.1e .. %.1e; the factored diagonal (before) %.1e .. %.1e]\n", nu, dmin, dmax, pmin, pmax, smin, smax); }
        if (dbg_on() >= 3 && S->ne_nfix == 0 && S->ne_nkd == 0) {      /* (BRISK_CONEDBG=3, a check: M^-1 (A H^-1 A' + delta I) x against x, the factor and the update against the operator) */
            const int n = S->n, m = S->m; double *x = (double *)malloc(sizeof(double) * m), *ax = (double *)malloc(sizeof(double) * n), *hx = (double *)malloc(sizeof(double) * n), *t = (double *)malloc(sizeof(double) * mm);
            for (int i = 0; i < m; i++) x[i] = 1.0 + 0.5 * sin(1.0 + i);
            for (int j = 0; j < n; j++) ax[j] = 0.0;
            At_mul(S, 1.0, x, ax);
            for (int j = 0; j < S->nf; j++) ax[j] = 0.0;
            Hinv_apply(S, ax, hx); for (int i = 0; i < mm; i++) t[i] = 0.0; A_mul(S, 1.0, hx, t);
            for (int i = 0; i < m; i++) t[i] += S->sreg0 * x[i];
            double tn = 0; for (int i = 0; i < m; i++) tn = fmax(tn, fabs(t[i]));
            schol_set_solve_seq(1); schol_solve(S->ne, t, S->swork2); schol_set_solve_seq(0); ne_wood_apply(S, t);
            double e_ = 0, xn = 0; for (int i = 0; i < m; i++) { e_ = fmax(e_, fabs(t[i] - x[i])); xn = fmax(xn, fabs(x[i])); }
            double vn = 0; { const double *u0 = U; for (int i = 0; i < mm; i++) vn = fmax(vn, fabs(u0[i])); }
            socp_printf("      [the normal equations against A H^-1 A': |M^-1 t - x| / |x| %.1e (|t| %.1e, |v| %.1e)]\n", e_ / xn, tn, vn);
            free(x); free(ax); free(hx); free(t); }
    }
    if (bad == 0 && S->nf > 0) {
        /* W = S^-1 A_f, T = A_f' W (+ a small shift) = L L' */
        const int nf = S->nf;
        if (!S->ne_W) S->ne_W = (double *)malloc(sizeof(double) * (size_t)m * nf);
        if (!S->ne_xf) S->ne_xf = (double *)malloc(sizeof(double) * ((size_t)nf + 1));
        if (!S->ne_T) S->ne_T = (double *)malloc(sizeof(double) * (size_t)nf * nf);
        for (int f = 0; f < nf; f++) {
            double *w = S->ne_W + (size_t)f * m;
            for (int i = 0; i < m; i++) w[i] = 0.0;
            for (int p = S->Ap[f]; p < S->Ap[f + 1]; p++) w[S->Ai[p]] += S->Ax[p];
            schol_set_solve_seq(1); schol_solve(S->ne, w, S->swork2); schol_set_solve_seq(0);
            if (S->ne_nu > 0) ne_wood_apply(S, w);         /* (20 October 2026, LP/SOCP: W = M^-1 A_f with the Woodbury update, not S^-1 A_f) */
        }
        double *T = S->ne_T;
        if (S->ne_Af) {            /* T = A_f' W by dgemm, in column blocks of 32 on the threads (T is symmetric: the row-major reads below see the same entries) */
            const int nbk = (nf + 31) / 32; const double one = 1.0, zero = 0.0;
            #pragma omp parallel for schedule(dynamic, 1) num_threads(g_cone_nt) if (g_cone_nt > 1 && !omp_in_parallel())
            for (int bk = 0; bk < nbk; bk++) {
                const int g0 = bk * 32, gb = nf - g0 < 32 ? nf - g0 : 32;
                SC_BL(dgemm_)("T", "N", &nf, &gb, &m, &one, S->ne_Af, &m, S->ne_W + (size_t)g0 * m, &m, &zero, T + (size_t)g0 * nf, &nf);
            }
        } else
        for (int f = 0; f < nf; f++) for (int g = 0; g <= f; g++) {
            double v = 0; const double *w = S->ne_W + (size_t)g * m;
            for (int p = S->Ap[f]; p < S->Ap[f + 1]; p++) v += S->Ax[p] * w[S->Ai[p]];
            T[f * nf + g] = v;
        }
        for (int f = 0; f < nf; f++) {
            double d = T[f * nf + f] * (1.0 + 1e-12);
            for (int k = 0; k < f; k++) d -= T[f * nf + k] * T[f * nf + k];
            if (!(d > 0)) return 1;
            d = sqrt(d); T[f * nf + f] = d;
            for (int g = f + 1; g < nf; g++) { double v = T[g * nf + f]; for (int k = 0; k < f; k++) v -= T[g * nf + k] * T[f * nf + k]; T[g * nf + f] = v / d; }
        }
    }
    { static int ktiny = -1, ktmax = 0; if (ktiny < 0) { const char *e_ = getenv("BRISK_CONENETINY"); ktiny = e_ ? atoi(e_) : 0; const char *m_ = getenv("BRISK_CONENETINYMAX"); ktmax = m_ ? atoi(m_) : 0; }   /* (tests: 1 = go on with the dropped pivots; NETINYMAX=k: go on with up to k dropped in a factorization) */
      if ((ktiny || nereg > 0) && bad == 0) return 0;
      if (nt > 0 && S->ne_ntf == nt && bad == 0) return 0;     /* (corrected exactly: above) */
      if (nt > 0 && S->ne_depmode && bad == 0) return 0;      /* (dropped on rows found dependent: above) */
      if (bad == 0 && nt <= ktmax) return 0;
      /* (27 October 2026, the SOCP developer) kept past tiny pivots when the full system costs more than twice as much a
       * factorization: the raised pivots are corrected by the refinement against the system itself, and the rule after
       * each iteration leaves the normal equations when the refinement then takes too many steps. At the apex of a cone
       * (an exact least-squares fit, the optimum t = 0: the owner's independent benchmark's column-scaled SOCPs) hundreds
       * of rows lose the cone's terms together and the full system there costs 2.5 times the normal equations
       * (L_A_scaling_4096: its two last iterations in the full system 23 s of 38.8). Off where the full system is cheap:
       * keeping them unconditionally took TV denoising from 17 to 106 iterations (20 October). BRISK_CONENETINYCOST=0: off */
      static int tcost = -1; if (tcost < 0) { const char *e_ = getenv("BRISK_CONENETINYCOST"); tcost = e_ ? atoi(e_) : 1; }
      if (tcost && bad == 0 && nt > 0 && S->cflops > 2.0 * schol_flops(S->ne) + 1e6) {
          if (dbg_on() && !S->ne_tinykept) socp_printf("      [the normal equations kept past %d tiny pivots: the full system costs %.1f times as much]\n", nt, S->cflops / fmax(schol_flops(S->ne), 1.0));
          S->ne_tinykept = 1; return 0; } }
    return bad != 0 || nt > 0;       /* (a tiny pivot: dependent rows, the product form has lost them) */
}
/* t = M^-1 (S t) for the Woodbury update M = S + U D U': t -= W C^-1 U' t (W = S^-1 U, C = D^-1 + U' W, LU-factored) */
static void ne_wood_apply(const Socp *S, double *t) {
    const int m = S->ne_mm > 0 ? S->ne_mm : S->m, nu = S->ne_nu; const double *U = S->ne_wU, *W = S->ne_wW, *C = S->ne_wC; double g[NE_MAXWOODRANK];   /* (m: the system's unknowns) */
    for (int f = 0; f < nu; f++) { double v = 0; const double *uf = U + (size_t)f * m; for (int i = 0; i < m; i++) v += uf[i] * t[i]; g[f] = v; }
    /* (25 October 2026, the SOCP developer) the factorization swaps whole rows, its multipliers too (as LAPACK's dgetf2), so
     * every interchange is applied before the forward substitution (as dgetrs): applied at each step, a later swap's
     * multipliers met an earlier order. Harmless at rank 2 (v and a_0: no swap after an elimination), wrong at rank 51
     * (a portfolio's risk cone with its dense tail columns: |M^-1 t - x| / |x| 2.9e4 at the third factorization) */
    for (int f = 0; f < nu; f++) { const int pv = S->ne_wpiv[f]; if (pv != f) { const double x_ = g[f]; g[f] = g[pv]; g[pv] = x_; } }
    for (int f = 0; f < nu; f++) for (int h = f + 1; h < nu; h++) g[h] -= C[h * nu + f] * g[f];
    for (int f = nu - 1; f >= 0; f--) { double v = g[f]; for (int h = f + 1; h < nu; h++) v -= C[f * nu + h] * g[h]; g[f] = v / C[f * nu + f]; }
    for (int f = 0; f < nu; f++) { const double *wf = W + (size_t)f * m, a_ = g[f]; for (int i = 0; i < m; i++) t[i] -= a_ * wf[i]; }
}
/* (23 October 2026) the rows that fix a cone's entry (ne_find_fixed) in a solve of the normal equations M dy = t: with S
 * those rows and R the others, M_SS is diagonal (a^2 h_t for the row a x_t = b of cone k, h = H_k^-1 e_t) and the factored
 * matrix is M_RR - M_RS M_SS^-1 M_SR (ne_factor), so  t_R -= M_RS M_SS^-1 t_S  before the solve and
 * dy_S = M_SS^-1 (t_S - M_SR dy_R)  after (ne_prec); M_RS's column of the row is a A_{R,k} h. h = H_k^-1 e_t: */
static void ne_fix_h(const Socp *S, int kc, double *h) {
    const int d = S->q[kc], tf = S->ne_fixpos[kc]; const double *w = S->wbar + S->qs[kc]; const double ie2 = 1.0 / (S->eta[kc] * S->eta[kc]);
    const double ut = tf == 0 ? w[0] : -w[tf];
    for (int i = 0; i < d; i++) { const double ui = i == 0 ? w[0] : -w[i]; h[i] = ie2 * (2.0 * ui * ut - (i == tf ? (i == 0 ? 1.0 : -1.0) : 0.0)); }
}
/* v = (dx; dy) for r = (rx; ry) */
static void ne_prec(Socp *S, const double *r, double *v) {
    const int n = S->n, m = S->m;
    double *hx = S->w4, *t = S->rk;
    Hinv_apply(S, r, hx);
    if (S->ne_nfix > 0 || S->ne_nkd > 0) {
        /* (23 October 2026) the fixed rows (ne_find_fixed) folded into the two products: the row a x_t = b has
         * t_i = ry_i + a (H^-1 rx)_t, and  t_R - M_RS M_SS^-1 t_S = ry_R + A_R (H^-1 rx - h t_i / (a h_t)) ; after the solve
         * A' dy_R gives M_SR dy_R = a h' (A' dy_R)_k, and A' dy gains a dy_i at the fixed entry. The kept dense columns:
         * [S A_d; A_d' -H_d] (dy; dx_d) = (ry + A_s H_s^-1 rx_s; rx_d), then dx_s = H_s^-1 (A_s' dy - rx_s), dx_d from the
         * solve (the fixed entries are in cones, the kept columns linear: the two do not meet) */
        double h[NE_MAXCONE];
        for (int kc = 0; S->ne_nfix > 0 && kc < S->nq; kc++) {
            const int i = S->ne_fixrow[kc]; if (i < 0) continue;
            const int d = S->q[kc], s_ = S->qs[kc], tf = S->ne_fixpos[kc]; const double a = S->ne_fixa[kc];
            ne_fix_h(S, kc, h);
            const double ti = r[n + i] + a * hx[s_ + tf], al = ti / (a * h[tf]);
            S->ne_fixt[kc] = ti;
            for (int tt = 0; tt < d; tt++) hx[s_ + tt] -= al * h[tt];
        }
        for (int a = 0; a < S->ne_nkd; a++) hx[S->ne_kdcol[a]] = 0.0;
        for (int i = 0; i < m; i++) t[i] = r[n + i];
        A_mul(S, 1.0, hx, t);
        for (int kc = 0; S->ne_nfix > 0 && kc < S->nq; kc++) if (S->ne_fixrow[kc] >= 0) t[S->ne_fixrow[kc]] = 0.0;
        for (int a = 0; a < S->ne_nkd; a++) t[m + a] = r[S->ne_kdcol[a]];
        { const double ts = sc_time(); schol_set_solve_seq(1); schol_solve(S->ne, t, S->swork2); schol_set_solve_seq(0); S->t_solve += sc_time() - ts; S->nback++; }
        if (S->ne_nu > 0) ne_wood_apply(S, t);
        for (int kc = 0; S->ne_nfix > 0 && kc < S->nq; kc++) if (S->ne_fixrow[kc] >= 0) t[S->ne_fixrow[kc]] = 0.0;
        for (int j = 0; j < n; j++) hx[j] = -r[j];
        At_mul(S, 1.0, t, hx);
        for (int kc = 0; S->ne_nfix > 0 && kc < S->nq; kc++) {
            const int i = S->ne_fixrow[kc]; if (i < 0) continue;
            const int d = S->q[kc], s_ = S->qs[kc], tf = S->ne_fixpos[kc]; const double a = S->ne_fixa[kc];
            ne_fix_h(S, kc, h);
            double g = 0; for (int tt = 0; tt < d; tt++) g += h[tt] * (hx[s_ + tt] + r[s_ + tt]);
            const double dyi = (S->ne_fixt[kc] - a * g) / (a * a * h[tf]);
            t[i] = dyi; hx[s_ + tf] += a * dyi;
        }
        Hinv_apply(S, hx, v);
        for (int a = 0; a < S->ne_nkd; a++) v[S->ne_kdcol[a]] = t[m + a];
        for (int i = 0; i < m; i++) v[n + i] = t[i];
        return;
    }
    for (int i = 0; i < m; i++) t[i] = r[n + i];
    A_mul(S, 1.0, hx, t);
    { const double ts = sc_time(); schol_set_solve_seq(1); schol_solve(S->ne, t, S->swork2); schol_set_solve_seq(0); S->t_solve += sc_time() - ts; S->nback++; }
    if (S->ne_nu > 0) ne_wood_apply(S, t);
    double xfs[NE_MAXFREE]; double *xf = S->nf > NE_MAXFREE ? S->ne_xf : xfs;
    if (S->nf > 0) {
        const int nf = S->nf; const double *T = S->ne_T;
        for (int f = 0; f < nf; f++) { double g = -r[f]; for (int p = S->Ap[f]; p < S->Ap[f + 1]; p++) g += S->Ax[p] * t[S->Ai[p]]; xf[f] = g; }
        for (int f = 0; f < nf; f++) { double v = xf[f]; for (int k = 0; k < f; k++) v -= T[f * nf + k] * xf[k]; xf[f] = v / T[f * nf + f]; }
        for (int f = nf - 1; f >= 0; f--) { double v = xf[f]; for (int k = f + 1; k < nf; k++) v -= T[k * nf + f] * xf[k]; xf[f] = v / T[f * nf + f]; }
        for (int f = 0; f < nf; f++) { const double *w = S->ne_W + (size_t)f * m; const double a_ = xf[f]; for (int i = 0; i < m; i++) t[i] -= a_ * w[i]; }
    }
    for (int j = 0; j < n; j++) hx[j] = -r[j];
    At_mul(S, 1.0, t, hx);
    Hinv_apply(S, hx, v);
    for (int f = 0; f < S->nf; f++) v[f] = xf[f];
    for (int i = 0; i < m; i++) v[n + i] = t[i];
}

void schol_set_dynpiv(SChol *S, double p);
int schol_ndyn(const SChol *S);
static void kkt_fill(Socp *S);
static int kkt_factor(Socp *S) {
    Ldl *F = &S->F;
    F->dd = 0;
    if (S->use_dd) { S->sc_ok = 0; return ldl_numeric_dd(F); }
    if (!S->sc) {
        int f = ldl_numeric(F);
        return f;
    }
    schol_zero(S->sc);                  /* (the factorization is in place: the fill positions hold the last factor) */
    double *pm = schol_values(S->sc);
    #pragma omp parallel for schedule(static) num_threads(g_cone_nt) if (S->knz >= 1000000 && g_cone_nt > 1 && !omp_in_parallel())
    for (long q = 0; q < S->knz; q++) pm[S->soff[q]] = F->Ax[q];       /* (distinct positions: the same at any thread count) */
    const int dyn = getenv("BRISK_CONENODYN") == NULL;
    if (dyn) schol_set_dynpiv(S->sc, 1e-7);            /* (a pivot of the wrong sign is replaced, as in the column code) */
    S->sc_ok = schol_factor(S->sc, 0.0) == 0;
    F->ndyn = dyn ? schol_ndyn(S->sc) : 0;
    S->ndyn_total += F->ndyn;
    if (S->sc_ok) return 0;
    /* a pivot of the wrong sign. First a larger regularization of the factored matrix (GMRES removes it:
     * the system it solves keeps the base values); the level is kept for the later iterations */
    if (S->bump_on) {
        const double r0 = S->sreg, x0 = S->sregx, f0 = S->sregf;
        for (int t = 0; t < 2 && !S->sc_ok && S->bump < 1e4; t++) {
            S->bump *= 100.0; S->nbump++;
            S->sreg = S->sreg0 * S->bump; S->sregx = S->sreg0 * S->bump; S->sregf = fmax(1e-6, 1e-8 * S->bump);
            kkt_fill(S);
            schol_zero(S->sc); pm = schol_values(S->sc);
            for (long q = 0; q < S->knz; q++) pm[S->soff[q]] = F->Ax[q];
            S->sc_ok = schol_factor(S->sc, 0.0) == 0;
        }
        if (S->sc_ok) return 0;
        S->sreg = r0; S->sregx = x0; S->sregf = f0; S->bump = 1.0; S->bump_on = 0;
        kkt_fill(S);
    }
    S->nfallback++;                 /* this matrix by the column factorization, which replaces such pivots */
    return ldl_numeric(F);
}
/* t: the right-hand side in the order of the column factorization; solved in place */
/* (27 October 2026, the SOCP developer, a test: BRISK_CONEDDSOLVE1=1) the double-double factor applied in double precision
 * (its leading parts L and D^-1): the factorization in double-double gives an accurate factor of the matrix with the small
 * regularization, which the double factorization cannot; its rounding to double perturbs the preconditioner by about the
 * double precision times the factor's growth, which GMRES removes, while a solve costs a tenth of the double-double one
 * (joint_FC_5, SOCP: 19 iterations in double-double took 92 of 144 s, 52 s of them in the solves) */
static int dd_solve_double(void) { static int on = -1; if (on < 0) { const char *e_ = getenv("BRISK_CONEDDSOLVE1"); on = e_ ? atoi(e_) : 0; } return on; }
static void kkt_back(Socp *S, double *t) {
    if (S->sc && S->sc_ok) {
        const int nk = S->nk; const Ldl *F = &S->F;
        double *u = S->swork + nk;
        for (int i = 0; i < nk; i++) u[i] = t[F->iperm[i]];
        /* (20 October 2026, LP/SOCP, a test: BRISK_CONESOLVEPAR=k, the tree-parallel solve for a factor of at least k million
         * entries, where the sequential one is bound by one core's memory bandwidth; the same bits at any thread count) */
        static double spar = -1; if (spar < 0) { const char *e_ = getenv("BRISK_CONESOLVEPAR"); spar = e_ ? atof(e_) * 1e6 : 0; }
        const int seq_ = !(spar > 0 && (double)S->slnz >= spar);
        if (!(cone_part() && schol_solve_part(S->sc, u, S->swork, cone_part_nt()) == 0)) { schol_set_solve_seq(seq_); schol_solve(S->sc, u, S->swork); schol_set_solve_seq(0); }     /* (the sequential solve: the tree-parallel one is slower here;
                                     * 18 October 2026 at four threads: socp_svm's 432 solves 8.7 -> 18.0 s, socp_fir's 2.2 -> 3.5 s) */
        for (int i = 0; i < nk; i++) t[F->iperm[i]] = u[i];
    } else if (S->F.dd && !dd_solve_double()) {
        /* (kept off: on joint_FC_7 the solves 59.5 -> 55.7 s, db-shear-wall 1.57 -> 1.46, db-plane-strain-prism 10.6 -> 11.2 s: the
         * elimination trees are tall and the levels thin, a barrier a level; TRICKS 8) */
        static int ddpar = -1; if (ddpar < 0) { const char *e_ = getenv("BRISK_CONEDDPAR"); ddpar = e_ ? atoi(e_) : 0; }   /* (1: the level-set solve on the threads) */
        if (ddpar && g_cone_nt > 1 && !omp_in_parallel() && S->F.lnz >= 100000) {
            if (!S->F.lvp || !S->F.tr_ok) { ldl_levels(&S->F); ldl_transpose_dd(&S->F); S->F.tr_ok = 1; }
            ldl_solve_perm_dd_par(&S->F, t, g_cone_nt);
        } else ldl_solve_perm_dd(&S->F, t);
    } else ldl_solve_perm(&S->F, t);
}

/* fill and factor; a supernodal factorization that fails is repeated with more regularization */
static void kkt_fill(Socp *S);
static int kkt_factor_reg(Socp *S) {
    const double ts = sc_time();
    if (S->ne) {
        S->ne_ok = !ne_factor(S);
        if (!S->ne_ok && dbg_on()) socp_printf("      [the normal equations' factorization failed (tiny pivots so far %d): the full system from here]\n", S->ne_tiny);
        if (S->ne_ok) { S->F.ndyn = 0; S->F.dd = 0; S->t_factor += sc_time() - ts; S->nfact++; return 0; }
        ne_free(S); kkt_fill(S);                                   /* failed: the full system from here on */
    }
    if (dbg_on() && !S->sc) { S->F.pivdbg = 1; S->F.pivnf = S->nf; S->F.pivn = S->n; S->F.pivnaux = S->naux; for (int c = 0; c < 4; c++) { S->F.pivmin[c] = 1e300; S->F.pivcnt[c] = 0; } }
    /* (27 October 2026, the SOCP developer, a test: BRISK_CONEBUMPDECAY=1) a raised regularization of the factored matrix tried
     * one level (100 times) lower at the next factorization: the raise is kept for the later iterations, but never lowered
     * again, so one stretch of wrong-sign pivots left the rest of a solve with a poor preconditioner (db-plate-yield-line-fox,
     * SOCP: raised to 1e-4, the solves to 1e-4 .. 1e-5, the dual residual stalled at 2e-6; raised to 1e-6 only, OPTIMAL). A
     * level that fails again costs one factorization */
    static int bdecay = -1; if (bdecay < 0) { const char *e_ = getenv("BRISK_CONEBUMPDECAY"); bdecay = e_ ? atoi(e_) : 1; }   /* (the default since 27 October: the mezo set unchanged, none of its files raises the level) */
    /* (a lower level that fails again holds the decay off for the next 4 factorizations: at most a fifth more factorizations
     * on a problem that needs the raised level throughout) */
    double b_try = 0.0;
    if (bdecay && S->bump_on && S->bump > 1.0 && !S->use_dd) {
        if (S->bump_hold > 0) S->bump_hold--;
        else {
            S->bump /= 100.0; if (S->bump < 1.5) S->bump = 1.0;
            S->sreg = S->sreg0 * S->bump; S->sregx = S->sreg0 * S->bump; S->sregf = fmax(1e-6, 1e-8 * S->bump);
            kkt_fill(S); b_try = S->bump;
            if (dbg_on()) socp_printf("      [the factored matrix's regularization tried 100 times lower: %.1e]\n", S->sregx);
        }
    }
    int f = kkt_factor(S);
    if (b_try > 0.0 && S->bump > b_try) S->bump_hold = 4;
    if (S->F.pivdbg) socp_printf("      [pivots: min |d| free %.1e cone %.1e aux %.1e rows %.1e; below 1e-6: %ld %ld %ld %ld]\n", S->F.pivmin[0], S->F.pivmin[1], S->F.pivmin[2], S->F.pivmin[3], S->F.pivcnt[0], S->F.pivcnt[1], S->F.pivcnt[2], S->F.pivcnt[3]);
    /* pivots of the wrong sign: from here on the factorization is in double-double, with a
     * regularization at its rounding level */
    /* (22 October 2026, LP/SOCP) up to 2e9 flops (about 7 s at the column code's 3e8 flops a second) after a direction that
     * was not finite (dd_force, below): at 5e8 socp_portimpact_20000x100 (SOCP, 6.6e8 flops) was left to the raised
     * regularization, its factor overflowed after replaced pivots, the direction was NaN and the attempt ended REDUCED (68
     * s, 47 iterations; with double-double OPTIMAL in 21 iterations, 42 s). Up to 2e9 always, socp_beamform_200x3000 (SOCP)
     * took double-double it did not need (3.9 -> 11.7 s); relative to the solve's work (16 supernodal factorizations) it cost
     * socp_huber_200kx5000 (SOCP, 4.1e10 flops at 24 GF/s supernodal) 74 -> 258 s. BRISK_CONEDDMAX=f: the base cap f */
    static double ddmax = -1; if (ddmax < 0) { const char *e_ = getenv("BRISK_CONEDDMAX"); ddmax = e_ ? atof(e_) : 5e8; }
    const int dd_cheap = S->cflops <= ddmax || (S->dd_force && S->cflops <= 4.0 * ddmax);
    if (S->dd_on && !S->use_dd && S->want_dd && S->sc && !dd_cheap && !S->dd_polish) {
        /* too large for double-double: a larger regularization of the factored matrix instead
         * (GMRES removes it: the system it solves keeps the base values) */
        S->want_dd = 0;
        if (S->bump < 1e4) {
            S->bump *= 100.0; S->nbump++;
            S->sreg = S->sreg0 * S->bump; S->sregx = S->sreg0 * S->bump; S->sregf = fmax(1e-6, 1e-8 * S->bump);
            kkt_fill(S);
            f = kkt_factor(S);
        }
    }
    if (S->dd_on && !S->use_dd && S->want_dd) {
        S->use_dd = 1; S->sreg = 1e-12; S->sregx = 1e-14;
        { const char *e_ = getenv("BRISK_CONEDDDYN"); if (e_) S->F.dyn_delta = atof(e_); }   /* (test: the replaced pivot's value in double-double) */
        { const char *e_ = getenv("BRISK_CONEDDREG"); if (e_) sscanf(e_, "%lf,%lf", &S->sreg, &S->sregx); }   /* (test: the regularization in double-double: rows, cones) */
        if (S->ne) { ne_free(S); S->ne_ok = 0; }               /* (the rest of the solve on the full system) */
        kkt_fill(S);
        f = kkt_factor(S);
    }
    if (S->use_dd) S->ndd++;
    S->t_factor += sc_time() - ts; S->nfact++;
    return f;
}

/* solves [ -H A' ; A 0 ] [sx; sy] = [rx; ry] with refinement against the unregularized system */
/* v = P r: the factored (regularized) matrix applied to r = (rx; ry) of the unknowns (x; y) */
static void kkt_prec_(Socp *S, const double *r, double *v);
static void kkt_prec(Socp *S, const double *r, double *v) { const double t0 = sc_time(); kkt_prec_(S, r, v); S->t_prec += sc_time() - t0; }   /* (t_prec: the whole application, the solve's permutations and rotations included) */
static void kkt_prec_(Socp *S, const double *r, double *v) {
    const int n = S->n, m = S->m, nk = S->nk;
    if (S->ne && S->ne_ok) { ne_prec(S, r, v); return; }
    const Ldl *F = &S->F;
    double *t = S->rk;
    if (S->sc && S->sc_ok) {
        /* (18 October 2026, LP/SOCP: the supernodal factor takes the unknowns in their own order (it applies its
         * permutation itself): the round trip through the column factorization's order, four scattered passes over the
         * nk unknowns, is left out; the same arithmetic. TV denoising 300: 10 of 70 samples were those passes.
         * 23 October 2026, the SOCP developer: also with the small cones in the eigenbasis of their scaling, whose
         * unknown is Q' x: Q' on the way in, Q on the way out, as below) */
        const int naux = S->naux; double *u = S->swork + nk;
        memcpy(u, r, sizeof(double) * (size_t)n);
        for (int i = 0; i < naux; i++) u[n + i] = 0.0;
        memcpy(u + n + naux, r + n, sizeof(double) * (size_t)m);
        if (S->anyrot) for (int k = 0; k < S->nq; k++) if (S->rot[k]) {
            const int d = S->q[k], s = S->qs[k]; const double *Q = S->Qr + 16 * (size_t)k;
            for (int i = 0; i < d; i++) { double a = 0; for (int j = 0; j < d; j++) a += Q[j + i * d] * r[s + j]; u[s + i] = a; }
        }
        { const double ts = sc_time();
          if (!(cone_part() && schol_solve_part(S->sc, u, S->swork, cone_part_nt()) == 0)) { schol_set_solve_seq(1); schol_solve(S->sc, u, S->swork); schol_set_solve_seq(0); }
          S->t_solve += sc_time() - ts; S->nback++; }
        memcpy(v, u, sizeof(double) * (size_t)n);
        memcpy(v + n, u + n + naux, sizeof(double) * (size_t)m);
        if (S->anyrot) for (int k = 0; k < S->nq; k++) if (S->rot[k]) {
            const int d = S->q[k], s = S->qs[k]; const double *Q = S->Qr + 16 * (size_t)k;
            for (int j = 0; j < d; j++) { double a = 0; for (int i = 0; i < d; i++) a += Q[j + i * d] * u[s + i]; v[s + j] = a; }
        }
        return;
    }
    /* (23 October 2026, the SOCP developer) the right-hand side into the factor's order in one sequential pass, read
     * through the permutation (the auxiliary unknowns' entries 0), in place of a pass of zeros and two scattered passes:
     * the same values; socp_snl_20000 (SOCP, 945,812 unknowns): 3.4 s of its 21 s were these passes */
    /* (27 October 2026) the passes through the permutation on the threads for a large system (copies: the same bits):
     * joint_FC_5 (SOCP, 932,953 unknowns): 6.6 s of its 53.6 s of solves */
    const int ptn = nk >= 200000 && g_cone_nt > 1 && !omp_in_parallel() ? g_cone_nt : 1;
    { const int *pm = F->perm, npa = n + S->naux, na = S->naux;
      #pragma omp parallel for schedule(static) num_threads(ptn) if (ptn > 1)
      for (int p = 0; p < nk; p++) { const int o = pm[p]; t[p] = o < n ? r[o] : o < npa ? 0.0 : r[o - na]; } }
    /* rotated cones: the unknown is Q' x */
    if (S->anyrot) for (int k = 0; k < S->nq; k++) if (S->rot[k]) {
        const int d = S->q[k], s = S->qs[k]; const double *Q = S->Qr + 16 * (size_t)k;
        double rr[4];
        for (int i = 0; i < d; i++) { double a = 0; for (int j = 0; j < d; j++) a += Q[j + i * d] * r[s + j]; rr[i] = a; }
        for (int i = 0; i < d; i++) t[F->iperm[s + i]] = rr[i];
    }
    { const double ts = sc_time(); kkt_back(S, t); S->t_solve += sc_time() - ts; S->nback++; }
    #pragma omp parallel for schedule(static) num_threads(ptn) if (ptn > 1)
    for (int i = 0; i < n; i++) v[i] = t[F->iperm[i]];
    #pragma omp parallel for schedule(static) num_threads(ptn) if (ptn > 1)
    for (int i = 0; i < m; i++) v[n + i] = t[F->iperm[n + S->naux + i]];
    if (S->anyrot) for (int k = 0; k < S->nq; k++) if (S->rot[k]) {
        const int d = S->q[k], s = S->qs[k]; const double *Q = S->Qr + 16 * (size_t)k;
        double xt[4];
        for (int i = 0; i < d; i++) xt[i] = t[F->iperm[s + i]];
        for (int j = 0; j < d; j++) { double a = 0; for (int i = 0; i < d; i++) a += Q[j + i * d] * xt[i]; v[s + j] = a; }
    }
}
/* w = K v for the matrix without regularization, K = [ -H A' ; A 0 ] */
static void kkt_mul(Socp *S, const double *v, double *w) {
    const int n = S->n, m = S->m;
    const double tm0 = sc_time();
    H_apply(S, v, w);
    for (int i = 0; i < n; i++) w[i] = -w[i];
    /* (27 October 2026, the SOCP developer, a test: BRISK_CONEFREEPROX=1) the free variables' regularization as a proximal
     * term of the step (Altman and Gondzio 1999; Friedlander and Orban 2012): delta |x_f - x_f^k|^2 centred at the iterate,
     * so the regularized system is the one solved and the refinement removes only the factor's rounding (the solution is
     * unchanged: the term vanishes at convergence). Without it every solve is refined against the unregularized system,
     * 26-32 applications of the factor an iteration on the structural-design SOCPs, whose factor carries 1e-6 there */
    { static double fprox = -1; if (fprox < 0) { const char *e_ = getenv("BRISK_CONEFREEPROX"); fprox = e_ ? atof(e_) : 0.0; }   /* (rho: the share of the factor's free regularization kept in the solved system; 1 the whole) */
      if (fprox > 0 && !S->use_dd) { const double dp = fprox * S->sregf; for (int i = 0; i < S->nf; i++) w[i] -= dp * v[i]; } }
    if (S->fpr && !S->use_dd) for (int i = 0; i < S->nf; i++) w[i] -= S->fpr[i] * v[i];     /* (BRISK_CONEFREEPROXREL: kkt_fill) */
    if (S->nd > 0) Bt_mul(S, 1.0, v + n, w); else At_mul(S, 1.0, v + n, w);
    /* the (base) dual regularization is part of the system that is solved. 20 October 2026 (LP/SOCP): on the normal
     * equations' route below the target accuracy (min(1e-9, tol / 100)): that route has no double-double end phase,
     * which on the full system lowers it to 1e-12, and the base 1e-9 left a floor of about 1e-9 |y| on the primal
     * residual (MOI's SecondOrderCone_Nonnegatives, two rows, at tol 1e-10: 8.6e-10 for 20 iterations while tau grew
     * to 4e30; ISSUES 123). The factorization keeps its 1e-9 (the preconditioner); GMRES removes the difference. */
    const double dl = S->use_dd ? S->sreg : (S->ne && S->ne_ok && S->ne_dl > 0 ? fmin(S->sreg0, S->ne_dl) : S->sreg0);
    for (int i = 0; i < m; i++) w[n + i] = dl * v[n + i];
    if (S->nd > 0) { B_mul(S, 1.0, v, w + n); dn_mul(S, 1.0, v, w + n, v + n, w); }     /* (the dense block read once for both) */
    else A_mul(S, 1.0, v, w + n);
    S->t_mul += sc_time() - tm0;
}
/* solves [ -H A' ; A 0 ] [sx; sy] = [rx; ry]: the factorization of the regularized matrix is the
 * preconditioner of GMRES on the matrix itself (at most maxref steps; 0: the preconditioner alone).
 * One step is a step of iterative refinement with the best step length; a few steps remove the
 * directions in which the regularization is not small against the matrix. */
static void kkt_solve(Socp *S, const double *rx, const double *ry, double *sx, double *sy, int maxref) {
    const int n = S->n, m = S->m, N = n + m;
    double *e = S->sk, *s = S->gs;
    S->solves++;
    memcpy(e, rx, sizeof(double) * (size_t)n); memcpy(e + n, ry, sizeof(double) * (size_t)m);
    const double bn = nrm2(N, e);
    kkt_prec(S, e, s);
    S->last_res = 0.0;
    static int gmdbg = -1; if (gmdbg < 0) { const char *e = getenv("BRISK_CONEDBG"); gmdbg = e && atoi(e) >= 2; }
    static int gmcap = -1; if (gmcap < 0) { const char *e_ = getenv("BRISK_CONEGMMAX"); gmcap = e_ ? atoi(e_) : 0; }   /* (test: at most k GMRES steps) */
    if (gmcap > 0 && maxref > gmcap) maxref = gmcap;
    if (maxref > 0 && bn > 0) {
        const int kmax = maxref < S->gm_max ? maxref : S->gm_max;
        double *V = S->gV, *Z = S->gZ, *Hm = S->gH, *cs = S->gcs, *sn = S->gcs + kmax + 1, *g = S->gcs + 2 * (kmax + 1);
        static double gcap = -1; if (gcap < 0) { const char *e_ = getenv("BRISK_CONEREFMUCAP"); gcap = e_ ? atof(e_) : 1.0; }   /* (test: the absolute target at most this fraction of the solve's own right-hand side) */
        const double tol = fmax(S->gm_tol * bn, fmin(S->gm_abs, gcap * bn));
        double *w = S->lastx;                 /* N entries of work (lastx and lasty are contiguous: see the allocation) */
        kkt_mul(S, s, w);
        for (int i = 0; i < n; i++) w[i] = rx[i] - w[i];
        for (int i = 0; i < m; i++) w[n + i] = ry[i] - w[n + i];
        double beta = nrm2(N, w);
        S->last_res = beta / bn;
        /* (26 October 2026, a test: BRISK_CONEREFMU=4) the preconditioned solve kept without refinement when it already meets
         * the step's needs, as the QP method's targets: its residual's dual block below a tenth of the dual residual, its
         * primal block below a tenth of the primal one, and its effect on the gap sum |x_j r_j| + sum |y_i r_i| below a
         * tenth of the complementarity (largest entries; at least 1e-2 of the tolerance) */
        if (S->tg_d > 0) {
            double ed = 0, ep = 0, eg = 0;
            if (S->tg_w) {       /* (5: in the measure of the problem as given, the 2-norms of the residuals unscaled) */
                for (int i = 0; i < n; i++) { const double a = fabs(w[i]), u = a / S->Ec[i]; ed += u * u; eg += fabs(S->x[i]) * a; }
                for (int i = 0; i < m; i++) { const double a = fabs(w[n + i]), u = a / S->Dr[i]; ep += u * u; eg += fabs(S->y[i]) * a; }
                ed = sqrt(ed); ep = sqrt(ep);
            } else {
            for (int i = 0; i < n; i++) { const double a = fabs(w[i]); if (a > ed) ed = a; eg += fabs(S->x[i]) * a; }
            for (int i = 0; i < m; i++) { const double a = fabs(w[n + i]); if (a > ep) ep = a; eg += fabs(S->y[i]) * a; }
            }
            static double scap = -1; if (scap < 0) { const char *e_ = getenv("BRISK_CONEREFMUSELF"); scap = e_ ? atof(e_) : 1.0; }   /* (test: also the solve's own residual at most this fraction of its right-hand side) */
            const double f_ = S->tg_f, fl_ = S->tg_fl;      /* (6, 7: the residual targets times the fraction the step aims at, f = sigma; at least the floors) */
            static int rpiv = -1; if (rpiv < 0) rpiv = getenv("BRISK_CONEREFMUPIV") != NULL;   /* (test: no skip with replaced pivots or in double-double) */
            if (f_ > 0 && !(rpiv && (S->F.ndyn > 0 || S->use_dd)) && ed <= fmax(f_ * S->tg_d, fmax(fl_, S->tg_fd)) && ep <= fmax(f_ * S->tg_p, fmax(fl_, S->tg_fp)) && eg <= fmax(f_ * S->tg_g, fl_) && beta <= scap * bn) { S->tg_skip++; beta = 0.0; S->last_res = 0.0; }   /* (a solve that meets the step's needs has succeeded: last_res is read as the solve's success by the switches to double-double, which took nql180 (SOCP) to 14 of 20 factorizations in double-double, 27 -> 67 s) */
        }
        if (S->rot_mode == 2 && S->nsmallc > 0) {      /* (the residual of the preconditioned solve in the small cones: the switch below) */
            double rs = 0; for (int k = 0; k < S->nq; k++) if (S->q[k] <= SOCP_DENSE_MAX) for (int i = 0; i < S->q[k]; i++) rs += w[S->qs[k] + i] * w[S->qs[k] + i];
            S->it_rs2 += rs; S->it_rt2 += beta * beta;
        }
        /* (20 October 2026, LP/SOCP, a test: BRISK_CONERECYCLE=1) the right-hand sides of one factorization (p = K^{-1}[c; b], the
         * predictor, the corrector, the centrality correctors, the dual refinement) share the preconditioner's weak directions:
         * the previous solve's Krylov space (V, Z = M^{-1} V, A Z = V H) is used first: the least-squares combination of its
         * vectors for this residual (a small triangular solve on the rotated H), the residual deflated by it, then GMRES on the
         * rest. k + 1 dot products and 2 k + 1 axpys, no product with the matrix. */
        static int rcy = -1; if (rcy < 0) { const char *e_ = getenv("BRISK_CONERECYCLE"); rcy = e_ ? atoi(e_) : 0; }
        if (rcy && S->rc_k > 0 && S->rc_k < 64 && S->rc_fact == S->nfact && S->rc_N == N && beta > tol) {
            const int k = S->rc_k, kk = S->rc_kmax;
            const double *cs0 = S->gcs, *sn0 = S->gcs + kk + 1;
            double c[65], t[65], yv[64]; int okk = 1;
            for (int j = 0; j <= k; j++) c[j] = dotv(N, V + (size_t)j * N, w);
            for (int j = 0; j < k; j++) { const double a = cs0[j] * c[j] + sn0[j] * c[j + 1]; c[j + 1] = -sn0[j] * c[j] + cs0[j] * c[j + 1]; c[j] = a; }
            for (int j = k - 1; j >= 0; j--) {
                double a = c[j];
                for (int l = j + 1; l < k; l++) a -= Hm[(size_t)l * (kk + 1) + j] * yv[l];
                yv[j] = a / Hm[(size_t)j * (kk + 1) + j];
                if (!isfinite(yv[j])) okk = 0;
            }
            if (okk) {
                for (int j = 0; j < k; j++) t[j] = c[j];
                t[k] = 0.0;
                for (int j = k - 1; j >= 0; j--) { const double a = cs0[j] * t[j] - sn0[j] * t[j + 1]; t[j + 1] = sn0[j] * t[j] + cs0[j] * t[j + 1]; t[j] = a; }
                for (int j = 0; j < k; j++) { const double a = yv[j]; const double *zj = Z + (size_t)j * N; for (int i = 0; i < N; i++) s[i] += a * zj[i]; }
                for (int j = 0; j <= k; j++) { const double a = t[j]; const double *vj = V + (size_t)j * N; for (int i = 0; i < N; i++) w[i] -= a * vj[i]; }
                const double b2 = nrm2(N, w);
                if (b2 < 0.9 * beta) {       /* (kept only when it gains: a least-squares combination from a stalled space is noise) */
                    S->rc_used++; S->rc_gain += log(fmax(beta / fmax(b2, 1e-300), 1e-300));
                    if (gmdbg) socp_printf("        [recycled %d vectors: residual %.2e -> %.2e]\n", k, beta / bn, b2 / bn);
                    beta = b2; S->last_res = beta / bn;
                } else {
                    for (int j = 0; j < k; j++) { const double a = yv[j]; const double *zj = Z + (size_t)j * N; for (int i = 0; i < N; i++) s[i] -= a * zj[i]; }
                    for (int j = 0; j <= k; j++) { const double a = t[j]; const double *vj = V + (size_t)j * N; for (int i = 0; i < N; i++) w[i] += a * vj[i]; }
                    if (gmdbg) socp_printf("        [recycled %d vectors: no gain (%.2e -> %.2e), undone]\n", k, beta / bn, b2 / bn);
                }
            }
        }
        S->rc_k = 0;
        if (gmdbg) {          /* (the residual of the preconditioned solve by block: free, linear, cone columns, rows) */
            double rf = 0, rl = 0, rq = 0, ry_ = 0;
            for (int i = 0; i < S->nf; i++) rf += w[i] * w[i];
            for (int i = S->nf; i < S->nf + S->nl; i++) rl += w[i] * w[i];
            for (int i = S->nf + S->nl; i < n; i++) rq += w[i] * w[i];
            if (gmdbg) { double rs = 0, rb = 0, rmax = 0; int kmax_ = -1;
                for (int k = 0; k < S->nq; k++) { double t_ = 0; for (int i = 0; i < S->q[k]; i++) t_ += w[S->qs[k] + i] * w[S->qs[k] + i];
                    if (S->q[k] <= SOCP_DENSE_MAX) rs += t_; else rb += t_; if (t_ > rmax) { rmax = t_; kmax_ = k; } }
                socp_printf("        [cones: small %.1e large %.1e; the largest in cone %d (dimension %d, rotated %d) %.1e]\n", sqrt(rs) / bn, sqrt(rb) / bn, kmax_, kmax_ >= 0 ? S->q[kmax_] : 0, kmax_ >= 0 ? (int)(kmax_ >= S->nq - S->nr_rot) : 0, sqrt(rmax) / bn); }
            for (int i = 0; i < m; i++) ry_ += w[n + i] * w[n + i];
            socp_printf("        [solve residual / |rhs|: free %.1e linear %.1e cones %.1e rows %.1e]\n", sqrt(rf) / bn, sqrt(rl) / bn, sqrt(rq) / bn, sqrt(ry_) / bn);
        }
        if (beta > tol) {
            for (int i = 0; i < N; i++) V[i] = w[i] / beta;
            g[0] = beta;
            double rhist[64]; rhist[0] = beta;      /* (the residual norm after each step, for the stagnation test) */
            int k = 0;
            for (; k < kmax; k++) {
                double *zk = Z + (size_t)k * N, *vk1 = V + (size_t)(k + 1) * N;
                kkt_prec(S, V + (size_t)k * N, zk);
                kkt_mul(S, zk, vk1);
                S->refine_total++;
                double *hk = Hm + (size_t)k * (kmax + 1);
                for (int j = 0; j <= k; j++) {
                    const double *vj = V + (size_t)j * N;
                    const double h = dotv(N, vj, vk1);
                    hk[j] = h;
                    for (int i = 0; i < N; i++) vk1[i] -= h * vj[i];
                }
                const double hn = nrm2(N, vk1);
                hk[k + 1] = hn;
                if (hn > 0) for (int i = 0; i < N; i++) vk1[i] /= hn;
                for (int j = 0; j < k; j++) { const double a = cs[j] * hk[j] + sn[j] * hk[j + 1]; hk[j + 1] = -sn[j] * hk[j] + cs[j] * hk[j + 1]; hk[j] = a; }
                const double rr = hypot(hk[k], hk[k + 1]);
                if (!(rr > 0) || !isfinite(rr)) break;
                cs[k] = hk[k] / rr; sn[k] = hk[k + 1] / rr;
                hk[k] = rr; hk[k + 1] = 0.0;
                g[k + 1] = -sn[k] * g[k]; g[k] = cs[k] * g[k];
                if (gmdbg) socp_printf("        gmres %d: %.2e\n", k + 1, fabs(g[k + 1]) / bn);
                if (k + 1 < 64) rhist[k + 1] = fabs(g[k + 1]);
                if (fabs(g[k + 1]) <= tol || !(hn > 0)) { k++; break; }
                /* (20 October 2026, LP/SOCP) GMRES stops when its last two steps together gained less than a factor 1/c^2 = 11 on
                 * the residual (BRISK_CONEGMSTAG=c, 0.3; 0 = off): the preconditioner has run out (late in a degenerate solve the
                 * double factorization contracts by 0.5-0.9 a step and the ten steps of the cap give two digits), and a direction
                 * accurate to the residual reached is taken. Only in the refinement regime, after two steps and above a floor
                 * (below). 79 SOCP + 43 CBLIB files: the same status lines (sched_200_100_orig REDUCED -> OPTIMAL), 24,576 -> 23,782
                 * solves, time 0.962 (0.932 over 5 s); nql180 28.6 -> 21.5 s, grouplasso_2k 14.5 -> 10.3 s. A flat cap of two steps
                 * gains more (0.914, 0.748 over 5 s, joint_FC_12 124 -> 70 s) but breaks the truss and sched_100_50_orig: open. */
                { static double stag = -1; static int stagmin = 2; if (stag < 0) { const char *e_ = getenv("BRISK_CONEGMSTAG"); stag = e_ ? atof(e_) : 0.3; const char *m_ = getenv("BRISK_CONEGMSTAGMIN"); stagmin = m_ ? atoi(m_) : 2; }
                  /* (the gain over the last two steps, not one: a Krylov history has plateaus before a drop) */
                  /* (and only in the refinement regime: the preconditioned solve's own residual below stagres; a solve that
                   *  starts from a residual of order one, after replaced pivots, is rebuilt by GMRES with an irregular history) */
                  static double stagres = -1; if (stagres < 0) { const char *e_ = getenv("BRISK_CONEGMSTAGRES"); stagres = e_ ? atof(e_) : 1e-6; }   /* (sched_100_50_orig: a solve that starts at 1e4 and gains ten orders in seven steps is still going; stopped it costs 31 -> 59 iterations) */
                  static double stagfl = -1; if (stagfl < 0) { const char *e_ = getenv("BRISK_CONEGMSTAGFLOOR"); stagfl = e_ ? atof(e_) : 1e-10; }   /* (no stop below this residual: a rounding plateau at 3e-12 made sched_100_100_orig miss its last iteration) */
                  /* (and only while GMRES has not shown that it converges: its total gain since the start below 1/stagtot; a solve that
                   *  started from a residual of order 1e4 after replaced pivots and gained ten orders in seven steps is still going) */
                  static double stagtot = -1; if (stagtot < 0) { const char *e_ = getenv("BRISK_CONEGMSTAGTOT"); stagtot = e_ ? atof(e_) : 0; }   /* (a test, off: the gate above does this job) */
                  if (stag > 0 && k + 1 >= stagmin && k >= 1 && k + 1 < 64 && S->last_res <= stagres && rhist[k + 1] > stagfl * bn && rhist[k + 1] > stagtot * rhist[0] && rhist[k + 1] > stag * stag * rhist[k - 1]) { if (gmdbg) socp_printf("        [gmres stopped by stagnation at step %d]\n", k + 1); k++; break; } }
            }
            if (rcy && k > 0) { S->rc_k = k; S->rc_kmax = kmax; S->rc_fact = S->nfact; S->rc_N = N; }
            if (k > 0) {
                double *yv = S->gcs + 3 * (kmax + 1);
                for (int j = k - 1; j >= 0; j--) {
                    double a = g[j];
                    for (int l = j + 1; l < k; l++) a -= Hm[(size_t)l * (kmax + 1) + j] * yv[l];
                    yv[j] = a / Hm[(size_t)j * (kmax + 1) + j];
                }
                int okk = 1; for (int j = 0; j < k; j++) if (!isfinite(yv[j])) okk = 0;
                if (okk) {
                    for (int j = 0; j < k; j++) { const double a = yv[j]; const double *zj = Z + (size_t)j * N; for (int i = 0; i < N; i++) s[i] += a * zj[i]; }
                    S->last_res = fabs(g[k]) / bn;
                }
            }
        }
    }
    if (maxref > 0 && (S->last_res > 1e-10 || !isfinite(S->last_res)) && S->F.ndyn > 0) S->want_dd = 1;      /* replaced pivots and a solve that did not refine: double-double next */
    if (maxref > 0 && (S->last_res > S->it_maxres || !isfinite(S->last_res))) S->it_maxres = isfinite(S->last_res) ? S->last_res : 1e300;
    memcpy(sx, s, sizeof(double) * (size_t)n); memcpy(sy, s + n, sizeof(double) * (size_t)m);
}


/* refinement of a direction (dx, dy, dz, dtau) against the dual equation dz = eta rD + c dtau - A'dy:
 * when the miss is more than the reduction asked of the dual residual, and more than a tenth of its
 * tolerance, one more solve with the miss as right-hand side (at most twice). (With a tenth and a
 * hundredth the refinement ran where it does not pay: 24-40 % more solves on two large problems.)
 * Returns the number of refinements. */
static int dual_refine(Socp *S, double tol, double etaf, const double *rD, double dtau, double byv, double cx, double ncs,
                       double *dx, double *dy, double *dz, double *t2) {
    const int n = S->n, m = S->m;
    int nref = 0;
    for (int rf = 0; rf < 2 && S->dz_refine; rf++) {
        const double flo = 0.1 * tol * fmax(S->tau * (1.0 + ncs), fmax(byv, -cx));
        double *wd = S->wdz;
        for (int j = 0; j < n; j++) wd[j] = etaf * rD[j] + S->c[j] * dtau - dz[j];
        At_mul(S, -1.0, dy, wd);
        for (int j = 0; j < S->nf; j++) wd[j] = 0.0;
        double dm = 0, rm = 0;
        for (int j = S->nf; j < n; j++) { dm = fmax(dm, fabs(wd[j])); rm = fmax(rm, fabs(etaf * rD[j])); }
        if (!(dm > fmax(rm, flo))) break;
        for (int i = 0; i < m; i++) S->wdy[i] = 0.0;
        kkt_solve(S, wd, S->wdy, S->wdx, S->wdy2, S->refine_max);
        for (int j = 0; j < n; j++) dx[j] += S->wdx[j];
        for (int i = 0; i < m; i++) dy[i] += S->wdy2[i];
        H_apply(S, S->wdx, t2);
        for (int j = S->nf; j < n; j++) dz[j] -= t2[j];
        S->dz_count++; nref++;
    }
    return nref;
}

/* largest alpha in (0, amax] with x + alpha d in the cone (second-order cone part of dimension d) */
static double soc_step(int d, const double *x, const double *dx, double amax) {
    double a = dx[0] * dx[0], b = x[0] * dx[0], c = x[0] * x[0];
    for (int i = 1; i < d; i++) { a -= dx[i] * dx[i]; b -= x[i] * dx[i]; c -= x[i] * x[i]; }
    /* a al^2 + 2 b al + c >= 0, c > 0 */
    double al = amax;
    if (c <= 0) return 0.0;
    const double disc = b * b - a * c;
    if (a < 0) { const double r = (-b - sqrt(fmax(disc, 0.0))) / a; if (r > 0 && r < al) al = r; }       /* the positive root */
    else if (a > 0) { if (b < 0 && disc >= 0) { const double r = c / (-b + sqrt(disc)); if (r > 0 && r < al) al = r; } }
    else if (b < 0) { const double r = -c / (2.0 * b); if (r < al) al = r; }
    if (dx[0] < 0) { const double r = -x[0] / dx[0]; if (r < al) al = r; }
    return al;
}
static double cone_step(const Socp *S, const double *x, const double *dx, double amax) {
    double al = amax;
    const int par = cone_vpar(S);
    #pragma omp parallel for schedule(static) num_threads(g_cone_nt) reduction(min:al) if (par)
    for (int i = S->nf; i < S->nf + S->nl; i++) if (dx[i] < 0) { const double r = -x[i] / dx[i]; if (r < al) al = r; }
    if (!par) { for (int k = 0; k < S->nq; k++) al = soc_step(S->q[k], x + S->qs[k], dx + S->qs[k], al); return al; }
    double ac = amax;      /* (each cone's step from amax, the minimum taken after: soc_step(.., a) = min(a, the cone's own step)) */
    #pragma omp parallel for schedule(static) num_threads(g_cone_nt) reduction(min:ac)
    for (int k = 0; k < S->nq; k++) { const double a_ = soc_step(S->q[k], x + S->qs[k], dx + S->qs[k], amax); if (a_ < ac) ac = a_; }
    return al < ac ? al : ac;
}

/* 1 if x and z are in the interior of the cone (the test of nt_update) */
/* (20 October 2026, LP/SOCP, the arc search) the point p + sin(a) d1 + (1 - cos a) d2 strictly inside the cones, tau, kappa > 0 */
static int arc_inside(const Socp *S, double a, const double *x1, const double *z1, double t1_, double k1_, const double *x2, const double *z2, double t2_, double k2_) {
    const double sa = sin(a), ca = 1.0 - cos(a);
    if (!(S->tau + sa * t1_ + ca * t2_ > 0) || !(S->kap + sa * k1_ + ca * k2_ > 0)) return 0;
    for (int i = S->nf; i < S->nf + S->nl; i++) if (!(S->x[i] + sa * x1[i] + ca * x2[i] > 0) || !(S->z[i] + sa * z1[i] + ca * z2[i] > 0)) return 0;
    for (int k = 0; k < S->nq; k++) {
        const int d = S->q[k], s = S->qs[k];
        double xn = 0, zn = 0; const double x0 = S->x[s] + sa * x1[s] + ca * x2[s], z0 = S->z[s] + sa * z1[s] + ca * z2[s];
        for (int i = 1; i < d; i++) { const double xi = S->x[s + i] + sa * x1[s + i] + ca * x2[s + i], zi = S->z[s + i] + sa * z1[s + i] + ca * z2[s + i]; xn += xi * xi; zn += zi * zi; }
        if (!(x0 > 0) || !(z0 > 0) || !(x0 * x0 - xn > 0) || !(z0 * z0 - zn > 0)) return 0;
    }
    return 1;
}
static int cone_interior(const Socp *S) {
    for (int i = 0; i < S->nl; i++) if (!(S->x[S->nf + i] > 0) || !(S->z[S->nf + i] > 0)) return 0;
    for (int k = 0; k < S->nq; k++) {
        const int d = S->q[k], s = S->qs[k];
        const double *x = S->x + s, *z = S->z + s;
        double xn = 0, zn = 0;
        for (int i = 1; i < d; i++) { xn += x[i] * x[i]; zn += z[i] * z[i]; }
        xn = sqrt(xn); zn = sqrt(zn);
        if (!((x[0] - xn) * (x[0] + xn) > 0) || !((z[0] - zn) * (z[0] + zn) > 0) || !(x[0] > 0) || !(z[0] > 0)) return 0;
    }
    return S->tau > 0 && S->kap > 0;
}

/* Nesterov-Todd scaling from x and z; lambda = W x. Returns 0, or 1 if a point is not interior. */
static int nt_update(Socp *S) {
    for (int i = 0; i < S->nl; i++) {
        const double xi = S->x[S->nf + i], zi = S->z[S->nf + i];
        if (!(xi > 0) || !(zi > 0)) return 1;
        S->h[i] = zi / xi; S->lam[S->nf + i] = sqrt(xi * zi);
    }
    for (int k = 0; k < S->nq; k++) {
        const int d = S->q[k], s = S->qs[k];
        const double *x = S->x + s, *z = S->z + s; double *w = S->wbar + s;
        double xn = 0, zn = 0, xz = x[0] * z[0];
        for (int i = 1; i < d; i++) { xn += x[i] * x[i]; zn += z[i] * z[i]; xz += x[i] * z[i]; }
        xn = sqrt(xn); zn = sqrt(zn);
        const double xx = (x[0] - xn) * (x[0] + xn), zz = (z[0] - zn) * (z[0] + zn);
        if (!(xx > 0) || !(zz > 0) || !(x[0] > 0) || !(z[0] > 0)) return 1;
        const double sx = sqrt(xx), sz = sqrt(zz);
        const double gam = sqrt(0.5 * (1.0 + xz / (sx * sz)));
        w[0] = (z[0] / sz + x[0] / sx) / (2.0 * gam);
        for (int i = 1; i < d; i++) w[i] = (z[i] / sz - x[i] / sx) / (2.0 * gam);
        /* a^2 - |q|^2 = 1 to rounding: enforced on a */
        double qq = 0.0; for (int i = 1; i < d; i++) qq += w[i] * w[i];
        w[0] = sqrt(1.0 + qq);
        S->eta[k] = sqrt(sz / sx);
        W_apply(S, k, 0, x, S->lam + s);

    }
    return 0;
}

static void *xcalloc(size_t n, size_t s) { void *p = calloc(n ? n : 1, s); return p; }

/* (23 October 2026, the SOCP developer) A's dense rows kept as one dense block. A row is a candidate when it holds at least
 * max(1000, n / 20) entries; the block (the candidates over the union of their columns) is kept when it is at least half
 * full and holds at least a quarter of A's entries. A product then reads 8 bytes an entry instead of 12 (no index) in
 * contiguous runs, and the product with the full system reads the block once for A x and A'y together (the FIR filters
 * of CBLIB: 1000 rows of about 40,000 entries, 99.9 % of A, read about 37 times an iteration). BRISK_CONEDENSEROWS=0: off */
static void dn_build(Socp *S) {
    const int m = S->m, n = S->n; const long nz = S->Ap[n];
    S->nd = 0;
    { static int on = -1; if (on < 0) { const char *e_ = getenv("BRISK_CONEDENSEROWS"); on = !(e_ && atoi(e_) == 0); } if (!on) return; }
    if (nz < 1000000 || m <= 0) return;
    const double thr = fmax(1000.0, n / 20.0);
    int *cnt = (int *)xcalloc((size_t)m, sizeof(int));
    for (long p = 0; p < nz; p++) cnt[S->Ai[p]]++;
    int nd = 0; long ed = 0;
    for (int i = 0; i < m; i++) if (cnt[i] >= thr) { nd++; ed += cnt[i]; }
    if (nd == 0 || ed < nz / 4) { free(cnt); return; }
    int *dmap = (int *)malloc(sizeof(int) * (size_t)m), *dcm = (int *)malloc(sizeof(int) * (size_t)n);
    int *drow = (int *)malloc(sizeof(int) * (size_t)nd);
    { int t = 0; for (int i = 0; i < m; i++) { dmap[i] = cnt[i] >= thr ? t : -1; if (dmap[i] >= 0) drow[t++] = i; } }
    free(cnt);
    int nc = 0;
    for (int j = 0; j < n; j++) { dcm[j] = -1; for (int p = S->Ap[j]; p < S->Ap[j + 1]; p++) if (dmap[S->Ai[p]] >= 0) { dcm[j] = nc++; break; } }
    if ((double)ed < 0.5 * (double)nd * (double)nc) { free(dmap); free(dcm); free(drow); return; }
    double *Dm = (double *)calloc((size_t)nd * (size_t)nc, sizeof(double));
    int *dcol = (int *)malloc(sizeof(int) * ((size_t)nc + 1));
    int *Bp = (int *)malloc(sizeof(int) * ((size_t)n + 1)), *Bi = (int *)malloc(sizeof(int) * ((size_t)(nz - ed) + 1)), *Bq = (int *)malloc(sizeof(int) * ((size_t)nz + 1));
    double *Bx = (double *)malloc(sizeof(double) * ((size_t)(nz - ed) + 1));
    const int dch = nc / 32 < 256 ? 256 : nc / 32 > 2048 ? 2048 : nc / 32, nch = (nc + dch - 1) / dch;
    double *dxs = (double *)malloc(sizeof(double) * ((size_t)nc + 1)), *dacc = (double *)malloc(sizeof(double) * ((size_t)nc + 1)), *dpart = (double *)malloc(sizeof(double) * ((size_t)nch * nd + 1));
    if (!Dm || !dcol || !Bp || !Bi || !Bq || !Bx || !dxs || !dacc || !dpart) {      /* (out of memory: the products by A's own copies) */
        free(Dm); free(dcol); free(Bp); free(Bi); free(Bq); free(Bx); free(dxs); free(dacc); free(dpart); free(dmap); free(dcm); free(drow); return;
    }
    for (int j = 0; j < n; j++) if (dcm[j] >= 0) dcol[dcm[j]] = j;
    long t = 0;
    for (int j = 0; j < n; j++) {
        Bp[j] = (int)t;
        for (int p = S->Ap[j]; p < S->Ap[j + 1]; p++) {
            const int i = S->Ai[p];
            if (dmap[i] >= 0) { Dm[(size_t)dmap[i] * nc + dcm[j]] = S->Ax[p]; Bq[p] = -1; }
            else { Bi[t] = i; Bx[t] = S->Ax[p]; Bq[p] = (int)t; t++; }
        }
    }
    Bp[n] = (int)t;
    S->nd = nd; S->ncd = nc; S->dch = dch; S->drow = drow; S->dmap = dmap; S->dcol = dcol; S->dcm = dcm; S->Dm = Dm;
    S->Bp = Bp; S->Bi = Bi; S->Bx = Bx; S->Bq = Bq; S->dxs = dxs; S->dacc = dacc; S->dpart = dpart;
    if (dbg_on()) socp_printf("  [A's dense rows: %d rows over %d columns as a dense block (%.1f %% full, %.1f %% of A's entries)]\n", nd, nc, 100.0 * ed / ((double)nd * nc), 100.0 * ed / nz);
}
/* builds the pattern of the factored matrix, orders it and maps the entries */
/* (18 October 2026, LP/SOCP) the boost of rotated cone k by theta: with (a, b) the rotated pair, a -> a e^-theta,
 * b -> b e^theta, an automorphism of the cone; on the second-order form's first two entries x' = B x,
 * B = [ch -sh; -sh ch], z' = B^-T z, the columns A B^-1, the costs B^-T c. The pattern is the same (both columns on
 * the same rows); the factored matrix's entries, the row copy and the cumulative angle are updated. */
static void cone_boost(Socp *S, int k, double th) {
    const int j = S->qs[k], len = S->Ap[j + 1] - S->Ap[j];
    const double ch = cosh(th), sh = sinh(th);
    double *x = S->x + j, *z = S->z + j;
    { const double a = ch * x[0] - sh * x[1], b = -sh * x[0] + ch * x[1]; x[0] = a; x[1] = b; }
    { const double a = ch * z[0] + sh * z[1], b = sh * z[0] + ch * z[1]; z[0] = a; z[1] = b; }
    { const double a = ch * S->c[j] + sh * S->c[j + 1], b = sh * S->c[j] + ch * S->c[j + 1]; S->c[j] = a; S->c[j + 1] = b; }
    double *A0 = S->Ax + S->Ap[j], *A1 = S->Ax + S->Ap[j + 1];
    for (int t = 0; t < len; t++) { const double a = ch * A0[t] + sh * A1[t], b = sh * A0[t] + ch * A1[t]; A0[t] = a; A1[t] = b; }
    if (S->bpos && S->bpos[k]) for (int t = 0; t < 2 * len; t++) S->F.Ax[S->bpos[k][t]] = S->Ax[S->Ap[j] + t];
    else if (S->rot && S->rot[k] && S->rG[k]) {      /* (a cone in the eigenbasis of its scaling: its columns' dense copy, from which kkt_fill writes A Q) */
        double *G = S->rG[k]; const int nr = S->rnr[k];
        for (int rr = 0; rr < nr; rr++) { const double a = ch * G[rr] + sh * G[rr + nr], b = sh * G[rr] + ch * G[rr + nr]; G[rr] = a; G[rr + nr] = b; }
    }
    if (S->Rq) for (int t = 0; t < 2 * len; t++) S->Rx[S->Rq[S->Ap[j] + t]] = S->Ax[S->Ap[j] + t];
    if (S->nd > 0) for (int t = 0; t < 2 * len; t++) {
        const int p = S->Ap[j] + t, col = t < len ? j : j + 1;
        if (S->Bq[p] >= 0) S->Bx[S->Bq[p]] = S->Ax[p]; else S->Dm[(size_t)S->dmap[S->Ai[p]] * S->ncd + S->dcm[col]] = S->Ax[p];
    }
    S->bth[k] += th; S->nboost++;
}
/* a point of the boosted problem back to the unboosted one: x = B^-1 x', z = B' z' on each boosted pair */
static void cone_unboost(const Socp *S, double *x, double *z) {
    if (!S->bth) return;
    for (int k = S->nq - S->nr_rot; k < S->nq; k++) {
        const double th = S->bth[k]; if (!(th != 0.0) || th != th) continue;
        const int j = S->qs[k]; const double ch = cosh(th), sh = sinh(th);
        if (x) { const double a = ch * x[j] + sh * x[j + 1], b = sh * x[j] + ch * x[j + 1]; x[j] = a; x[j + 1] = b; }
        if (z) { const double a = ch * z[j] - sh * z[j + 1], b = -sh * z[j] + ch * z[j + 1]; z[j] = a; z[j + 1] = b; }
    }
}
static int kkt_analyse(Socp *S);
/* the full system's structures freed (the pattern, the orders, both factors, the small cones' eigenbasis data, the maps), for
 * kkt_setup again; the iterate, the scaling and the problem stay */
static void kkt_teardown(Socp *S) {
    Ldl *F = &S->F;
    free(F->Ap); free(F->Ai); free(F->Ax); free(F->etree); free(F->Lnz); free(F->Lp); free(F->Li); free(F->Lx); free(F->D); free(F->Dinv);
    free(F->Lxl); free(F->Dl); free(F->Dinvl); free(F->fwl); free(F->lvp); free(F->lvn); free(F->Tp); free(F->Ti); free(F->Tx); free(F->Txl);
    free(F->sign); free(F->iw); free(F->bw); free(F->fw); free(F->perm); free(F->iperm);
    F->Ap = F->Ai = F->etree = F->Lnz = F->Lp = F->Li = F->sign = F->iw = F->perm = F->iperm = F->lvp = F->lvn = F->Tp = F->Ti = NULL;
    F->Ax = F->Lx = F->D = F->Dinv = F->Lxl = F->Dl = F->Dinvl = F->fwl = F->fw = F->Tx = F->Txl = NULL; F->bw = NULL;
    F->tr_ok = 0; F->dd = 0; F->lnz = 0; F->nlv = 0;
    if (S->sc) schol_free(S->sc);
    S->sc = NULL; S->sc_ok = 0; free(S->soff); S->soff = NULL; free(S->swork); S->swork = NULL;
    for (int k = 0; k < S->nq; k++) {
        if (S->kblk) free(S->kblk[k]);
        if (S->ku) free(S->ku[k]);
        if (S->kv) free(S->kv[k]);
        if (S->rrows) free(S->rrows[k]);
        if (S->rG) free(S->rG[k]);
        if (S->rapos) free(S->rapos[k]);
        if (S->bpos) { free(S->bpos[k]); S->bpos[k] = NULL; }
    }
    free(S->rot); free(S->rrows); free(S->rnr); free(S->rG); free(S->rpos0); free(S->Qr); free(S->rapos);
    free(S->kblk); free(S->ku); free(S->kv); free(S->kdiag); free(S->auxu); free(S->auxv);
    S->rot = NULL; S->rrows = NULL; S->rnr = NULL; S->rG = NULL; S->rpos0 = NULL; S->Qr = NULL; S->rapos = NULL;
    S->kblk = S->ku = S->kv = NULL; S->kdiag = NULL; S->auxu = S->auxv = NULL; S->anyrot = 0;
}
/* the solver's model of the time of a factorization (tf) and of a solve with the factor (tsv), from the factorization's
 * flops and the factor's nonzeros; the same model gives the ratio of the two (rr_est) */
static void kkt_model(const Socp *S, double *tf, double *tsv) {
    const double lz = (double)(S->sc ? S->slnz : S->F.lnz);
    *tf = S->cflops / (S->sc ? 6e9 : 1.2e9); *tsv = lz * (S->sc ? 3e-9 : 1.5e-9) + (double)S->Ap[S->n] * 3e-9 + (double)(S->n + S->m) * 2e-8;
}
static int kkt_setup(Socp *S) {
    const int n = S->n, m = S->m;
    S->auxu = (int *)xcalloc((size_t)S->nq, sizeof(int)); S->auxv = (int *)xcalloc((size_t)S->nq, sizeof(int));
    S->naux = 0;
    for (int k = 0; k < S->nq; k++) {
        if (S->q[k] > SOCP_DENSE_MAX) { S->auxv[k] = n + S->naux; S->auxu[k] = n + S->naux + 1; S->naux += 2; }
        else { S->auxu[k] = S->auxv[k] = -1; }
    }
    const int nk = n + S->naux + m;
    S->nk = nk;
    /* upper triangle in the original numbering, as triplets (row <= col) */
    /* small cones in the eigenbasis of their scaling ("rotated"): the d x d block -H of a cone whose
     * scaling has eigenvalues 1e9 and 1e-9 loses the small one to cancellation when it is factored
     * by scalar pivots, and the factorization is then no preconditioner (nql180: GMRES at its cap of
     * 10 steps in every solve; in double-double 3 steps). With H = Q diag(lambda) Q' known in closed
     * form (eigenvalues eta^2 (w0 + |w1|)^2, eta^2 / (w0 + |w1|)^2, eta^2), the unknown is Q' x: the
     * block is diagonal with the exact eigenvalues and the columns of A become A Q (the union of the
     * d patterns, values written every iteration); the operator GMRES solves stays in x. */
    S->rot = (unsigned char *)xcalloc((size_t)S->nq + 1, 1);
    S->rrows = (int **)xcalloc((size_t)S->nq + 1, sizeof(int *)); S->rnr = (int *)xcalloc((size_t)S->nq + 1, sizeof(int));
    S->rG = (double **)xcalloc((size_t)S->nq + 1, sizeof(double *)); S->rpos0 = (long *)xcalloc((size_t)S->nq + 1, sizeof(long));
    S->Qr = (double *)xcalloc(16 * ((size_t)S->nq + 1), sizeof(double)); S->rapos = (int **)xcalloc((size_t)S->nq + 1, sizeof(int *));
    long nrot_tr = 0;
    if (S->rot_mode == 1) {               /* (from the start, BRISK_CONEROT=1; or after the switch: rot_mode 1 then) */
        int *mark = (int *)malloc(sizeof(int) * (size_t)m); for (int i = 0; i < m; i++) mark[i] = -1;
        for (int k = 0; k < S->nq; k++) {
            const int d = S->q[k], s = S->qs[k];
            if (S->auxu[k] >= 0 || d < 2) continue;
            int cnt_ = 0;
            for (int j = 0; j < d; j++) for (int p = S->Ap[s + j]; p < S->Ap[s + j + 1]; p++) if (mark[S->Ai[p]] != k) { mark[S->Ai[p]] = k; cnt_++; }
            S->rot[k] = 1; S->anyrot = 1; S->rnr[k] = cnt_;
            S->rrows[k] = (int *)malloc(sizeof(int) * (size_t)(cnt_ + 1));
            int c = 0;
            for (int j = 0; j < d; j++) for (int p = S->Ap[s + j]; p < S->Ap[s + j + 1]; p++) if (mark[S->Ai[p]] != -2 - k) { mark[S->Ai[p]] = -2 - k; S->rrows[k][c++] = S->Ai[p]; }
            for (int a = 1; a < cnt_; a++) { const int v = S->rrows[k][a]; int b_ = a; while (b_ > 0 && S->rrows[k][b_ - 1] > v) { S->rrows[k][b_] = S->rrows[k][b_ - 1]; b_--; } S->rrows[k][b_] = v; }
            /* the d columns as a dense cnt x d matrix (by columns) */
            S->rG[k] = (double *)xcalloc((size_t)cnt_ * d + 1, sizeof(double));
            for (int j = 0; j < d; j++) for (int p = S->Ap[s + j]; p < S->Ap[s + j + 1]; p++) {
                int lo = 0, hi = cnt_ - 1; const int r = S->Ai[p];
                while (lo < hi) { const int mid = (lo + hi) >> 1; if (S->rrows[k][mid] < r) lo = mid + 1; else hi = mid; }
                S->rG[k][lo + (size_t)j * cnt_] += S->Ax[p];
            }
            nrot_tr += (long)d * cnt_ - (S->Ap[s + d] - S->Ap[s]);
        }
        free(mark);
    }
    S->nrot_tr = nrot_tr;
    return S->kkt_defer ? 0 : kkt_analyse(S);
}
/* the full system's pattern, ordering and analysis (the second part of kkt_setup; 20 October 2026, LP/SOCP: deferred
 * while the normal equations with dense linear columns serve, and done when the full system is first needed) */
static int kkt_analyse(Socp *S) {
    const double flop_ratio = 110.0;
    double nd0 = 0;                   /* dense rows found by the first AMD pass */
    const int n = S->n, m = S->m, nk = S->nk; const long nrot_tr = S->nrot_tr;
    S->kkt_defer = 0;
    long nz = (long)nk + S->Ap[n] + nrot_tr;
    for (int k = 0; k < S->nq; k++) nz += S->auxu[k] < 0 ? (long)S->q[k] * (S->q[k] - 1) / 2 : 2L * S->q[k];
    int *ti = (int *)malloc(sizeof(int) * (size_t)nz), *tj = (int *)malloc(sizeof(int) * (size_t)nz);
    long t = 0;
    for (int i = 0; i < nk; i++) { ti[t] = i; tj[t] = i; t++; }                 /* diagonals first: entry i is the diagonal of i */
    long *blk0 = (long *)xcalloc((size_t)S->nq, sizeof(long)), *u0 = (long *)xcalloc((size_t)S->nq, sizeof(long)), *v0 = (long *)xcalloc((size_t)S->nq, sizeof(long));
    for (int k = 0; k < S->nq; k++) {
        const int d = S->q[k], s = S->qs[k];
        if (S->auxu[k] < 0) { blk0[k] = t; for (int j = 0; j < d; j++) for (int i = 0; i < j; i++) { ti[t] = s + i; tj[t] = s + j; t++; } }
        else {
            v0[k] = t; for (int i = 0; i < d; i++) { ti[t] = s + i; tj[t] = S->auxv[k]; t++; }
            u0[k] = t; for (int i = 0; i < d; i++) { ti[t] = s + i; tj[t] = S->auxu[k]; t++; }
        }
    }
    const long a0 = t;
    {
        int k = 0;
        for (int j = 0; j < n; j++) {
            while (k < S->nq && S->qs[k] + S->q[k] <= j) k++;
            if (k < S->nq && S->rot[k] && j >= S->qs[k]) {
                if (j == S->qs[k]) { S->rpos0[k] = t; for (int i = 0; i < S->q[k]; i++) for (int r = 0; r < S->rnr[k]; r++) { ti[t] = S->qs[k] + i; tj[t] = n + S->naux + S->rrows[k][r]; t++; } }
                continue;
            }
            for (int p = S->Ap[j]; p < S->Ap[j + 1]; p++) { ti[t] = j; tj[t] = n + S->naux + S->Ai[p]; t++; }
        }
    }
    nz = t;
    /* ---- the ordering. AMD on the full symmetric pattern, first with a low threshold for the rows it sets
     * aside as dense (fast), then, when the factor is large, with its standard threshold. AMD removes the
     * dense rows before it orders the rest, which hides them from the degrees: with a few hundred dense rows
     * next to many small groups of unknowns (a regression with a dense data matrix) the fill was six times
     * that of the order that takes, in each group, the unknowns next to a dense row last. So for each AMD
     * order the variant with those unknowns moved behind the others is counted too; the order with the
     * least work is kept. */
    int *cnt = (int *)xcalloc((size_t)nk + 1, sizeof(int));
    for (long e = 0; e < nz; e++) { cnt[tj[e] + 1]++; if (ti[e] != tj[e]) cnt[ti[e] + 1]++; }
    for (int i = 0; i < nk; i++) cnt[i + 1] += cnt[i];
    int *fi = (int *)malloc(sizeof(int) * (size_t)(cnt[nk] + 1)), *nx = (int *)malloc(sizeof(int) * (size_t)nk);
    memcpy(nx, cnt, sizeof(int) * (size_t)nk);
    for (long e = 0; e < nz; e++) { fi[nx[tj[e]]++] = ti[e]; if (ti[e] != tj[e]) fi[nx[ti[e]]++] = tj[e]; }
    /* singleton rows (one column in A): AMD puts a node of degree 1 first, before its column; the
     * pivot of the row is then the dual regularization alone (1e-9) and the column of L below it
     * carries entries of 1e9. Such rows are left out of the graph AMD orders and placed right after
     * their column, where the pivot is delta + a^2 / H_jj. (The general rule below moves any row
     * that comes before all its columns.) */
    int *owner = (int *)malloc(sizeof(int) * (size_t)nk), *rmap = (int *)malloc(sizeof(int) * (size_t)nk), nsing = 0;
    for (int i = 0; i < nk; i++) owner[i] = -1;
    if (!getenv("BRISK_CONENOSING")) {
        int *rc = (int *)xcalloc((size_t)m, sizeof(int)), *rcol = (int *)malloc(sizeof(int) * (size_t)m);
        for (int j = 0; j < n; j++) for (int p = S->Ap[j]; p < S->Ap[j + 1]; p++) { rc[S->Ai[p]]++; rcol[S->Ai[p]] = j; }
        for (int i = 0; i < m; i++) if (rc[i] == 1) { owner[n + S->naux + i] = rcol[i]; nsing++; }
        free(rc); free(rcol);
    }
    int nred = 0;
    for (int i = 0; i < nk; i++) rmap[i] = owner[i] < 0 ? nred++ : -1;
    int *cntr = cnt, *fir = fi;
    if (nsing) {
        cntr = (int *)xcalloc((size_t)nred + 1, sizeof(int)); fir = (int *)malloc(sizeof(int) * (size_t)(cnt[nk] + 1));
        int w = 0;
        for (int i = 0; i < nk; i++) if (rmap[i] >= 0) {
            for (int p = cnt[i]; p < cnt[i + 1]; p++) if (rmap[fi[p]] >= 0) fir[w++] = rmap[fi[p]];
            cntr[rmap[i] + 1] = w;
        }
    }
    /* the singleton rows of each column, for the expansion of an order of the reduced graph */
    int *sing_ptr = (int *)xcalloc((size_t)nk + 2, sizeof(int)), *sing_row = (int *)malloc(sizeof(int) * (size_t)(nsing + 1));
    for (int i = 0; i < nk; i++) if (owner[i] >= 0) sing_ptr[owner[i] + 1]++;
    for (int i = 0; i < nk; i++) sing_ptr[i + 1] += sing_ptr[i];
    {
        int *nxs = (int *)malloc(sizeof(int) * (size_t)(nk + 1)); memcpy(nxs, sing_ptr, sizeof(int) * (size_t)(nk + 1));
        for (int i = 0; i < nk; i++) if (owner[i] >= 0) sing_row[nxs[owner[i]]++] = i;
        free(nxs);
    }
    if (dbg_on() && nsing) socp_printf("  [%d singleton rows placed after their columns]\n", nsing);
    Ldl *F = &S->F;
    memset(F, 0, sizeof(*F));
    F->n = nk;
    F->perm = (int *)malloc(sizeof(int) * (size_t)nk); F->iperm = (int *)malloc(sizeof(int) * (size_t)nk);
    F->Ap = (int *)malloc(sizeof(int) * ((size_t)nk + 1));
    F->Ai = (int *)malloc(sizeof(int) * (size_t)(nz + 1)); F->Ax = (double *)xcalloc((size_t)nz + 1, sizeof(double));
    int *pos = (int *)malloc(sizeof(int) * (size_t)nz);
    F->etree = (int *)malloc(sizeof(int) * (size_t)nk); F->Lnz = (int *)malloc(sizeof(int) * (size_t)nk);
    F->Lp = (int *)malloc(sizeof(int) * ((size_t)nk + 1));
    F->D = (double *)malloc(sizeof(double) * (size_t)nk); F->Dinv = (double *)malloc(sizeof(double) * (size_t)nk);
    F->iw = (int *)malloc(sizeof(int) * 3 * (size_t)nk); F->bw = (unsigned char *)malloc((size_t)nk); F->fw = (double *)malloc(sizeof(double) * (size_t)nk);
    F->sign = (int *)malloc(sizeof(int) * (size_t)nk);
    int *bestp = (int *)malloc(sizeof(int) * (size_t)nk), *cand = (int *)malloc(sizeof(int) * (size_t)nk);
    double bestfl = -1.0; long bestlnz = 0;
    int symb_fail = 0;
#define KKT_COUNT() do {                                                                                          \
        for (int i_ = 0; i_ < nk; i_++) F->iperm[F->perm[i_]] = i_;                                               \
        int *c_ = F->iw; memset(c_, 0, sizeof(int) * ((size_t)nk + 1));                                           \
        for (long e_ = 0; e_ < nz; e_++) { const int pi_ = F->iperm[ti[e_]], pj_ = F->iperm[tj[e_]]; c_[(pi_ > pj_ ? pi_ : pj_) + 1]++; } \
        for (int i_ = 0; i_ < nk; i_++) c_[i_ + 1] += c_[i_];                                                     \
        memcpy(F->Ap, c_, sizeof(int) * ((size_t)nk + 1));                                                        \
        for (long e_ = 0; e_ < nz; e_++) {                                                                        \
            const int pi_ = F->iperm[ti[e_]], pj_ = F->iperm[tj[e_]];                                             \
            const int col_ = pi_ > pj_ ? pi_ : pj_, row_ = pi_ > pj_ ? pj_ : pi_;                                 \
            pos[e_] = c_[col_]++; F->Ai[pos[e_]] = row_;                                                          \
        }                                                                                                         \
        symb_fail = ldl_symbolic(F);                                                                              \
        fl_ = 0; for (int i_ = 0; i_ < nk; i_++) fl_ += (double)F->Lnz[i_] * F->Lnz[i_];                          \
    } while (0)
    {
        static double dn[2] = { 3.0, 10.0 };
        { const char *e_ = getenv("BRISK_CONEAMDDENSE"); if (e_) dn[0] = atof(e_); }    /* (27 October 2026, a test: AMD's dense threshold of the first pass) */
        nd0 = 1;
        const long big = 1000000L;
        int var_gain = 0;                 /* the variant of the first pass reduced the work */
        for (int t = 0; t < 2; t++) {
            if (t == 1 && bestfl >= 0 && (bestlnz <= big || nd0 == 0)) break;       /* (no dense row at the low threshold: the same order) */
            /* the first pass's dense rows were the right ones when moving their neighbours last paid: no
             * second pass then (AMD at the higher threshold orders the rows between the two thresholds, at
             * a cost that can be large: 27 s on a lasso problem with 4,000 such rows, for the same order) */
            if (t == 1 && var_gain && !getenv("BRISK_CONEAMD2")) break;
            /* nor when no unknown has a degree between the two thresholds, in AMD's graph (AMD: dense above
             * max(16, a sqrt(n)), the degree without the diagonal) and in the variant's: the same dense rows,
             * the same order and the same variant (group lasso, 4,502 dense rows at both: 0.3 s of AMD and
             * two symbolic counts for the same factor) */
            if (t == 1 && !getenv("BRISK_CONEAMD2")) {
                const double ra = fmin((double)nred, fmax(16.0, dn[0] * sqrt((double)nred))), rb = fmin((double)nred, fmax(16.0, dn[1] * sqrt((double)nred)));
                const double va = fmax(16.0, dn[0] * sqrt((double)nk)), vb = fmax(16.0, dn[1] * sqrt((double)nk));
                long between = 0, betw_r = 0;
                for (int i = 0; i < nred; i++) { const double d_ = (double)(cntr[i + 1] - cntr[i] - 1); if (d_ > ra && d_ <= rb) betw_r++; }
                for (int i = 0; i < nk && !between; i++) { const double d_ = (double)(cnt[i + 1] - cnt[i] - 1); if (d_ > va && d_ <= vb) between++; }
                between += betw_r;
                if (!between) { if (dbg_on()) socp_printf("  [AMD (dense %g): no degree between the thresholds, the same order]\n", dn[1]); break; }
                /* nor when those rows cannot matter: the first pass put its k dense rows last, which adds at most k
                 * to every column count of the factor, so at most 2 k nnz(L) + k^2 nk to the work (the sum of the
                 * squared column counts); when that bound is below 1% of the work, the second pass can gain at most
                 * that (socp_svmsparse_50kx2000: two such rows next to the 2,000 feature columns, 1.3 s of AMD for
                 * the same factor) */
                if (bestfl > 0 && !getenv("BRISK_CONEAMD2B")) {
                    const double kb = (double)betw_r, bound = 2.0 * kb * (double)bestlnz + kb * kb * (double)nk;
                    if (bound < 0.01 * bestfl) { if (dbg_on()) socp_printf("  [AMD (dense %g) skipped: %ld rows between the thresholds, at most %.2e of the work %.2e]\n", dn[1], betw_r, bound, bestfl); break; }
                }
            }
            double ctl[AMD_CONTROL], info[AMD_INFO], fl_;
            amd_defaults(ctl);
            ctl[AMD_DENSE] = dn[t];
            const double ta = sc_time();
            {
                int *cred = nsing ? (int *)malloc(sizeof(int) * (size_t)(nred + 1)) : cand;
                if (amd_order(nred, cntr, fir, cred, ctl, info) < 0) for (int i = 0; i < nred; i++) cred[i] = i;
                if (dbg_on()) { const double thr_ = fmax(16.0, ctl[AMD_DENSE] * sqrt((double)nred)); double P_ = 0;
                    for (int i = 0; i < nred; i++) { const double d_ = (double)(cntr[i + 1] - cntr[i] - 1); if (d_ <= thr_) P_ += d_ * d_; }
                    socp_printf("  [AMD itself %.3fs on %d unknowns, %d entries; sum of the kept degrees squared %.2e]\n", sc_time() - ta, nred, cntr[nred], P_); }
                if (nsing) {
                    int *inv = (int *)malloc(sizeof(int) * (size_t)nk), w = 0;
                    for (int i = 0; i < nk; i++) if (rmap[i] >= 0) inv[rmap[i]] = i;
                    for (int r = 0; r < nred; r++) { const int i = inv[cred[r]]; cand[w++] = i; for (int p = sing_ptr[i]; p < sing_ptr[i + 1]; p++) cand[w++] = sing_row[p]; }
                    free(inv); free(cred);
                }
            }
            memcpy(F->perm, cand, sizeof(int) * (size_t)nk);
            KKT_COUNT();
            if (dbg_on()) socp_printf("  [AMD (dense %g) %.3fs: nnz(L) %ld, work %.2e, %g dense rows]\n", ctl[AMD_DENSE], sc_time() - ta, F->lnz, fl_, info[AMD_NDENSE]);
            if (!symb_fail && (bestfl < 0 || fl_ < bestfl)) { bestfl = fl_; bestlnz = F->lnz; memcpy(bestp, cand, sizeof(int) * (size_t)nk); }
            if (t == 0) nd0 = info[AMD_NDENSE];
            if (info[AMD_NDENSE] > 0) {
                /* the variant: [not next to a dense row] [next to one] [the dense rows], AMD's order within each */
                const double thr = fmax(16.0, ctl[AMD_DENSE] * sqrt((double)nk));
                unsigned char *cls = F->bw;
                for (int i = 0; i < nk; i++) cls[i] = (double)(cnt[i + 1] - cnt[i] - 1) > thr ? 2 : 0;
                for (int i = 0; i < nk; i++) if (cls[i] == 2) for (int p_ = cnt[i]; p_ < cnt[i + 1]; p_++) if (cls[fi[p_]] == 0) cls[fi[p_]] = 1;
                for (int i = 0; i < nk; i++) if (owner[i] >= 0) cls[i] = cls[owner[i]];
                int w = 0;
                for (int c_ = 0; c_ < 3; c_++) for (int i = 0; i < nk; i++) if (cls[cand[i]] == c_) F->perm[w++] = cand[i];
                F->cap = bestfl >= 0 ? 2 * bestlnz : 0;
                KKT_COUNT();
                F->cap = 0;
                if (dbg_on()) socp_printf("  [ ... unknowns next to dense rows last: nnz(L) %ld, work %.2e%s]\n", F->lnz, fl_, symb_fail ? " (given up)" : "");
                if (t == 0 && !symb_fail && fl_ < 0.95 * bestfl) var_gain = 1;
                if (!symb_fail && fl_ < bestfl) { bestfl = fl_; bestlnz = F->lnz; memcpy(bestp, F->perm, sizeof(int) * (size_t)nk); }
            }
        }
        /* (20 October 2026, LP/SOCP) nested dissection (nd.c, the LP method's ordering) as one more candidate, kept when its
         * count is lower: the FIR filters' full system has 151 dense columns and AMD's factor 40 million entries where
         * MOSEK's graph-partitioning order gives 4 million. Tried when the best order so far costs 1e9 flops or more and
         * the pattern is not dense (BRISK_CONEND=0 off, =1 always). */
        /* (on the mesh-like problems nd.c's order costs less to factor (db-plane-strain-prism 1.35e8 -> 5.85e7 flops with four
         * bisections a level, 6.6e7 with one, 8.6e7 with one and leaves of 5,000; joint_FC_7 3.5e8 -> 1.87e8; nql180old 2.3e9
         * -> 1.57e9), and with one bisection and large leaves the order costs a tenth of the time (prism 10 -> 0.9 s); on
         * the FIR filter it has more than twice AMD's fill, on TV denoising 5.8 against 4.8 million entries: the count
         * decides. MOSEK's smaller factor on the FIR filter is its separation of the 151 dense columns, not the order.
         * BRISK_CONEND=0 off, =1 always with nd.c's own settings, =2 (the default) one bisection and leaves of 5,000 when the
         * best order costs 1e8 flops or more and the pattern is not dense.) */
        /* (kept off, 20 October 2026, the night: the 122 cone programs with the rule on against off, the same binary alone on
         * the machine: time 1.030 (1.06 over 5 s), no file better than 0.73 and several worse: nql180 19.8 -> 28.8 s (21 -> 23
         * iterations), socp_mpc_30x8x2000 5.3 -> 8.2, TV denoising 1.17, nql180old 8.6 -> 10.7; the lower count did not turn
         * into time on the supernodal factorizations, and the order's own cost stayed; TRICKS 8, IDEAS.) */
        { static int ndt = -1; if (ndt < 0) { const char *e_ = getenv("BRISK_CONEND"); ndt = e_ ? atoi(e_) : 0; }
          const double dens = (double)(cnt[nk] - nk) / fmax(1.0, (double)nk * (nk - 1.0));
          if (ndt == 1 || (ndt == 2 && nk >= 1000 && bestfl >= 1e8 && dens < 0.05)) {
            const double ta = sc_time(); double fl_;
            extern void nd_set_params(int ntry, int leaf);
            nd_set_params(ndt == 2 ? 1 : -1, ndt == 2 ? 5000 : -1);       /* (one bisection a level, leaves of 5,000: a tenth of the time, most of the gain) */
            int *xadj = (int *)malloc(sizeof(int) * ((size_t)nred + 1)), *adjncy = (int *)malloc(sizeof(int) * ((size_t)cntr[nred] + 1));
            int *cred = (int *)malloc(sizeof(int) * ((size_t)nred + 1));
            size_t w = 0; xadj[0] = 0;
            for (int i = 0; i < nred; i++) { for (int p = cntr[i]; p < cntr[i + 1]; p++) if (fir[p] != i) adjncy[w++] = fir[p]; xadj[i + 1] = (int)w; }
            extern int nd_order(int n, const int *xadj, const int *adjncy, int *perm);
            int st_ = nd_order(nred, xadj, adjncy, cred);
            free(xadj); free(adjncy);
            nd_set_params(-1, -1);
            if (st_ == 0) {
                if (nsing) {
                    int *inv = (int *)malloc(sizeof(int) * (size_t)nk), w2 = 0;
                    for (int i = 0; i < nk; i++) if (rmap[i] >= 0) inv[rmap[i]] = i;
                    for (int r = 0; r < nred; r++) { const int i = inv[cred[r]]; cand[w2++] = i; for (int p = sing_ptr[i]; p < sing_ptr[i + 1]; p++) cand[w2++] = sing_row[p]; }
                    free(inv);
                } else memcpy(cand, cred, sizeof(int) * (size_t)nred);
                { int okp = 1; unsigned char *seen = (unsigned char *)calloc((size_t)nk + 1, 1); for (int i = 0; i < nk && okp; i++) { if (cand[i] < 0 || cand[i] >= nk || seen[cand[i]]) okp = 0; else seen[cand[i]] = 1; } free(seen);
                  if (!okp) { if (dbg_on()) socp_printf("  [nested dissection: not a permutation (nred %d, nk %d, nsing %d)]\n", nred, nk, nsing); st_ = 1; } }
                if (st_ == 0) {
                memcpy(F->perm, cand, sizeof(int) * (size_t)nk);
                F->cap = bestfl >= 0 ? 2 * bestlnz : 0;
                KKT_COUNT();
                F->cap = 0;
                if (dbg_on()) socp_printf("  [nested dissection %.3fs: nnz(L) %ld, work %.2e%s]\n", sc_time() - ta, F->lnz, fl_, symb_fail ? (symb_fail == 2 ? " (given up: more fill than twice the best)" : " (the count failed)") : "");
                if (!symb_fail && (bestfl < 0 || fl_ < bestfl)) { bestfl = fl_; bestlnz = F->lnz; memcpy(bestp, cand, sizeof(int) * (size_t)nk); S->nd_used = 1; }
                }
            }
            free(cred);
          } }
        /* free variables: their pivots are the small regularization unless they come after the rows
         * they appear in. The variant with the free variables last (the order within the two groups
         * kept) is taken when its work is not much larger. */
        S->free_last = 0;
        if (S->nf > 0 && bestfl >= 0 && !getenv("BRISK_CONENOFREELAST")) {
            double fl_; int w = 0;
            for (int i = 0; i < nk; i++) if (bestp[i] >= S->nf) F->perm[w++] = bestp[i];
            for (int i = 0; i < nk; i++) if (bestp[i] < S->nf) F->perm[w++] = bestp[i];
            const double lim = 1.1;        /* (a random LP: the variant with 1.4x the work took 3x the time) */
            F->cap = (long)(lim * 1.5 * (double)bestlnz) + 100000;
            KKT_COUNT();
            F->cap = 0;
            if (dbg_on()) socp_printf("  [ ... free variables last: nnz(L) %ld, work %.2e%s]\n", F->lnz, fl_, symb_fail ? " (given up)" : "");
            if (!symb_fail && fl_ <= lim * bestfl + 1e6) { bestfl = fl_; bestlnz = F->lnz; memcpy(bestp, F->perm, sizeof(int) * (size_t)nk); S->free_last = 1; }
        }
        if (bestfl < 0) for (int i = 0; i < nk; i++) bestp[i] = i;
        double fl_;
        /* no row before all its columns: a row eliminated before every column it has gets the pivot
         * delta (1e-9), and the entries of L below it are a_ij / delta: the factorization is lost
         * there (nql60: 396 such rows, GMRES at its cap). Such a row is moved to right after the
         * first of its columns in the order (a local move: little fill; the pivot is then
         * delta + a^2 / H_jj). nql60: 568 -> 349 backsolves, 17 -> 15 iterations.
         * (24 October 2026, the SOCP developer) and after the last of its columns when that order's count is not larger:
         * after the first column j alone the pivot is delta + a_rj^2 / |d_j|, which is tiny when a_rj is small against the
         * pivot d_j of a dense unknown (the antenna arrays' two rows of the main lobe, a(theta_0)' w = 1, have only the
         * 400 weights as columns, and those are the dense unknowns last: pivot 4e-8 against d_j of about -6,000, the
         * factored system wrong by 1e-6 to 1e-4 in the weights and two GMRES steps on every solve); after the last
         * column the pivot is delta + a_r' S^-1 a_r, the row's whole Schur complement. socp_antenna_200x6000 (SOCP):
         * 323 -> 218 solves, the whole run 14.5 -> 12.9 s; BRISK_CONEROWLAST=0: after the first column alone */
        if (!getenv("BRISK_CONENOROWFIX")) {
            int *rpos = (int *)malloc(sizeof(int) * (size_t)nk), *after = (int *)malloc(sizeof(int) * (size_t)nk), nmoved = 0;
            for (int i = 0; i < nk; i++) { rpos[bestp[i]] = i; after[i] = -1; }
            int *cmin = (int *)malloc(sizeof(int) * (size_t)m);
            for (int i = 0; i < m; i++) cmin[i] = -1;
            for (int j = 0; j < n; j++) for (int p = S->Ap[j]; p < S->Ap[j + 1]; p++) { const int i = S->Ai[p]; if (cmin[i] < 0 || rpos[j] < rpos[cmin[i]]) cmin[i] = j; }
            for (int i = 0; i < m; i++) { const int r = n + S->naux + i; if (cmin[i] >= 0 && rpos[r] < rpos[cmin[i]] && owner[r] < 0) { after[r] = cmin[i]; nmoved++; } }
            /* a local repair, not a reordering. (24 October 2026, the default: every such row, BRISK_CONEROWFRAC=0.2 the
             * former cut-off) Up to then nothing was moved when more than a fifth of the rows were (tv_100000: half of them;
             * AMD's order then the right one, 17 instead of 20 iterations): a threshold. Without it, on the 158 SOCP files no
             * status changed, iterations 2,569 -> 2,562, and the six files it changes all faster in paired timing (two
             * threads, the whole run): beam7 106 -> 80 s, socp_rocket_4000 4.6 -> 2.5 s, strictmin_2D_43_primal 6.3 -> 5.0 s,
             * 2013_firL1Linfeps 62.6 -> 59.1 s, socp_portimpact_5000x50 2.9 -> 2.4 s, socp_branchflow_20000 3.6 -> 3.2 s
             * (bench/socp_start/rowfr_time.txt); socp_tv_450 20.6 -> 21.8 s in one pair */
            { const char *e_ = getenv("BRISK_CONEROWFRAC"); const double fr_ = e_ ? atof(e_) : 1.0;
            if (nmoved > fr_ * m) { if (dbg_on()) socp_printf("  [%d rows before all their columns: more than a fifth of the rows, left in place]\n", nmoved); nmoved = 0; }
            }
            if (nmoved) {
                int *head = (int *)malloc(sizeof(int) * (size_t)nk), *next = (int *)malloc(sizeof(int) * (size_t)nk), *tail = (int *)malloc(sizeof(int) * (size_t)nk);
                /* F->perm = bestp with each moved row r right after the unknown after[r] */
                #define ROW_MOVE() do { for (int i = 0; i < nk; i++) head[i] = tail[i] = -1; \
                    for (int i = 0; i < nk; i++) { const int r = bestp[i]; if (after[r] >= 0) { const int j = after[r]; next[r] = -1; if (head[j] < 0) head[j] = r; else next[tail[j]] = r; tail[j] = r; } } \
                    int w = 0; \
                    for (int i = 0; i < nk; i++) { const int v = bestp[i]; if (after[v] >= 0) continue; F->perm[w++] = v; for (int r = head[v]; r >= 0; r = next[r]) F->perm[w++] = r; } } while (0)
                int nlast = 0;
                { static int rl = -1; if (rl < 0) { const char *e_ = getenv("BRISK_CONEROWLAST"); rl = e_ ? atoi(e_) : 1; }
                  if (rl) {     /* the last column of each moved row, when it is not the first */
                      int *cmax = cmin; for (int i = 0; i < m; i++) cmax[i] = -1;
                      for (int j = 0; j < n; j++) for (int p = S->Ap[j]; p < S->Ap[j + 1]; p++) { const int i = S->Ai[p]; if (cmax[i] < 0 || rpos[j] > rpos[cmax[i]]) cmax[i] = j; }
                      int *last = (int *)malloc(sizeof(int) * (size_t)nk);
                      for (int v = 0; v < nk; v++) { last[v] = after[v]; if (after[v] >= 0 && cmax[v - n - S->naux] != after[v]) { last[v] = cmax[v - n - S->naux]; nlast++; } }
                      if (nlast) {
                          double fl_first;
                          ROW_MOVE(); F->cap = 0; KKT_COUNT(); fl_first = symb_fail ? -1 : fl_;
                          int *sv = after; after = last; last = sv;
                          /* (the count abandoned once its fill exceeds the first order's: strictmin_2D_43_primal, 10,953 such
                           * rows, 3.6e11 against 1.6e7 flops, counted in full 0.16 s) */
                          F->cap = fl_first >= 0 ? F->lnz : 0;
                          const long cap_ = F->cap;
                          ROW_MOVE(); KKT_COUNT(); F->cap = 0;
                          const int abandoned = symb_fail != 0;
                          if (symb_fail || fl_first < 0 || fl_ > fl_first) { sv = after; after = last; last = sv; symb_fail = 0; nlast = -nlast; }
                          if (dbg_on()) { if (abandoned) socp_printf("  [%d rows after their last column instead of their first: more fill than %ld, the count abandoned, not taken]\n", -nlast, cap_);
                                          else socp_printf("  [%d rows after their last column instead of their first: work %.2e against %.2e, %s]\n", nlast < 0 ? -nlast : nlast, fl_, fl_first, nlast > 0 ? "taken" : "not taken"); }
                      }
                      free(last);
                  } }
                ROW_MOVE();
                #undef ROW_MOVE
                memcpy(bestp, F->perm, sizeof(int) * (size_t)nk);
                free(head); free(next); free(tail);
                if (dbg_on()) socp_printf("  [%d rows moved after their first column%s]\n", nmoved, nlast > 0 ? " or their last" : "");
            }
            free(rpos); free(after); free(cmin);
        }
        memcpy(F->perm, bestp, sizeof(int) * (size_t)nk);
        { const double tk_ = sc_time();
        KKT_COUNT();
        if (dbg_on()) socp_printf("  [the final count %.3fs]\n", sc_time() - tk_); }
        S->cflops = fl_;
    }
#undef KKT_COUNT
    free(fi); free(nx); free(cand); free(owner); free(rmap); free(sing_ptr); free(sing_row); if (nsing) { free(cntr); free(fir); }
    if (symb_fail) { free(bestp); free(cnt); free(pos); free(ti); free(tj); free(blk0); free(u0); free(v0); return 1; }
    /* the supernodal factorization, with the same order, when the factor is large or dense */
    {
        const char *e = getenv("BRISK_CONEBACKEND");
        /* (dense rows: the supernodal update batches the data rows into one product per tile, svm_5000 1.0 -> 0.65 s) */
        /* (19 October 2026, LP/SOCP: also when both are moderately high, columns of L longer than 15 on average and more
         * than 60 flops per entry: an LP with 20 dense columns handed over by the LP method, 26 and 87, factorizations
         * 3.8 -> 1.8 s; DIMACS nql60old 0.61 -> 0.45 s; TV denoising and truss design, columns of 6, stay by columns,
         * where the supernodal code was slower) */
        const int want = e ? atoi(e) : (F->lnz > 40000 && (F->lnz > 40L * nk || S->cflops > flop_ratio * (double)F->lnz || (F->lnz > 15L * nk && S->cflops > 60.0 * (double)F->lnz) || (nd0 >= 32 && S->Ap[n] >= 20L * m)));
        if (want && nk >= 64) {
            int *deg = (int *)xcalloc((size_t)nk + 1, sizeof(int)), **nbr = (int **)malloc(sizeof(int *) * ((size_t)nk + 1));
            for (long q = nk; q < nz; q++) { deg[ti[q]]++; deg[tj[q]]++; }
            int *pool = (int *)malloc(sizeof(int) * (size_t)(2 * (nz - nk) + 1));
            { long o = 0; for (int i = 0; i < nk; i++) { nbr[i] = pool + o; o += deg[i]; deg[i] = 0; } }
            for (long q = nk; q < nz; q++) { nbr[ti[q]][deg[ti[q]]++] = tj[q]; nbr[tj[q]][deg[tj[q]]++] = ti[q]; }
            const int om = schol_set_amd(2);
            schol_set_perm(bestp);
            const double tsa_ = sc_time();
            const double oam = schol_set_amalg(cone_amalg((double)F->lnz / (double)nk));
            S->sc = schol_analyze_adj(nk, deg, nbr, (size_t)1 << 31);
            schol_set_amalg(oam);
            if (dbg_on()) socp_printf("  [supernodal analysis %.3fs]\n", sc_time() - tsa_);
            schol_set_perm(NULL);
            schol_set_amd(om);
            free(deg); free(nbr); free(pool);
            if (S->sc) S->slnz = (long)schol_nnz(S->sc);
        }
    }
    free(bestp);
    if (S->sc) {
        signed char *sg = (signed char *)malloc((size_t)nk + 1);
        for (int i = 0; i < nk; i++) sg[i] = i < n ? -1 : i < n + S->naux ? (((i - n) & 1) ? 1 : -1) : 1;
        schol_set_signs(S->sc, sg); schol_set_tinypiv(S->sc, 0.0);
        /* (23 October 2026, the SOCP developer) the factorization over the assembled values, which kkt_factor zeroes and
         * writes again before every factorization (as the LP method does): one array of the factor's size instead of two,
         * the same arithmetic (socp_svmsparse_200kx20k (SOCP): 3.2 GB each) */
        if (!getenv("BRISK_CONENOINPLACE")) schol_set_inplace(S->sc, 1);
        free(sg);
        S->soff = (size_t *)malloc(sizeof(size_t) * (size_t)(nz + 1)); S->knz = nz;
        int bad = 0; const double tso_ = sc_time();
        /* (19 October 2026, LP/SOCP: on the threads, each lookup independent; a group lasso's 4.5e6 entries 0.27 -> 0.08 s) */
        #pragma omp parallel for schedule(static) num_threads(g_cone_nt) reduction(|:bad) if (nz >= 1000000 && g_cone_nt > 1 && !omp_in_parallel())
        for (long q = 0; q < nz; q++) { S->soff[pos[q]] = schol_offset(S->sc, (size_t)ti[q], (size_t)tj[q]); if (S->soff[pos[q]] == (size_t)-1) bad = 1; }
        if (dbg_on()) socp_printf("  [offsets %.3fs]\n", sc_time() - tso_);
        if (bad) { schol_free(S->sc); S->sc = NULL; free(S->soff); S->soff = NULL; }
        else S->swork = (double *)xcalloc(2 * (size_t)nk + 2, sizeof(double));
    }
    free(F->Li); free(F->Lx); F->Li = NULL; F->Lx = NULL;     /* (allocated at the column factorization's first use: ldl_alloc) */
    for (int i = 0; i < nk; i++) F->sign[F->iperm[i]] = i < n ? -1 : i < n + S->naux ? (((i - n) & 1) ? 1 : -1) : 1;
    /* maps */
    S->kdiag = (int *)malloc(sizeof(int) * (size_t)nk);
    for (int i = 0; i < nk; i++) S->kdiag[i] = pos[i];
    S->kblk = (int **)xcalloc((size_t)S->nq, sizeof(int *)); S->ku = (int **)xcalloc((size_t)S->nq, sizeof(int *)); S->kv = (int **)xcalloc((size_t)S->nq, sizeof(int *));
    for (int k = 0; k < S->nq; k++) {
        const int d = S->q[k], s = S->qs[k];
        if (S->auxu[k] < 0) {
            S->kblk[k] = (int *)malloc(sizeof(int) * (size_t)(d * (d + 1) / 2));
            int c = 0; long e = blk0[k];
            for (int j = 0; j < d; j++) for (int i = 0; i <= j; i++) S->kblk[k][c++] = i == j ? pos[s + j] : pos[e++];
        } else {
            S->ku[k] = (int *)malloc(sizeof(int) * (size_t)d); S->kv[k] = (int *)malloc(sizeof(int) * (size_t)d);
            for (int i = 0; i < d; i++) { S->kv[k][i] = pos[v0[k] + i]; S->ku[k][i] = pos[u0[k] + i]; }
        }
    }
    /* the A entries of the unrotated columns do not change: written once; the rotated cones keep
     * the positions of theirs (written by kkt_fill) */
    {
        long e = a0; int k = 0;
        for (int j = 0; j < n; j++) {
            while (k < S->nq && S->qs[k] + S->q[k] <= j) k++;
            if (k < S->nq && S->rot[k] && j >= S->qs[k]) {
                if (j == S->qs[k]) { const long np_ = (long)S->q[k] * S->rnr[k]; S->rapos[k] = (int *)malloc(sizeof(int) * (size_t)(np_ + 1)); for (long q_ = 0; q_ < np_; q_++) S->rapos[k][q_] = pos[e++]; }
                continue;
            }
            if (S->bpos && k < S->nq && k >= S->nq - S->nr_rot && (j == S->qs[k] || j == S->qs[k] + 1)) {   /* (the boost of a rotated cone rewrites these) */
                const int off = j == S->qs[k] ? 0 : S->Ap[S->qs[k] + 1] - S->Ap[S->qs[k]];
                if (!S->bpos[k]) S->bpos[k] = (int *)malloc(sizeof(int) * (size_t)(S->Ap[S->qs[k] + 2] - S->Ap[S->qs[k]] + 1));
                for (int p = S->Ap[j]; p < S->Ap[j + 1]; p++) S->bpos[k][off + p - S->Ap[j]] = pos[e + p - S->Ap[j]];
            }
            for (int p = S->Ap[j]; p < S->Ap[j + 1]; p++) F->Ax[pos[e++]] = S->Ax[p];
        }
    }
    free(ti); free(tj); free(cnt); free(pos); free(blk0); free(u0); free(v0);
    return 0;
}

static void socp_free(Socp *S) {
    Ldl *F = &S->F;
    free(F->Ap); free(F->Ai); free(F->Ax); free(F->etree); free(F->Lnz); free(F->Lp); free(F->Li); free(F->Lx); free(F->D); free(F->Dinv);
    free(F->Lxl); free(F->Dl); free(F->Dinvl); free(F->fwl);
    free(F->lvp); free(F->lvn); free(F->Tp); free(F->Ti); free(F->Tx); free(F->Txl);
    free(F->sign); free(F->iw); free(F->bw); free(F->fw); free(F->perm); free(F->iperm);
    if (S->sc) schol_free(S->sc);
    free(S->soff); free(S->swork);
    for (int k = 0; k < S->nq; k++) { if (S->kblk) free(S->kblk[k]); if (S->ku) free(S->ku[k]); if (S->kv) free(S->kv[k]); }
    for (int k = 0; k < S->nq; k++) { if (S->rrows) free(S->rrows[k]); if (S->rG) free(S->rG[k]); if (S->rapos) free(S->rapos[k]); }
    free(S->rot); free(S->rrows); free(S->rnr); free(S->rG); free(S->rpos0); free(S->Qr); free(S->rapos);
    free(S->kblk); free(S->ku); free(S->kv); free(S->kdiag); free(S->auxu); free(S->auxv);
    free(S->q); free(S->qs); free(S->Ap); free(S->Ai); free(S->Ax); free(S->Rp); free(S->Ri); free(S->Rx); free(S->R0x); free(S->Rq);     /* (Rq: 28 October 2026, a leak found by the address sanitizer, 4 bytes per entry of A per solve at two threads or more on 100,000 entries) */
    if (S->nd > 0) { free(S->drow); free(S->dmap); free(S->dcol); free(S->dcm); free(S->Dm); free(S->Bp); free(S->Bi); free(S->Bq); free(S->Bx); free(S->dxs); free(S->dacc); free(S->dpart); } free(S->b); free(S->c);
    free(S->Dr); free(S->Ec); free(S->x); free(S->y); free(S->z); free(S->h); free(S->eta); free(S->wbar); free(S->lam);
    free(S->w1); free(S->w2); free(S->w3); free(S->w4); free(S->rk); free(S->sk); ne_free(S); free(S->swork2);
    free(S->lastx); free(S->gs); free(S->gV); free(S->gZ); free(S->gH); free(S->gcs); free(S->fcone);
}

void socp_default_opts(SocpOpts *o) { o->tol = 1e-8; o->maxit = 200; o->timelimit = 0; o->verbose = 1; o->equil = getenv("BRISK_CONEEQ") ? atoi(getenv("BRISK_CONEEQ")) : 1; o->step = 0; o->attempt = 0; o->wx = o->wy = o->wz = NULL; o->wlam = 0.99; o->wsave = 0; o->cutoff = INFINITY; o->noretry = 0; }
const char *socp_status_str(int s) {
    return s == SOCP_OPTIMAL ? "OPTIMAL" : s == SOCP_PINF ? "PRIMAL INFEASIBLE" : s == SOCP_DINF ? "DUAL INFEASIBLE" : s == SOCP_MAXIT ? "ITERATION LIMIT"
         : s == SOCP_TIME ? "TIME LIMIT" : s == SOCP_REDUCED ? "SOLVED TO REDUCED ACCURACY" : s == SOCP_NUMERR ? "NUMERICAL DIFFICULTIES"
         : s == SOCP_CUTOFF ? "STOPPED BY THE CUTOFF" : "INPUT ERROR";
}
/* (28 October 2026) the status line's word: a solve stopped by the calling thread's flag (socp_stop_call) is "STOPPED", not at its
 * time limit; the caller's summary says why */
static const char *socp_status_line(int s) { return s == SOCP_TIME && socp_stop_call && *socp_stop_call ? "STOPPED" : socp_status_str(s); }

/* errors of a point of the problem as given (after the rotated cones' map): in the scaled problem x, y, z are
 * divided by tau; the scaling is undone inside. e[0] primal, e[1] dual, e[2] gap, e[3] complementarity */
typedef struct { const double *b0, *c0; double nb, nc; } Orig;

/* The direction of the homogeneous model for a complementarity term alone (no residuals): tv holds
 * the term per linear variable and, per cone, in the scaled space; ttk that of tau kappa. Used by
 * the centrality correctors. */
typedef struct {
    Socp *S; const double *px, *py; double denom0;
    double *r3, *t1, *t2, *rhsx, *rhsy, *qx, *qy;
} DirCtx;
static void corr_solve(DirCtx *C, const double *tv, double ttk, double *dx, double *dy, double *dz, double *dtau, double *dkap, int maxref) {
    Socp *S = C->S;
    const int n = S->n, m = S->m;
    double *r3 = C->r3, *t1 = C->t1, *t2 = C->t2;
    for (int i = 0; i < S->nf; i++) r3[i] = 0.0;
    for (int i = S->nf; i < S->nf + S->nl; i++) r3[i] = tv[i] / S->x[i];
    for (int k = 0; k < S->nq; k++) {
        const int d = S->q[k], s = S->qs[k]; const double *l = S->lam + s, *dd = tv + s;
        double det = l[0] * l[0], ld = l[0] * dd[0];
        for (int i = 1; i < d; i++) { det -= l[i] * l[i]; ld -= l[i] * dd[i]; }
        double *pp = t1 + s;                               /* lambda \ dd */
        pp[0] = ld / det;
        for (int i = 1; i < d; i++) pp[i] = (dd[i] - pp[0] * l[i]) / l[0];
        W_apply(S, k, 0, pp, r3 + s);
    }
    for (int j = 0; j < n; j++) C->rhsx[j] = -r3[j];
    for (int i = 0; i < m; i++) C->rhsy[i] = 0.0;
    kkt_solve(S, C->rhsx, C->rhsy, C->qx, C->qy, maxref);
    *dtau = (ttk / S->tau - dotv(m, S->b, C->qy) + dotv(n, S->c, C->qx)) / (C->denom0 + S->kap / S->tau);
    for (int j = 0; j < n; j++) dx[j] = C->qx[j] + *dtau * C->px[j];
    for (int i = 0; i < m; i++) dy[i] = C->qy[i] + *dtau * C->py[i];
    *dkap = (ttk - S->kap * *dtau) / S->tau;
    H_apply(S, dx, t2);
    for (int j = 0; j < n; j++) dz[j] = j < S->nf ? 0.0 : r3[j] - t2[j];
}

/* the violation of the cone K by x (0 when x is in the closed cone; the free part unconstrained) */
static double cone_viol(const Socp *S, const double *x) {
    double v = 0;
    for (int i = S->nf; i < S->nf + S->nl; i++) v = fmax(v, -x[i]);
    for (int k = 0; k < S->nq; k++) {
        const int d = S->q[k], s = S->qs[k]; double xn = 0;
        for (int i = 1; i < d; i++) xn += x[s + i] * x[s + i];
        v = fmax(v, sqrt(xn) - x[s]);
    }
    return v;
}
/* Infeasibility certificates by projection (ISSUES 42). Near a certificate the iterate's ray is
 * accurate to 1e-3 .. 1e-5 long before the 1e-8 the test asks for, and the iteration crawls
 * there (mpc_1500: 63 iterations, GMRES at its cap). Dual infeasibility: the correction dx with
 * A(x + dx) = 0 is one solve with the factored KKT matrix (dx = H^-1 A' dy, an H-weighted least
 * norm correction); when x + dx is still in K and c'(x + dx) < 0 it is a certificate up to the
 * solve's residual. Primal infeasibility: z := -A'y makes A'y + z = 0 exactly; a certificate when
 * z is in K* and the free part of A'y vanishes. Returns 1 and replaces x (or z) when it worked. */
static int cert_project_dual(Socp *S, double tol, double *dx, double *dy) {
    const int n = S->n, m = S->m;
    double *w = S->w1, *xt = S->w3, *z0 = S->w4;
    memcpy(xt, S->x, sizeof(double) * (size_t)n);
    const double nx = nrmi(n, S->x);
    /* alternating projections: onto A x = 0 (one solve), then into the cone (the entries the
     * correction pushed out are set to a small margin: the ray has zeros there), a few rounds */
    for (int round = 0; round < 3; round++) {
        for (int i = 0; i < m; i++) w[i] = 0.0;
        A_mul(S, 1.0, xt, w);
        for (int i = 0; i < m; i++) w[i] = -w[i];
        for (int j = 0; j < n; j++) z0[j] = 0.0;
        kkt_solve(S, z0, w, dx, dy, S->refine_max);
        for (int j = 0; j < n; j++) xt[j] += dx[j];
        const double viol = cone_viol(S, xt), nxt = nrmi(n, xt);
        const double cx = dotv(n, S->c, xt);
        for (int i = 0; i < m; i++) w[i] = 0.0;
        A_mul(S, 1.0, xt, w);
        if (dbg_on()) socp_printf("      [certificate projection %d: cone violation %.1e (|x| %.1e), c'x %.2e, |A x| %.2e (need %.1e), solve residual %.1e]\n", round, viol, nxt, cx, nrmi(m, w), tol * fmax(-cx, 0), S->last_res);
        if (!(cx < 0) || !isfinite(cx)) return 0;
        /* a certificate up to the tolerance: A x and the cone violation relative to the scale of x,
         * as the optimality measures are (the point is returned as it is: zeroing the entries
         * outside the cone would move A x by far more) */
        if (viol <= tol * nxt && nrmi(m, w) <= tol * (-cx)) { memcpy(S->x, xt, sizeof(double) * (size_t)n); return 1; }
        if (viol == 0) return 0;                       /* in the cone but not accurate enough: the solve is the limit */
        /* onto the cone: the entries the correction pushed out go to the boundary (the ray has
         * zeros there; a certificate needs the closed cone only) */
        for (int i = S->nf; i < S->nf + S->nl; i++) if (xt[i] < 0) xt[i] = 0.0;
        for (int k = 0; k < S->nq; k++) {
            const int d = S->q[k], q0 = S->qs[k]; double xn = 0;
            for (int i = 1; i < d; i++) xn += xt[q0 + i] * xt[q0 + i];
            xn = sqrt(xn);
            if (xt[q0] < xn) { if (xt[q0] + xn <= 0) { for (int i = 0; i < d; i++) xt[q0 + i] = 0.0; } else { const double f = 0.5 * (xt[q0] + xn); for (int i = 1; i < d; i++) xt[q0 + i] *= f / xn; xt[q0] = f; } }
        }
    }
    (void)nx;
    return 0;
}
static int cert_project_primal(Socp *S, double tol, double byv) {
    const int n = S->n;
    double *zt = S->w3;
    for (int j = 0; j < n; j++) zt[j] = 0.0;
    At_mul(S, -1.0, S->y, zt);                 /* -A'y */
    for (int j = 0; j < S->nf; j++) if (!(fabs(zt[j]) <= tol * byv)) return 0;
    if (!(cone_viol(S, zt) <= tol * nrmi(n, zt))) return 0;
    memcpy(S->z, zt, sizeof(double) * (size_t)n);
    return 1;
}

/* (25 October 2026) a point in the rotated cones' mapped coordinates ((u + v), (u - v)) / sqrt 2 of the problem with the
 * fixed sides' automorphism (u' = gamma u, v' = v / gamma, the fixed side's row times rho) taken back to those of the
 * problem as given: u = u' / gamma, v = gamma v', z inversely, y = rho y' on the fixed side's row (x, z: n entries; y: m) */
static void rot_ungamma(int nq0, int nq, const int *qs, const double *g, const double *rho, const int *row, double *x, double *y, double *z) {
    const double rs = sqrt(0.5);
    for (int k = nq0; k < nq; k++) {
        if (g[k] == 1.0) continue;
        const int j = qs[k];
        if (x) { const double u = rs * (x[j] + x[j + 1]) / g[k], v = rs * (x[j] - x[j + 1]) * g[k]; x[j] = rs * (u + v); x[j + 1] = rs * (u - v); }
        if (z) { const double u = rs * (z[j] + z[j + 1]) * g[k], v = rs * (z[j] - z[j + 1]) / g[k]; z[j] = rs * (u + v); z[j + 1] = rs * (u - v); }
        if (y && row[k] >= 0) y[row[k]] *= rho[k];
    }
}
/* (25 October 2026, a test: BRISK_CONEROTFIX=2) the fixed side's row holds no other column, so its multiplier is the one
 * that solves the fixed column's dual equation c_f - a y_r - z_f = 0 (a presolve's postsolve of a singleton row): with the
 * row normalized to the entry 1 the mapped-back residual of that equation is the scaled one times |a| */
static void rot_rowdual(int nq0, int nq, const int *qs, const int *sd, const int *row, const double *ae, const double *c, double *y, const double *z) {
    const double rs = sqrt(0.5);
    for (int k = nq0; k < nq; k++) {
        if (row[k] < 0 || !sd[k]) continue;
        const int j = qs[k]; const double zf = sd[k] == 1 ? rs * (z[j] - z[j + 1]) : rs * (z[j] + z[j + 1]);
        y[row[k]] = (c[j + (sd[k] == 1)] - zf) / ae[k];
    }
}
static int socp_solve_core(const SocpProb *P, const SocpOpts *opt, SocpRes *R) {
    const double t0 = sc_time();
    SocpOpts od; if (!opt) { socp_default_opts(&od); opt = &od; }
    Socp S_; Socp *S = &S_;
    memset(S, 0, sizeof(*S));
    memset(R, 0, sizeof(*R));
    const int m = P->m, n = P->n;
    int nn = P->nf + P->nl;
    for (int k = 0; k < P->nq; k++) { if (P->q[k] < 1) { R->status = SOCP_INPUT; return SOCP_INPUT; } nn += P->q[k]; }
    for (int k = 0; k < P->nr; k++) { if (P->r[k] < 2) { R->status = SOCP_INPUT; return SOCP_INPUT; } nn += P->r[k]; }
    if (nn != n || m < 0 || n < 1) { R->status = SOCP_INPUT; return SOCP_INPUT; }
    S->m = m; S->n = n; S->nf = P->nf; S->nl = P->nl; S->nq = P->nq + P->nr;
    S->verbose = opt->verbose; S->refine_max = 10;
    S->nr_rot = P->nr;
    S->q = (int *)xcalloc((size_t)S->nq, sizeof(int)); S->qs = (int *)xcalloc((size_t)S->nq + 1, sizeof(int));
    { int s = S->nf + S->nl; for (int k = 0; k < S->nq; k++) { S->q[k] = k < P->nq ? P->q[k] : P->r[k - P->nq]; S->qs[k] = s; s += S->q[k]; } S->qs[S->nq] = s; }
    /* cones of dimension 1 are half-lines: kept as cones (x0 >= 0), the algebra is the same */
    /* (25 October 2026, the SOCP developer) a rotated cone 2 u v >= |w|^2 one of whose two sides is fixed by an equality row
     * of its own (v = beta / a: a single entry in its column and in its row) is moved by its automorphism u -> gamma u,
     * v -> v / gamma with gamma the fixed value (u fixed: 1 / its value), so that the fixed side is 1 and the cone's point is
     * of the size of the rest of it; the solution is mapped back at the end. The quadratic-over-linear terms x_i^2 / (2 y_i)
     * with y_i fixed at 10^U(-4, 4) (socp_rotscale_2000x1000, SOCP): 288 iterations and an objective 1e-5 off while every
     * error read 1e-8, the errors measured against right-hand sides up to 1e4 that hid the small cones; with the map 26
     * iterations and the objective right to 1e-9. BRISK_CONEROTFIX=0: off */
    double *rgam = NULL, *rrho = NULL, *rae = NULL, *A0x_ = NULL; int *rrow = NULL, *rsd = NULL;     /* (rsd: the fixed side, 1 = v, 2 = u) */
    static int rf = -1; if (rf < 0) { const char *e_ = getenv("BRISK_CONEROTFIX"); rf = e_ ? atoi(e_) : 1; }
    const int rotrow1 = rf == 2;
    {
      if (rf && P->nr > 0) {
        int *rcnt = (int *)xcalloc((size_t)m + 1, sizeof(int));
        for (int p = 0; p < P->Ap[n]; p++) if (P->Ax[p] != 0.0) rcnt[P->Ai[p]]++;
        for (int k = P->nq; k < S->nq; k++) {
            const int j1 = S->qs[k]; double g = 1.0, rho = 1.0, ae_ = 0.0; int ir = -1, sd = 0;
            for (int side = 0; side < 2 && g == 1.0; side++) {
                const int j = j1 + 1 - side;          /* (v first, then u) */
                int nz = 0, pr = -1; for (int p = P->Ap[j]; p < P->Ap[j + 1]; p++) if (P->Ax[p] != 0.0) { nz++; pr = p; }
                if (nz != 1 || rcnt[P->Ai[pr]] != 1) continue;
                const double val = P->b[P->Ai[pr]] / P->Ax[pr];
                if (val > 0 && isfinite(val) && val != 1.0) { g = side == 0 ? val : 1.0 / val; rho = rf == 2 ? 1.0 / P->b[P->Ai[pr]] : 1.0 / val; ir = P->Ai[pr]; sd = side + 1; ae_ = P->Ax[pr]; }
            }
            if (g != 1.0) {
                if (!rgam) { rgam = (double *)xcalloc((size_t)S->nq + 1, sizeof(double)); rrho = (double *)xcalloc((size_t)S->nq + 1, sizeof(double)); rrow = (int *)xcalloc((size_t)S->nq + 1, sizeof(int)); rsd = (int *)xcalloc((size_t)S->nq + 1, sizeof(int)); rae = (double *)xcalloc((size_t)S->nq + 1, sizeof(double)); }
                rgam[k] = g; rrho[k] = rho; rrow[k] = ir; rsd[k] = sd; rae[k] = ae_;
            }
        }
        free(rcnt);
        /* the fixed side's own row a v = beta divided by the fixed value as well (the side's column then keeps its entry a and
         * the row reads a v' = a): with the entry a gamma left in it, Ruiz's passes started from entries up to 1e4 and ended
         * at row factors 1e-2 .. 2e4 against 0.5 .. 3 for the same problem written with the side at 1 (32 -> 26 iterations
         * there, its first attempt OPTIMAL) */
        if (rgam) for (int k = P->nq; k < S->nq; k++) if (rgam[k] == 0.0) { rgam[k] = 1.0; rrho[k] = 1.0; rrow[k] = -1; }
        if (rgam && dbg_on()) { int c_ = 0; for (int k = P->nq; k < S->nq; k++) c_ += rgam[k] != 1.0; socp_printf("  [%d rotated cones with a fixed side moved to 1 by their automorphism]\n", c_); }
      } }
    /* ---- copy of A by columns; rotated cones: columns (c1, c2) -> ((c1 + c2), (c1 - c2)) / sqrt 2 */
    const double rs = sqrt(0.5);
    {
        long nz = P->Ap[n];
        for (int k = P->nq; k < S->nq; k++) { const int j = S->qs[k]; nz += (P->Ap[j + 2] - P->Ap[j]); }
        S->Ap = (int *)malloc(sizeof(int) * ((size_t)n + 1)); S->Ai = (int *)malloc(sizeof(int) * (size_t)(nz + 1)); S->Ax = (double *)malloc(sizeof(double) * (size_t)(nz + 1));
        double *acc = (double *)xcalloc((size_t)m, sizeof(double)); int *mk = (int *)malloc(sizeof(int) * ((size_t)m + 1)); int *lst = (int *)malloc(sizeof(int) * ((size_t)m + 1));
        double *acc0 = rgam ? (double *)xcalloc((size_t)m, sizeof(double)) : NULL;      /* (the problem as given's entries, for the error measures) */
        if (rgam) A0x_ = (double *)malloc(sizeof(double) * (size_t)(nz + 1));
        for (int i = 0; i < m; i++) mk[i] = -1;
        long w = 0; int kr = P->nq;
        for (int j = 0; j < n; j++) {
            S->Ap[j] = (int)w;
            const int rot = kr < S->nq && (j == S->qs[kr] || j == S->qs[kr] + 1);
            if (!rot) { for (int p = P->Ap[j]; p < P->Ap[j + 1]; p++) if (P->Ax[p] != 0.0) { S->Ai[w] = P->Ai[p]; S->Ax[w] = P->Ax[p]; if (A0x_) A0x_[w] = P->Ax[p]; w++; } }
            else {
                const int j1 = S->qs[kr], j2 = j1 + 1; const double s2 = j == j1 ? 1.0 : -1.0;
                /* (u = u' / gamma, v = gamma v'; the fixed side's column unchanged: its factor and its row's cancel) */
                const int fv = rgam && rsd[kr] == 1, fu = rgam && rsd[kr] == 2;
                const double fx = rgam && (fu || fv) && rotrow1 ? (fu ? 1.0 / rgam[kr] : rgam[kr]) * rrho[kr] : 1.0;     /* (the fixed column times its value and its row's factor: 1, or 1 / a under BRISK_CONEROTFIX=2) */
                const double g1 = rgam && !fu ? 1.0 / rgam[kr] : fx, g2 = rgam && !fv ? rgam[kr] : fx;
                int nl_ = 0;
                for (int p = P->Ap[j1]; p < P->Ap[j1 + 1]; p++) { const int i = P->Ai[p]; if (mk[i] != j) { mk[i] = j; lst[nl_++] = i; acc[i] = 0; if (acc0) acc0[i] = 0; } acc[i] += rs * g1 * P->Ax[p]; if (acc0) acc0[i] += rs * P->Ax[p]; }
                for (int p = P->Ap[j2]; p < P->Ap[j2 + 1]; p++) { const int i = P->Ai[p]; if (mk[i] != j) { mk[i] = j; lst[nl_++] = i; acc[i] = 0; if (acc0) acc0[i] = 0; } acc[i] += s2 * rs * g2 * P->Ax[p]; if (acc0) acc0[i] += s2 * rs * P->Ax[p]; }
                for (int t = 0; t < nl_; t++) if (acc[lst[t]] != 0.0) { S->Ai[w] = lst[t]; S->Ax[w] = acc[lst[t]]; if (A0x_) A0x_[w] = acc0[lst[t]]; w++; }
                if (j == j2) kr++;
            }
        }
        S->Ap[n] = (int)w;
        free(acc); free(acc0); free(mk); free(lst);
    }
    S->b = (double *)xcalloc((size_t)m, sizeof(double)); S->c = (double *)xcalloc((size_t)n, sizeof(double));
    memcpy(S->b, P->b, sizeof(double) * (size_t)m); memcpy(S->c, P->c, sizeof(double) * (size_t)n);
    if (rgam) for (int k = P->nq; k < S->nq; k++) if (rrow[k] >= 0) S->b[rrow[k]] *= rrho[k];     /* (the fixed side's row: a v' = a) */
    for (int k = P->nq; k < S->nq; k++) { const int j = S->qs[k]; const double g = rgam ? rgam[k] : 1.0, c1 = S->c[j] / g, c2 = S->c[j + 1] * g; S->c[j] = rs * (c1 + c2); S->c[j + 1] = rs * (c1 - c2); }
    /* the problem as given (after the map): kept for the error measures */
    double *A0x = A0x_; if (!A0x) { A0x = (double *)malloc(sizeof(double) * (size_t)(S->Ap[n] + 1)); memcpy(A0x, S->Ax, sizeof(double) * (size_t)S->Ap[n]); }
    double *b0 = (double *)xcalloc((size_t)m, sizeof(double)), *c0 = (double *)xcalloc((size_t)n, sizeof(double));
    memcpy(b0, S->b, sizeof(double) * (size_t)m); memcpy(c0, S->c, sizeof(double) * (size_t)n);
    if (rgam) {      /* (the fixed sides' automorphism not in the problem the errors are measured on) */
        memcpy(b0, P->b, sizeof(double) * (size_t)m);
        for (int k = P->nq; k < S->nq; k++) { const int j = S->qs[k]; const double c1 = P->c[j], c2 = P->c[j + 1]; c0[j] = rs * (c1 + c2); c0[j + 1] = rs * (c1 - c2); }
    }
    const double nb0 = nrmi(m, b0), nc0 = nrmi(n, c0);
    /* ---- equilibration */
    S->Dr = (double *)malloc(sizeof(double) * ((size_t)m + 1)); S->Ec = (double *)malloc(sizeof(double) * (size_t)n);
    for (int i = 0; i < m; i++) S->Dr[i] = 1.0;
    for (int j = 0; j < n; j++) S->Ec[j] = 1.0;
    /* (27 October 2026, the SOCP developer, a test: BRISK_CONEEQBC=1) Ruiz's passes on the extended matrix [A b; c' 0]: b a
     * column and c a row of it, with their own factors eb and ec that become the norm scaling of b and c. Scaling a linear
     * column is the same as choosing that variable's own start balance (x_j 1/d_j times larger, z_j d_j times smaller), so the
     * equilibration of A alone picks the start's balance without regard to the costs and the right-hand side; with them, a
     * column of large cost is scaled down and a row of large right-hand side up (socp_svm_25k, SOCP: 28 iterations, 16 with b
     * divided by 10 more; IDEAS, the start's balance) */
    static int eqbc = -1; if (eqbc < 0) { const char *e_ = getenv("BRISK_CONEEQBC"); eqbc = e_ ? atoi(e_) : 0; }
    double eqb = 1.0, eqc = 1.0;
    if (opt->equil) {
        double *rn = (double *)malloc(sizeof(double) * ((size_t)m + 1)), *cn = (double *)malloc(sizeof(double) * (size_t)n);
        if (opt->equil == 2) {     /* (test: geometric-mean scaling passes first: each row and column scaled by 1/sqrt(max min)) */
            double *rl = (double *)malloc(sizeof(double) * ((size_t)m + 1)), *cl = (double *)malloc(sizeof(double) * (size_t)n);
            for (int pass = 0; pass < 8; pass++) {
                for (int i = 0; i < m; i++) { rn[i] = 0.0; rl[i] = 1e300; }
                for (int j = 0; j < n; j++) { double s = 0, l = 1e300; for (int p = S->Ap[j]; p < S->Ap[j + 1]; p++) { const double v = fabs(S->Ax[p]); if (v == 0) continue; if (v > s) s = v; if (v < l) l = v; if (v > rn[S->Ai[p]]) rn[S->Ai[p]] = v; if (v < rl[S->Ai[p]]) rl[S->Ai[p]] = v; } cn[j] = s; cl[j] = l; }
                for (int k = 0; k < S->nq; k++) { double s = 0, l = 1e300; for (int i = S->qs[k]; i < S->qs[k + 1]; i++) { if (cn[i] > s) s = cn[i]; if (cl[i] < l) l = cl[i]; } for (int i = S->qs[k]; i < S->qs[k + 1]; i++) { cn[i] = s; cl[i] = l; } }
                for (int i = 0; i < m; i++) rn[i] = rn[i] > 0 ? 1.0 / sqrt(sqrt(rn[i] * rl[i])) : 1.0;
                for (int j = 0; j < n; j++) cn[j] = cn[j] > 0 ? 1.0 / sqrt(sqrt(cn[j] * cl[j])) : 1.0;
                for (int j = 0; j < n; j++) for (int p = S->Ap[j]; p < S->Ap[j + 1]; p++) S->Ax[p] *= rn[S->Ai[p]] * cn[j];
                for (int i = 0; i < m; i++) S->Dr[i] *= rn[i];
                for (int j = 0; j < n; j++) S->Ec[j] *= cn[j];
            }
            free(rl); free(cl);
        }
        /* (24 October 2026, the SOCP developer; the default, BRISK_CONEEQSLACK=0: off) the slack blocks apart: a block (a linear column,
         * or a cone) all of whose columns have a single entry, each in a row of its own that holds no other such column, is
         * the slack of those rows (a cone constraint F x + g in K, or an inequality row). Its identity entries are not the
         * data: they hold every row's largest entry at 1, so the rows' data are never scaled up (rows scaled by 1e-4 in the
         * file stay so: 150 iterations and an infeasible point accepted). The rows of a slack block take one factor from
         * their data (the entries outside the slack columns), the block's columns its inverse (the slack entries stay as
         * they are, the cone is scaled as a whole). The 253 SOCP files and QCQPs: no status changed, the two badly scaled files
         * 150 -> 14 and 147 -> 13 iterations with the right objectives, the rest 3,850 -> 3,852 iterations, time 0.99
         * (bench/socp_eqslack/; ISSUES 279) */
        static int eqsl = -1; if (eqsl < 0) { const char *e_ = getenv("BRISK_CONEEQSLACK"); eqsl = e_ ? atoi(e_) : 1; }
        int *slrow = NULL, *slblk = NULL, nslb = 0;     /* the slack block of a row (-1: none); the block's first column of each slack block */
        int *colblk = NULL;                              /* the slack block of a column (-1: none) */
        if (eqsl) {
            slrow = (int *)malloc(sizeof(int) * ((size_t)m + 1)); colblk = (int *)malloc(sizeof(int) * ((size_t)n + 1));
            int *rcnt = (int *)calloc((size_t)m + 1, sizeof(int));
            for (int j = S->nf; j < n; j++) if (S->Ap[j + 1] - S->Ap[j] == 1) rcnt[S->Ai[S->Ap[j]]]++;
            for (int i = 0; i < m; i++) slrow[i] = -1;
            for (int j = 0; j < n; j++) colblk[j] = -1;
            slblk = (int *)malloc(sizeof(int) * ((size_t)S->nl + S->nq + 1));
            const int nbk = S->nl + S->nq;
            for (int b_ = 0; b_ < nbk; b_++) {
                const int j0 = b_ < S->nl ? S->nf + b_ : S->qs[b_ - S->nl], j1 = b_ < S->nl ? j0 + 1 : S->qs[b_ - S->nl + 1];
                int ok = 1;
                for (int j = j0; j < j1 && ok; j++) ok = S->Ap[j + 1] - S->Ap[j] == 1 && rcnt[S->Ai[S->Ap[j]]] == 1 && S->Ax[S->Ap[j]] != 0.0;
                if (!ok) continue;
                for (int j = j0; j < j1; j++) { slrow[S->Ai[S->Ap[j]]] = nslb; colblk[j] = nslb; }
                slblk[nslb++] = b_;
            }
            free(rcnt);
            if (dbg_on()) socp_printf("  [equilibration: %d slack blocks apart]\n", nslb);
            /* (27 October 2026, the SOCP developer; BRISK_CONEEQSLNORM=0: off) each row of a slack block first divided by the size
             * of its slack entry, so that the slack entries are +-1 before the passes below, which keep them as they find them.
             * Without it a row multiplied by 10^u in the file (the slack's entry with it) kept its slack entry 10^u: the
             * variable's scale, and so the start's balance, came from the way the row was written. The independent benchmark's
             * scaled copies (each row of the primal, or each variable, which the dual form sees as its rows, times 10^u, u in
             * [-3, 3]): the classification SOCP 13 -> 219 iterations, the dense-margin SOCP 10 -> 33 */
            static int eqsn = -1; if (eqsn < 0) eqsn = getenv("BRISK_CONEEQSLNORM") ? atoi(getenv("BRISK_CONEEQSLNORM")) : 1;
            if (eqsn && nslb > 0) {
                double *se = (double *)malloc(sizeof(double) * ((size_t)m + 1));
                for (int i = 0; i < m; i++) se[i] = 1.0;
                for (int j = 0; j < n; j++) if (colblk[j] >= 0) se[S->Ai[S->Ap[j]]] = 1.0 / fabs(S->Ax[S->Ap[j]]);
                for (int j = 0; j < n; j++) for (int p = S->Ap[j]; p < S->Ap[j + 1]; p++) S->Ax[p] *= se[S->Ai[p]];
                for (int i = 0; i < m; i++) S->Dr[i] *= se[i];
                free(se);
            }
        }
        /* (27 October 2026, a test: BRISK_CONEEQCR=1) the scaling of Curtis and Reid first, the passes below from its result: the
         * rows of a slack block one group (their slack columns left out and given the inverse factor, the slack entries +-1
         * stay), a cone's columns one group */
        static int eqcr = -1; if (eqcr < 0) eqcr = getenv("BRISK_CONEEQCR") ? atoi(getenv("BRISK_CONEEQCR")) : 0;
        if (eqcr) {
            int *rg = (int *)malloc(sizeof(int) * ((size_t)m + 1)), *cg = (int *)malloc(sizeof(int) * ((size_t)n + 1));
            int nrg = slrow ? nslb : 0;
            for (int i = 0; i < m; i++) rg[i] = slrow && slrow[i] >= 0 ? slrow[i] : nrg++;
            int ncg = 0;
            for (int j = 0; j < S->nf + S->nl; j++) cg[j] = colblk && colblk[j] >= 0 ? -1 : ncg++;
            for (int k = 0; k < S->nq; k++) { const int slk = colblk && colblk[S->qs[k]] >= 0; for (int j = S->qs[k]; j < S->qs[k + 1]; j++) cg[j] = slk ? -1 : ncg; if (!slk) ncg++; }
            static int crbc = -1; if (crbc < 0) crbc = getenv("BRISK_CONEEQCRBC") ? atoi(getenv("BRISK_CONEEQCRBC")) : 0;
            double *bsc_ = NULL, *csc_ = NULL;
            if (crbc) {     /* (b and c with the factors so far) */
                bsc_ = (double *)malloc(sizeof(double) * ((size_t)m + 1)); csc_ = (double *)malloc(sizeof(double) * ((size_t)n + 1));
                for (int i = 0; i < m; i++) bsc_[i] = S->b[i] * S->Dr[i];
                for (int j = 0; j < n; j++) csc_[j] = S->c[j] * S->Ec[j];
                nrg++; ncg++;
            }
            double *rho = (double *)calloc((size_t)nrg + 1, sizeof(double)), *gam = (double *)calloc((size_t)ncg + 1, sizeof(double));
            cr_scale(m, n, S->Ap, S->Ai, S->Ax, rg, nrg, cg, ncg, eqcr > 1 ? eqcr : 100, rho, gam, bsc_, csc_);
            free(bsc_); free(csc_);
            for (int j = 0; j < n; j++) {
                const double fc = cg[j] >= 0 ? exp(gam[cg[j]]) : (S->Ap[j + 1] > S->Ap[j] ? exp(-rho[rg[S->Ai[S->Ap[j]]]]) : 1.0);
                for (int p = S->Ap[j]; p < S->Ap[j + 1]; p++) S->Ax[p] *= exp(rho[rg[S->Ai[p]]]) * fc;
                S->Ec[j] *= fc;
            }
            for (int i = 0; i < m; i++) S->Dr[i] *= exp(rho[rg[i]]);
            free(rg); free(cg); free(rho); free(gam);
        }
        double *bmx = eqsl ? (double *)malloc(sizeof(double) * ((size_t)nslb + 1)) : NULL;
        static int eqpass = -1; if (eqpass < 0) eqpass = getenv("BRISK_CONEEQPASS") ? atoi(getenv("BRISK_CONEEQPASS")) : 10;
        int npass_ = 0; double dev_ = 0;
        for (int pass = 0; pass < eqpass; pass++) {
            npass_ = pass + 1;
            for (int i = 0; i < m; i++) rn[i] = 0.0;
            for (int j = 0; j < n; j++) { double s = 0; const int sl = colblk && colblk[j] >= 0; for (int p = S->Ap[j]; p < S->Ap[j + 1]; p++) { const double v = fabs(S->Ax[p]); if (v > s) s = v; if (!sl && v > rn[S->Ai[p]]) rn[S->Ai[p]] = v; } cn[j] = s; }
            double bm = 0, cm = 0;
            if (eqbc) {
                for (int i = 0; i < m; i++) { const double v = fabs(S->b[i]) * S->Dr[i] * eqb; if (v > rn[i]) rn[i] = v; if (v > bm) bm = v; }
                for (int j = S->nf; j < n; j++) { const double v = fabs(S->c[j]) * S->Ec[j] * eqc; if (v > cn[j]) cn[j] = v; if (v > cm) cm = v; }
            }
            if (eqsl) {
                /* a slack block's rows: one factor from the largest data entry over its rows; its columns the inverse */
                for (int t = 0; t < nslb; t++) bmx[t] = 0.0;
                for (int i = 0; i < m; i++) if (slrow[i] >= 0 && rn[i] > bmx[slrow[i]]) bmx[slrow[i]] = rn[i];
                for (int i = 0; i < m; i++) if (slrow[i] >= 0) rn[i] = bmx[slrow[i]] > 0 ? bmx[slrow[i]] : 1.0;
                for (int i = 0; i < m; i++) if (slrow[i] < 0 && rn[i] == 0.0) { /* (a row of slack entries alone: as before) */ }
            }
            for (int k = 0; k < S->nq; k++) { double s = 0; for (int i = S->qs[k]; i < S->qs[k + 1]; i++) if (cn[i] > s) s = cn[i]; for (int i = S->qs[k]; i < S->qs[k + 1]; i++) cn[i] = s; }
            double dev = 0;
            for (int i = 0; i < m; i++) { if (rn[i] > 0) { dev = fmax(dev, fabs(1.0 - rn[i])); rn[i] = 1.0 / sqrt(rn[i]); } else rn[i] = 1.0; }
            for (int j = 0; j < n; j++) {
                if (colblk && colblk[j] >= 0) { cn[j] = 1.0 / rn[S->Ai[S->Ap[j]]]; continue; }     /* (a slack column: the inverse of its row's factor) */
                if (cn[j] > 0) { dev = fmax(dev, fabs(1.0 - cn[j])); cn[j] = 1.0 / sqrt(cn[j]); } else cn[j] = 1.0; }
            if (eqbc) {
                if (bm > 0) { dev = fmax(dev, fabs(1.0 - bm)); eqb /= sqrt(bm); }
                if (cm > 0) { dev = fmax(dev, fabs(1.0 - cm)); eqc /= sqrt(cm); }
            }
            for (int j = 0; j < n; j++) for (int p = S->Ap[j]; p < S->Ap[j + 1]; p++) S->Ax[p] *= rn[S->Ai[p]] * cn[j];
            for (int i = 0; i < m; i++) S->Dr[i] *= rn[i];
            for (int j = 0; j < n; j++) S->Ec[j] *= cn[j];
            dev_ = dev;
            if (dev < 1e-2) break;
        }
        if (dbg_on()) socp_printf("  [equilibration: %d passes, the last change of a factor %.2e]\n", npass_, dev_);
        free(slrow); free(slblk); free(colblk); free(bmx);
        free(rn); free(cn);
    }
    for (int i = 0; i < m; i++) S->b[i] *= S->Dr[i];
    for (int j = 0; j < n; j++) S->c[j] *= S->Ec[j];
    if (S->Ap[n] >= 100000 && g_cone_nt > 1) {     /* the row copy of the scaled A, for A x on the threads */
        const int nz = S->Ap[n];
        S->Rp = (int *)xcalloc((size_t)m + 2, sizeof(int)); S->Ri = (int *)malloc(sizeof(int) * ((size_t)nz + 1)); S->Rx = (double *)malloc(sizeof(double) * ((size_t)nz + 1));
        for (int p = 0; p < nz; p++) S->Rp[S->Ai[p] + 2]++;
        for (int i = 0; i < m; i++) S->Rp[i + 2] += S->Rp[i + 1];
        S->Rq = (int *)malloc(sizeof(int) * ((size_t)nz + 1));
        for (int j = 0; j < n; j++) for (int p = S->Ap[j]; p < S->Ap[j + 1]; p++) { const int q = S->Rp[S->Ai[p] + 1]++; S->Ri[q] = j; S->Rx[q] = S->Ax[p]; S->Rq[p] = q; }
        S->R0x = (double *)malloc(sizeof(double) * ((size_t)nz + 1));
        for (int p = 0; p < nz; p++) S->R0x[S->Rq[p]] = A0x[p];
    }
    dn_build(S);
    if (dbg_on()) {     /* (the scaled data's spread: entries of A, b and c, and the factors) */
        double amn = 1e300, amx = 0, bmn = 1e300, bmx = 0, cmn = 1e300, cmx = 0, dmn = 1e300, dmx = 0, emn = 1e300, emx = 0;
        for (int p = 0; p < S->Ap[n]; p++) { const double v = fabs(S->Ax[p]); if (v > 0) { amn = fmin(amn, v); amx = fmax(amx, v); } }
        for (int i = 0; i < m; i++) { const double v = fabs(S->b[i]); if (v > 0) { bmn = fmin(bmn, v); bmx = fmax(bmx, v); } dmn = fmin(dmn, S->Dr[i]); dmx = fmax(dmx, S->Dr[i]); }
        for (int j = 0; j < n; j++) { const double v = fabs(S->c[j]); if (v > 0) { cmn = fmin(cmn, v); cmx = fmax(cmx, v); } emn = fmin(emn, S->Ec[j]); emx = fmax(emx, S->Ec[j]); }
        socp_printf("  [equilibrated: |A| %.1e .. %.1e, |b| %.1e .. %.1e, |c| %.1e .. %.1e; row factors %.1e .. %.1e, column factors %.1e .. %.1e]\n", amn, amx, bmn, bmx, cmn, cmx, dmn, dmx, emn, emx);
    }
    S->sb = fmax(1.0, nrmi(m, S->b)); S->scl = fmax(1.0, nrmi(n, S->c));
    if (dbg_on()) {      /* (27 October 2026, debug: the sizes of b and c after the equilibration: largest, root mean square and geometric mean of the nonzeros) */
        double bl = 0, br = 0, cl = 0, cr = 0; int bn = 0, cn_ = 0;
        for (int i = 0; i < m; i++) if (S->b[i] != 0.0) { bl += log(fabs(S->b[i])); br += S->b[i] * S->b[i]; bn++; }
        for (int j = 0; j < n; j++) if (S->c[j] != 0.0) { cl += log(fabs(S->c[j])); cr += S->c[j] * S->c[j]; cn_++; }
        socp_printf("  [the sizes of b and c: largest %.3e %.3e, root mean square %.3e %.3e, geometric mean %.3e %.3e (of %d and %d nonzeros)]\n",
                    nrmi(m, S->b), nrmi(n, S->c), bn ? sqrt(br / bn) : 0.0, cn_ ? sqrt(cr / cn_) : 0.0, bn ? exp(bl / bn) : 0.0, cn_ ? exp(cl / cn_) : 0.0, bn, cn_);
    }
    { static int nobc = -1; if (nobc < 0) nobc = getenv("BRISK_CONENOBCNORM") != NULL;   /* (27 October 2026, a test: b and c as the equilibration leaves them, not divided by their largest entries) */
      if (nobc) { S->sb = 1.0; S->scl = 1.0; } }
    if (eqbc && opt->equil) { S->sb = 1.0 / eqb; S->scl = 1.0 / eqc; if (dbg_on()) socp_printf("  [the extended equilibration: b divided by %.3e, c by %.3e]\n", S->sb, S->scl); }
    if (!opt->equil) { S->sb = 1.0; S->scl = 1.0; }
    if (getenv("BRISK_CONESBF")) S->sb *= atof(getenv("BRISK_CONESBF"));      /* (test) */
    if (getenv("BRISK_CONESCF")) S->scl *= atof(getenv("BRISK_CONESCF"));
    for (int i = 0; i < m; i++) S->b[i] /= S->sb;
    for (int j = 0; j < n; j++) S->c[j] /= S->scl;
    /* ---- workspace, pattern, ordering */
    S->x = (double *)xcalloc((size_t)n, sizeof(double)); S->y = (double *)xcalloc((size_t)m, sizeof(double)); S->z = (double *)xcalloc((size_t)n, sizeof(double));
    S->h = (double *)xcalloc((size_t)S->nl, sizeof(double)); S->eta = (double *)xcalloc((size_t)S->nq, sizeof(double));
    S->wbar = (double *)xcalloc((size_t)n, sizeof(double)); S->lam = (double *)xcalloc((size_t)n, sizeof(double));
    {   /* (20 October 2026, LP/SOCP) a second-order cone whose first entry has no rows and no cost is the free variables (t, x_f)
         * of core_freecone (or a user's): its start is shifted on its own, as a rotated cone's */
        for (int k = 0; k < S->nq; k++) {
            const int j = S->qs[k];
            if (S->q[k] > 1 && S->Ap[j + 1] == S->Ap[j] && S->c[j] == 0.0) { if (!S->fcone) S->fcone = (unsigned char *)xcalloc((size_t)S->nq + 1, 1); S->fcone[k] = 1; }
        }
    }
    {   /* (18 October 2026, LP/SOCP: the rotated cones that can be balanced by a boost, their two columns on the same rows) */
        static int bal = -1; if (bal < 0) { const char *e_ = getenv("BRISK_CONEBAL"); bal = e_ ? atoi(e_) : 1; }   /* (0: off) */
        if (bal && S->nr_rot > 0) {
            S->bpos = (int **)xcalloc((size_t)S->nq + 1, sizeof(int *)); S->bth = (double *)xcalloc((size_t)S->nq + 1, sizeof(double));
            for (int k = S->nq - S->nr_rot; k < S->nq; k++) {
                const int j = S->qs[k], l0 = S->Ap[j + 1] - S->Ap[j], l1 = S->Ap[j + 2] - S->Ap[j + 1];
                int same = l0 == l1;
                for (int t = 0; same && t < l0; t++) if (S->Ai[S->Ap[j] + t] != S->Ai[S->Ap[j + 1] + t]) same = 0;
                if (!same) S->bth[k] = NAN;      /* (not boosted) */
            }
        }
    }
    { const double dw = kkt_dense_work(S); if (dw > 0) { S->kkt_defer = 1; S->cflops = dw; } }
    { const char *e_ = getenv("BRISK_CONEROT"); S->rot_mode = e_ ? (atoi(e_) == 1 ? 1 : atoi(e_) == 2 ? 2 : 0) : 2; }
    for (int k = 0; k < S->nq; k++) if (S->q[k] >= 2 && S->q[k] <= SOCP_DENSE_MAX) S->nsmallc++;
    { const double ts = sc_time(); const int rc_ = kkt_setup(S); S->t_setup = sc_time() - ts;
      if (rc_) { socp_free(S); free(A0x); free(b0); free(c0); free(rgam); free(rrho); free(rrow); free(rsd); free(rae); R->status = SOCP_INPUT; return SOCP_INPUT; } }
    const size_t wn = (size_t)S->nk + 8;
    S->w1 = (double *)xcalloc(wn, sizeof(double)); S->w2 = (double *)xcalloc(wn, sizeof(double)); S->w3 = (double *)xcalloc(wn, sizeof(double));
    S->w4 = (double *)xcalloc(wn, sizeof(double)); S->rk = (double *)xcalloc(wn, sizeof(double)); S->sk = (double *)xcalloc(wn, sizeof(double));
    S->lastx = (double *)xcalloc(2 * wn, sizeof(double)); S->lasty = S->lastx + wn;
    S->swork2 = (double *)xcalloc((size_t)m + 8, sizeof(double));
    { const double ts = sc_time(); ne_setup(S); S->t_setup += sc_time() - ts; if (dbg_on()) socp_printf("  [kkt setup %.3fs, normal equations setup %.3fs]\n", S->t_setup - (sc_time() - ts), sc_time() - ts); }
    if (S->kkt_defer && !S->ne) {           /* (the normal equations not taken after all: the full system now) */
        const double ts = sc_time(); const int rc_ = kkt_analyse(S); S->t_setup += sc_time() - ts;
        if (rc_) { socp_free(S); free(A0x); free(b0); free(c0); free(rgam); free(rrho); free(rrow); free(rsd); free(rae); R->status = SOCP_INPUT; return SOCP_INPUT; }
    }
    S->gm_max = 10; S->gm_tol = 1e-12; S->gm_abs = 0.0; S->tg_d = S->tg_p = S->tg_g = 0.0; S->tg_f = 1.0; S->tg_fl = S->tg_fd = S->tg_fp = 0.0; S->tg_skip = 0; S->tg_w = 0;
    S->gs = (double *)xcalloc(wn, sizeof(double));
    S->gV = (double *)xcalloc((size_t)(S->gm_max + 1) * wn, sizeof(double)); S->gZ = (double *)xcalloc((size_t)S->gm_max * wn, sizeof(double));
    S->gH = (double *)xcalloc((size_t)(S->gm_max + 1) * (S->gm_max + 1), sizeof(double)); S->gcs = (double *)xcalloc(4 * ((size_t)S->gm_max + 1), sizeof(double));
    double *px = (double *)xcalloc((size_t)n, sizeof(double)), *py = (double *)xcalloc((size_t)m + 1, sizeof(double));
    double *qx = (double *)xcalloc((size_t)n, sizeof(double)), *qy = (double *)xcalloc((size_t)m + 1, sizeof(double));
    double *dx = (double *)xcalloc((size_t)n, sizeof(double)), *dy = (double *)xcalloc((size_t)m + 1, sizeof(double)), *dz = (double *)xcalloc((size_t)n, sizeof(double));
    double *rP = (double *)xcalloc((size_t)m + 1, sizeof(double)), *rD = (double *)xcalloc((size_t)n, sizeof(double));
    double *r3 = (double *)xcalloc((size_t)n, sizeof(double)), *rhsx = (double *)xcalloc((size_t)n, sizeof(double)), *rhsy = (double *)xcalloc((size_t)m + 1, sizeof(double));
    double *ds = (double *)xcalloc((size_t)n, sizeof(double)), *t1 = (double *)xcalloc((size_t)n, sizeof(double)), *t2 = (double *)xcalloc((size_t)n, sizeof(double));
    double *xo = (double *)xcalloc((size_t)n, sizeof(double)), *yo = (double *)xcalloc((size_t)m + 1, sizeof(double)), *zo = (double *)xcalloc((size_t)n, sizeof(double));
    double *bx = (double *)xcalloc((size_t)n, sizeof(double)), *by = (double *)xcalloc((size_t)m + 1, sizeof(double)), *bz = (double *)xcalloc((size_t)n, sizeof(double));
    double best = 1e300, best_e[4] = { 0, 0, 0, 0 };
    S->sreg = 1e-9; S->sregx = 1e-9; S->sregf = 1e-6;          /* regularization of the factored matrix: dual, cone and free blocks */
    { const char *e_ = getenv("BRISK_CONEREG"); if (e_) { double a_ = 0, b_ = 0, c_ = 0; const int k_ = sscanf(e_, "%lf,%lf,%lf", &a_, &b_, &c_); if (k_ >= 1) S->sreg = a_; if (k_ >= 2) S->sregx = b_; if (k_ >= 3) S->sregf = c_; } }   /* (test: dual, cone, free) */
    S->F.dyn_delta = 1e-7; S->ref_pred = 0;
    S->sreg0 = S->sreg; S->bump = 1.0; S->bump_on = getenv("BRISK_CONENOBUMP") == NULL;
    S->ne_dl = getenv("BRISK_CONENEDL0") ? 0.0 : 1e-2 * (opt->tol > 0 ? opt->tol : 1e-8);       /* (BRISK_CONENEDL0: the base 1e-9, as before) */
    S->dd_on = getenv("BRISK_CONENODD") == NULL;
    S->dz_refine = getenv("BRISK_CONENODZ") == NULL;
    S->wdz = (double *)xcalloc((size_t)n + 1, sizeof(double)); S->wdx = (double *)xcalloc((size_t)n + 1, sizeof(double));
    S->wdy = (double *)xcalloc((size_t)m + 1, sizeof(double)); S->wdy2 = (double *)xcalloc((size_t)m + 1, sizeof(double));
    const double ncs = nrm2(n, S->c);
    /* ---- start: the identity of the cone */
    for (int i = S->nf; i < S->nf + S->nl; i++) { S->x[i] = 1.0; S->z[i] = 1.0; }
    for (int k = 0; k < S->nq; k++) { S->x[S->qs[k]] = 1.0; S->z[S->qs[k]] = 1.0; }
    S->tau = 1.0; S->kap = 1.0;
    if (!opt->wx && !getenv("BRISK_CONENOLS") && nt_update(S) == 0) {
        /* the start: x the solution of A x = b of least norm, (y, z) the least-squares dual point
         * (min |z|, z = c - A'y), each moved into the cone along the identity when it is outside.
         * Both come from the first factorization (identity scaling). */
        kkt_fill(S);
        kkt_factor_reg(S);
        for (int j = 0; j < n; j++) dx[j] = 0.0;
        for (int i = 0; i < m; i++) dy[i] = 0.0;
        kkt_solve(S, dx, S->b, px, py, S->refine_max);
        kkt_solve(S, S->c, dy, qx, qy, S->refine_max);
        /* (22 October 2026, LP/SOCP, a test: BRISK_CONEEPI=1) a rotated cone 2 s t >= |r|^2 whose t is fixed by a row of
         * its own (t = beta: the epigraph of a quadratic cost; in the mapped coordinates the row x_j - x_{j+1} = delta)
         * starts at a point of that row: the tail r from the least-squares point, x_{j+1} such that the cone's
         * determinant x_j^2 - x_{j+1}^2 - |r|^2 is 1, x_j = x_{j+1} + delta, and its dual mu x^{-1} (on the central path);
         * it takes no part in the shared shift. The shift into the cone moved t off its row (a primal residual of 5e3 on
         * nonnegative least squares, the first five steps below 0.03) */
        double *epid = NULL;
        { static int epi = -1; if (epi < 0) { const char *e_ = getenv("BRISK_CONEEPI"); epi = e_ ? atoi(e_) : 1; }     /* (the default since 22 October 2026; BRISK_CONEEPI=0: off) */
          if (epi && S->nr_rot > 0) {
            int *rc_ = (int *)xcalloc((size_t)m + 1, sizeof(int));
            for (int p = 0; p < S->Ap[n]; p++) rc_[S->Ai[p]]++;
            for (int k = S->nq - S->nr_rot; k < S->nq; k++) {
                const int j = S->qs[k];
                /* (only a large cone, of more than 4 entries, the size from which the solver forms no dense block: the epigraph
                 * of a quadratic over many variables. The many small rotated cones of a structural model (chainsing-50000:
                 * 49,998 of dimension 3; db-plate-yield-line) are balanced against each other by the shared shift: started
                 * one by one, chainsing went 13 -> 16 iterations and db-plate past 300 s) */
                if (S->q[k] <= 4) continue;
                for (int p = S->Ap[j]; p < S->Ap[j + 1]; p++) {
                    const int i = S->Ai[p]; const double al = S->Ax[p];
                    if (rc_[i] != 2 || al == 0.0) continue;
                    double be = 0; for (int q = S->Ap[j + 1]; q < S->Ap[j + 2]; q++) if (S->Ai[q] == i) be = S->Ax[q];
                    if (!(fabs(be + al) <= 1e-12 * fabs(al))) continue;
                    const double dl = S->b[i] / al;
                    if (dl > 0) { if (!epid) { epid = (double *)xcalloc((size_t)S->nq, sizeof(double)); } epid[k] = dl; }
                    break;
                }
            }
            free(rc_);
          } }
        double sx = 1e300, sz = 1e300;
        for (int i = S->nf; i < S->nf + S->nl; i++) { sx = fmin(sx, px[i]); sz = fmin(sz, -qx[i]); }
        for (int k = 0; k < S->nq; k++) {
            const int j = S->qs[k], d = S->q[k];
            if (S->fcone && S->fcone[k]) continue;         /* (a free cone: its own shift below) */
            static int epish = -1; if (epish < 0) { const char *e_ = getenv("BRISK_CONEEPISH"); epish = e_ ? atoi(e_) : 0; }   /* (a test: 1 = the epigraph cone kept in the shared shift's margin, its own start below all the same) */
            if (epid && epid[k] > 0 && !epish) continue;    /* (an epigraph cone: its own start below) */
            double a = 0, b_ = 0;
            for (int i = 1; i < d; i++) { a += px[j + i] * px[j + i]; b_ += qx[j + i] * qx[j + i]; }
            sx = fmin(sx, px[j] - sqrt(a)); sz = fmin(sz, -qx[j] - sqrt(b_));
        }
        {   /* (21 October 2026, LP/SOCP, a test: BRISK_CONEDUALC=1) the dual start y = 0, z = c instead of the least-squares
             * dual when it is nearer the cone (needs the smaller shift): the SVM's least-squares dual has an entry of -1.05
             * and is shifted by 2.05, its costs need 1 */
            static int dc = -1; if (dc < 0) { const char *e_ = getenv("BRISK_CONEDUALC"); dc = e_ ? atoi(e_) : 0; }
            if (dc) {
                double szc = 1e300;
                for (int i = S->nf; i < S->nf + S->nl; i++) szc = fmin(szc, S->c[i]);
                for (int k = 0; k < S->nq; k++) { const int j = S->qs[k], d = S->q[k]; if (S->fcone && S->fcone[k]) continue; double b_ = 0; for (int i = 1; i < d; i++) b_ += S->c[j + i] * S->c[j + i]; szc = fmin(szc, S->c[j] - sqrt(b_)); }
                if (dbg_on()) socp_printf("  [start: the least-squares dual needs a shift of %.3e, the costs %.3e]\n", 1.0 - sz, 1.0 - szc);
                if (szc > sz || dc == 2) { for (int j = 0; j < n; j++) qx[j] = -S->c[j]; for (int i = 0; i < m; i++) qy[i] = 0.0; sz = szc; }
            }
        }
        if (dbg_on()) for (int k = S->nq - 1; k >= 0 && k >= S->nq - 3; k--) {     /* (the rotated cones' least-squares entries: a, b, |r|) */
            const int j = S->qs[k], d = S->q[k]; if (k < S->nq - (S->nr_rot)) break;
            const double a = sqrt(0.5) * (px[j] + px[j + 1]) * S->Ec[j], b_ = sqrt(0.5) * (px[j] - px[j + 1]) * S->Ec[j]; double r2 = 0; for (int i = 2; i < d; i++) r2 += px[j + i] * px[j + i] * S->Ec[j + i] * S->Ec[j + i];
            socp_printf("  [rotated cone %d (dimension %d), least squares: a %.3e b %.3e |r| %.3e; |r| / (sqrt2 max(a, b)) %.3e]\n", k, d, a, b_, sqrt(r2), sqrt(r2) / (sqrt(2.0) * fmax(fabs(a), fabs(b_))));
        }
        /* (20 October 2026, LP/SOCP, a test: BRISK_CONESTARTD=s, BRISK_CONESTARTP=s) the least-squares dual (primal) point divided
         * by s before the shift (s = 0: by the size of its most negative entry, so that the shift to the margin stays of order
         * one); the SVM's dual point has an entry of -270 and the shift of 271 puts every z there: 24 iterations) */
        { static double sd = -2, sp = -2; if (sd < -1) { const char *e_ = getenv("BRISK_CONESTARTD"); sd = e_ ? atof(e_) : -1; e_ = getenv("BRISK_CONESTARTP"); sp = e_ ? atof(e_) : -1; }
          if (sd >= 0) { const double f = sd > 0 ? sd : fmax(1.0, -sz); for (int j = 0; j < n; j++) qx[j] /= f; for (int i = 0; i < m; i++) qy[i] /= f; sz /= f; if (dbg_on()) socp_printf("  [start: the dual point divided by %.3e]\n", f); }
          if (sp >= 0) { const double f = sp > 0 ? sp : fmax(1.0, -sx); for (int j = 0; j < n; j++) px[j] /= f; sx /= f; if (dbg_on()) socp_printf("  [start: the primal point divided by %.3e]\n", f); } }
        int ok = 1;
        for (int j = 0; j < n; j++) if (!(fabs(px[j]) < 1e8) || !(fabs(qx[j]) < 1e8)) ok = 0;      /* (the scaled data are at most 1) */
        for (int i = 0; i < m; i++) if (!(fabs(qy[i]) < 1e8)) ok = 0;
        /* 19 October 2026 (LP/SOCP), a test, off (BRISK_CONEROTSHIFT=1): the shared shift along the identity from the
         * linear part and the second-order cones only; each rotated cone moved into itself by its own shift, to the margin
         * 1. The tail of a rotated cone that holds a quadratic (s >= |r|^2 / 2, r of the size of the data) has a
         * least-squares norm that grows with its dimension; as the minimum of the shared rule it shifts every small cone by
         * the same amount: TV denoising, a rotated cone of dimension 22502, shift 14.6 for all 22500 cones of dimension 3,
         * tau 1 -> 11 and the first three steps 0.20, 0.26, 0.90. Measured on the 79 SOCP files (ISSUES 207): all status
         * lines kept, iterations 1,305 -> 1,299; TV 15/18/18/19 -> 14/16/17/18, the SVMs 40/30/44 -> 38/27/41, but MPC
         * 17 -> 21 and branch flow 20 -> 23 (they gain from the larger shared shift: Mehrotra's rule per rotated cone
         * gives 21 and 30). A shift per cone for all cones de-centres the small cones against each other (TV 17-18). */
        static int rotsh = -1; if (rotsh < 0) { const char *e_ = getenv("BRISK_CONEROTSHIFT"); rotsh = e_ ? atoi(e_) : 0; }
        const int nrs = rotsh == 1 ? S->nr_rot : 0;
        /* (21 October 2026, LP/SOCP, a test: BRISK_CONEROTSHIFT=2) a cone whose own shift to the margin 1 is more than
         * ten times that of the rest (the linear part, or the cones' median when there is none) takes its own shift, the
         * rest share theirs: one cone of a quadratic cost over all the variables (2 s >= |R x|^2, its least-squares tail
         * of the size of sqrt(dimension)) no longer sets the shift of every variable (the quadratic-cost flow on a 250 x 250
         * grid: a shared shift of about 95) */
        unsigned char *sepx = NULL, *sepz = NULL;
        if (ok && rotsh == 2 && S->nq > 0) {
            double *rxk = (double *)malloc(sizeof(double) * (size_t)S->nq), *rzk = (double *)malloc(sizeof(double) * (size_t)S->nq);
            for (int k = 0; k < S->nq; k++) {
                const int j = S->qs[k], d = S->q[k]; double a = 0, b_ = 0;
                for (int i = 1; i < d; i++) { a += px[j + i] * px[j + i]; b_ += qx[j + i] * qx[j + i]; }
                rxk[k] = fmax(0.0, 1.0 - (px[j] - sqrt(a))); rzk[k] = fmax(0.0, 1.0 - (-qx[j] - sqrt(b_)));
            }
            double bx = 0, bz = 0;
            if (S->nl > 0) { double lx = 1e300, lz = 1e300; for (int i = S->nf; i < S->nf + S->nl; i++) { lx = fmin(lx, px[i]); lz = fmin(lz, -qx[i]); } bx = fmax(0.0, 1.0 - lx); bz = fmax(0.0, 1.0 - lz); }
            else { double *t_ = (double *)malloc(sizeof(double) * (size_t)S->nq);
                   memcpy(t_, rxk, sizeof(double) * (size_t)S->nq); qsort(t_, (size_t)S->nq, sizeof(double), cmp_dbl); bx = t_[S->nq / 2];
                   memcpy(t_, rzk, sizeof(double) * (size_t)S->nq); qsort(t_, (size_t)S->nq, sizeof(double), cmp_dbl); bz = t_[S->nq / 2]; free(t_); }
            int nsx = 0, nsz = 0;
            sepx = (unsigned char *)xcalloc((size_t)S->nq, 1); sepz = (unsigned char *)xcalloc((size_t)S->nq, 1);
            for (int k = 0; k < S->nq; k++) if (rxk[k] > 10.0 * fmax(1.0, bx) || rzk[k] > 10.0 * fmax(1.0, bz)) { sepx[k] = sepz[k] = 1; nsx++; nsz++; }     /* (both sides: TV denoising 14 iterations, 17 by sides) */
            if (nsx || nsz) {      /* the shared shift without the separated cones */
                sx = 1e300; sz = 1e300;
                for (int i = S->nf; i < S->nf + S->nl; i++) { sx = fmin(sx, px[i]); sz = fmin(sz, -qx[i]); }
                for (int k = 0; k < S->nq; k++) { if (S->fcone && S->fcone[k]) continue; if (!sepx[k]) sx = fmin(sx, 1.0 - rxk[k]); if (!sepz[k]) sz = fmin(sz, 1.0 - rzk[k]); }
                if (sx > 1e299) sx = 0;
                if (sz > 1e299) sz = 0;
                if (dbg_on()) { double mxr = 0, mzr = 0; for (int k = 0; k < S->nq; k++) if (sepx[k]) { mxr = fmax(mxr, rxk[k]); mzr = fmax(mzr, rzk[k]); }
                    socp_printf("  [start: %d cones with their own shift (largest %.3e, %.3e; the rest: %.3e, %.3e; base %.3e %.3e)]\n", nsx, mxr, mzr, 1.0 - sx, 1.0 - sz, bx, bz); }
            }
            free(rxk); free(rzk);
        }
        if (ok) {
            if (nrs > 0) {
                sx = 1e300; sz = 1e300;
                for (int i = S->nf; i < S->nf + S->nl; i++) { sx = fmin(sx, px[i]); sz = fmin(sz, -qx[i]); }
                for (int k = 0; k < S->nq - nrs; k++) {
                    const int j = S->qs[k], d = S->q[k]; double a = 0, b_ = 0;
                    for (int i = 1; i < d; i++) { a += px[j + i] * px[j + i]; b_ += qx[j + i] * qx[j + i]; }
                    sx = fmin(sx, px[j] - sqrt(a)); sz = fmin(sz, -qx[j] - sqrt(b_));
                }
                if (sx > 1e299) { sx = 0; sz = 0; }
            }
            static double smin = -1; if (smin < 0) { const char *e_ = getenv("BRISK_CONESHIFTMIN"); smin = e_ ? atof(e_) : 1e-8; }   /* (a test: the margin below which the shared shift moves to margin 1) */
            /* (23 October 2026, the SOCP developer) the shift also when the margin is below 1e-1 of the point's largest entry
             * (BRISK_CONESHIFTREL=r: r; 0: off): such a point is inside its cones but at their boundary for its own size, a
             * cone's eigenvalues 7e-6 against 1 (2013_firL2L1alph, SOCP: tau fell to 7e-4 and the dual error stalled for eight
             * iterations; 22 -> 12 iterations, MOSEK 12), while a point small in all its entries is centred and stays
             * (socp_meb_20000x5, SOCP: margin and entries 5e-5; 12 iterations, 16 with a margin of 1 for all). The 158 SOCP
             * files: no status changed, 3 fewer iterations' counts (also socp_meanvar_5000x50 16 -> 13, strictmin_2D_43_dual
             * 20 -> 16), none more; 1e-1 instead of 1e-2, the same day: three more fewer (socp_lasso_1000x4000, its dual point's
             * margin 0.07 against entries of 0.71: 15 -> 9 iterations, MOSEK 9; 2013_firL2Linfeps 21 -> 19, socp_robustls_20000x2000
             * 12 -> 11), none more */
            static double srel = -1; if (srel < 0) { const char *e_ = getenv("BRISK_CONESHIFTREL"); srel = e_ ? atof(e_) : 1e-1; }
            double xn_ = 0, zn_ = 0; if (srel > 0) for (int j = S->nf; j < n; j++) { zn_ = fmax(zn_, fabs(qx[j])); xn_ = fmax(xn_, fabs(px[j])); }
            double ax = sx < fmax(smin, srel * xn_) ? 1.0 - sx : 0.0, az = sz < fmax(smin, srel * zn_) ? 1.0 - sz : 0.0;
            /* (27 October 2026, the SOCP developer, a test: BRISK_CONEMEHSTART=1) Mehrotra's start (1992) instead of the shift to
             * the margin 1: each side shifted by 1.5 times its own least-squares estimate's violation, then both by the
             * balancing terms 0.5 x'z / e'z (primal) and 0.5 x'z / e'x (dual), e the cones' identity; kappa the mean product.
             * The start then follows the scale of b in x and of c in z (b / f gives x / f), as the solution does; with the
             * margin 1 the start's own scale is 1 on both sides whatever the estimates are (joint_FC_5, SOCP: the estimates'
             * largest entries 0.57 and 1.7e-4, the final balance z0'x / x0'z 3.5e3) */
            static int meh = -1; if (meh < 0) { const char *e_ = getenv("BRISK_CONEMEHSTART"); meh = e_ ? atoi(e_) : 0; }
            int meh_ok = 0;
            if (meh && nrs == 0 && !sepx && (S->nl + S->nq) > 0) {
                double mx = 1e300, mz = 1e300;
                for (int i = S->nf; i < S->nf + S->nl; i++) { mx = fmin(mx, px[i]); mz = fmin(mz, -qx[i]); }
                for (int k = 0; k < S->nq; k++) { if (S->fcone && S->fcone[k]) continue; const int j = S->qs[k], d = S->q[k]; double a_ = 0, b_ = 0;
                    for (int i = 1; i < d; i++) { a_ += px[j + i] * px[j + i]; b_ += qx[j + i] * qx[j + i]; }
                    mx = fmin(mx, px[j] - sqrt(a_)); mz = fmin(mz, -qx[j] - sqrt(b_)); }
                const double d1x = fmax(-1.5 * mx, 0.0), d1z = fmax(-1.5 * mz, 0.0);
                double xz = 0, ex = 0, ez = 0;
                for (int i = S->nf; i < S->nf + S->nl; i++) { const double xi = px[i] + d1x, zi = -qx[i] + d1z; xz += xi * zi; ex += xi; ez += zi; }
                for (int k = 0; k < S->nq; k++) { if (S->fcone && S->fcone[k]) continue; const int j = S->qs[k], d = S->q[k];
                    const double x0 = px[j] + d1x, z0 = -qx[j] + d1z; xz += x0 * z0; ex += x0; ez += z0;
                    for (int i = 1; i < d; i++) xz += px[j + i] * (-qx[j + i]); }
                if (xz > 0 && ex > 0 && ez > 0 && isfinite(xz)) {
                    const double axm = d1x + 0.5 * xz / ez, azm = d1z + 0.5 * xz / ex;
                    if (mx + axm > 0 && mz + azm > 0 && isfinite(axm) && isfinite(azm)) { ax = axm; az = azm; meh_ok = 1; }
                }
                if (dbg_on()) socp_printf("  [start: Mehrotra's shifts %s: primal %.3e, dual %.3e (margins %.3e, %.3e)]\n", meh_ok ? "taken" : "not taken", ax, az, mx, mz);
            }
            if (dbg_on()) { double zn = 0, xn = 0; for (int j = S->nf; j < n; j++) { zn = fmax(zn, fabs(qx[j])); xn = fmax(xn, fabs(px[j])); }
                socp_printf("  [start: shared shift primal %.3e (largest entry %.3e, margin %.3e), dual %.3e (largest entry %.3e, margin %.3e)]\n", ax, xn, sx, az, zn, sz); }
            for (int j = 0; j < S->nf; j++) { S->x[j] = px[j]; S->z[j] = 0.0; }
            for (int i = S->nf; i < S->nf + S->nl; i++) { S->x[i] = px[i] + ax; S->z[i] = -qx[i] + az; }
            for (int k = 0; k < S->nq; k++) {
                const int j = S->qs[k], d = S->q[k];
                double axk = ax, azk = az;
                if (k >= S->nq - nrs || (S->fcone && S->fcone[k]) || (sepx && (sepx[k] || sepz[k]))) {      /* a rotated cone, a free cone or a separated one: its own shift, to the margin 1 */
                    double a = 0, b_ = 0;
                    for (int i = 1; i < d; i++) { a += px[j + i] * px[j + i]; b_ += qx[j + i] * qx[j + i]; }
                    const double mx = px[j] - sqrt(a), mz = -qx[j] - sqrt(b_);
                    if (!sepx || sepx[k] || k >= S->nq - nrs || (S->fcone && S->fcone[k])) axk = mx < 1.0 ? 1.0 - mx : 0.0;
                    if (!sepx || sepz[k] || k >= S->nq - nrs || (S->fcone && S->fcone[k])) azk = mz < 1.0 ? 1.0 - mz : 0.0;
                }
                S->x[j] = px[j] + axk; S->z[j] = -qx[j] + azk;
                for (int i = 1; i < d; i++) { S->x[j + i] = px[j + i]; S->z[j + i] = -qx[j + i]; }
            }
            for (int i = 0; i < m; i++) S->y[i] = qy[i];
            free(sepx); free(sepz); sepx = sepz = NULL;
            if (epid) {
                double mu_ = 0; for (int i = S->nf; i < S->nf + S->nl; i++) mu_ += S->x[i] * S->z[i];
                mu_ = S->nl > 0 ? mu_ / S->nl : 1.0; if (!(mu_ > 0) || !isfinite(mu_)) mu_ = 1.0;
                for (int k = 0; k < S->nq; k++) {
                    if (!(epid[k] > 0)) continue;
                    const int j = S->qs[k], d = S->q[k]; const double dl = epid[k];
                    double r2 = 0; for (int i = 2; i < d; i++) r2 += px[j + i] * px[j + i];
                    static int edet = -1; if (edet < 0) { const char *e_ = getenv("BRISK_CONEEPIDET"); edet = e_ ? atoi(e_) : 0; }   /* (test: the determinant 1, delta^2, |r|^2 + delta^2) */
                    const double det = edet == 1 ? dl * dl : edet == 2 ? r2 + dl * dl : 1.0;
                    const double x1 = (det + r2 - dl * dl) / (2.0 * dl), x0 = x1 + dl;
                    S->x[j] = x0; S->x[j + 1] = x1; for (int i = 2; i < d; i++) S->x[j + i] = px[j + i];
                    S->z[j] = mu_ * x0 / det; S->z[j + 1] = -mu_ * x1 / det; for (int i = 2; i < d; i++) S->z[j + i] = -mu_ * px[j + i] / det;
                    if (dbg_on()) socp_printf("  [start: rotated cone %d (dimension %d) at its fixed entry: delta %.3e, |r|^2 %.3e, x0 %.3e, mu %.3e]\n", k, d, dl, r2, x0, mu_);
                }
                free(epid); epid = NULL;
            }
            {   /* (24 October 2026, the SOCP developer; the default, BRISK_CONEFIXST=0: off, 1: the tail 0, a test) a second-order cone
                 * whose first entry is fixed by a row of its own (x_0 = rho > 0: a norm bound |u| <= rho, the thermal limits of
                 * power flow) starts on that row: at the cone's centre dilated to its fixed entry, x = (rho, u) with u the
                 * least-squares tail cut to at most 0.9 rho (a dilation is an automorphism of the cone, so the point is as
                 * central as the centre itself when u = 0), and its dual at the central point mu x^-1 of the linear part's mean
                 * product. The shared shift put x_0 near 1 whatever rho is, off its row. The SOC relaxations of power flow
                 * (SOCPs; bench/socp_opf): 1354_pegase 77 -> 44 iterations, 4661_sdet 124 -> 70, 6468_rte 286 -> 69 (38 -> 11 s);
                 * the 150 other SOCP files and the 48 QCQPs unchanged (ISSUES 269). The second-order counterpart of the epigraph
                 * start of a rotated cone above */
                static int fst = -1; if (fst < 0) { const char *e_ = getenv("BRISK_CONEFIXST"); fst = e_ ? atoi(e_) : 2; }
                if (fst > 0 && S->nq > S->nr_rot) {
                    char *el = (char *)calloc((size_t)m + 1, 1); int *fr = (int *)malloc(sizeof(int) * ((size_t)S->nq + 1)), *fp = (int *)malloc(sizeof(int) * ((size_t)S->nq + 1));
                    double *fa = (double *)malloc(sizeof(double) * ((size_t)S->nq + 1));
                    const int nfx = ne_find_fixed(S, el, fr, fp, fa);
                    double mu_ = 0; for (int i = S->nf; i < S->nf + S->nl; i++) mu_ += S->x[i] * S->z[i];
                    mu_ = S->nl > 0 ? mu_ / S->nl : 1.0; if (!(mu_ > 0) || !isfinite(mu_)) mu_ = 1.0;
                    int nset = 0;
                    for (int k = 0; nfx > 0 && k < S->nq - S->nr_rot; k++) {
                        if (fr[k] < 0 || fp[k] != 0) continue;
                        const double rho = S->b[fr[k]] / fa[k]; if (!(rho > 0) || !isfinite(rho)) continue;
                        const int j = S->qs[k], d = S->q[k];
                        double t2 = 0; for (int i = 1; i < d; i++) t2 += px[j + i] * px[j + i];
                        const double cut = fst == 2 && t2 > 0 ? fmin(1.0, 0.9 * rho / sqrt(t2)) : 0.0;
                        S->x[j] = rho; for (int i = 1; i < d; i++) S->x[j + i] = cut * px[j + i];
                        double u2 = 0; for (int i = 1; i < d; i++) u2 += S->x[j + i] * S->x[j + i];
                        const double det = rho * rho - u2;
                        S->z[j] = mu_ * rho / det; for (int i = 1; i < d; i++) S->z[j + i] = -mu_ * S->x[j + i] / det;
                        nset++;
                    }
                    if (dbg_on()) socp_printf("  [start: %d second-order cones at the dilation to their fixed first entry (mu %.3e)]\n", nset, mu_);
                    free(el); free(fr); free(fp); free(fa);
                }
            }
        }
    }
    /* (20 October 2026, LP/SOCP, a test: BRISK_CONEMEHSTART=1) Mehrotra's balancing of the start: after the least-squares point
     * and its shift to the margin, both sides are shifted along the identity by half the mean product measured against the
     * other side (x += 0.5 x'z / e'z, z += 0.5 x'z / e'x), so that the first products are of one size; the SVM's start has x
     * of the size of the data and z near the unit shift (24 iterations; with b a tenth of its size 15) */
    if (getenv("BRISK_CONEMEHSTART") && S->nl + S->nq > 0) {
        double xz = 0, ex = 0, ez = 0;
        for (int i = S->nf; i < n; i++) xz += S->x[i] * S->z[i];
        for (int i = S->nf; i < S->nf + S->nl; i++) { ex += S->x[i]; ez += S->z[i]; }
        for (int k = 0; k < S->nq; k++) { ex += S->x[S->qs[k]]; ez += S->z[S->qs[k]]; }
        if (xz > 0 && ex > 0 && ez > 0 && isfinite(xz)) {
            const double dx2 = 0.5 * xz / ez, dz2 = 0.5 * xz / ex;
            for (int i = S->nf; i < S->nf + S->nl; i++) { S->x[i] += dx2; S->z[i] += dz2; }
            for (int k = 0; k < S->nq; k++) { S->x[S->qs[k]] += dx2; S->z[S->qs[k]] += dz2; }
            if (dbg_on()) socp_printf("  [start balanced: x'z %.3e, shifts %.3e (x) %.3e (z)]\n", xz, dx2, dz2);
        }
    }
    {   /* (22 October 2026, a test: BRISK_CONESTARTZE=1) the dual side from the identity (y = 0, z = e), the primal side from the
         * least-squares point as above: ISSUES 257 */
        static int ze = -1; if (ze < 0) { const char *e_ = getenv("BRISK_CONESTARTZE"); ze = e_ ? atoi(e_) : 0; }
        if (ze && !opt->wx) {
            for (int i = 0; i < m; i++) S->y[i] = 0.0;
            for (int i = S->nf; i < S->nf + S->nl; i++) S->z[i] = 1.0;
            for (int k = 0; k < S->nq; k++) { const int j = S->qs[k]; S->z[j] = 1.0; for (int i = 1; i < S->q[k]; i++) S->z[j + i] = 0.0; }
        }
    }
    {   /* (21 October 2026, LP/SOCP, a test: BRISK_CONESTARTXZ="fx,fz" the start's linear x and z multiplied by fx and fz) */
        const char *e_ = getenv("BRISK_CONESTARTXZ"); double fx = 1, fz = 1;
        if (e_ && sscanf(e_, "%lf,%lf", &fx, &fz) == 2) for (int i = S->nf; i < S->nf + S->nl; i++) { S->x[i] *= fx; S->z[i] *= fz; }
    }
    if (getenv("BRISK_CONEKAP0") || getenv("BRISK_CONEMEHSTART")) {     /* (test: kappa0 so that tau kappa is the mean product of x and z) */
        double xz0 = 0; for (int j = S->nf; j < n; j++) xz0 += S->x[j] * S->z[j];
        const double mx = xz0 / fmax(1.0, (double)(S->nl + S->nq));
        if (mx > 0 && isfinite(mx)) S->kap = mx / S->tau;
    }
    if (opt->wx) {
        /* the warm start: the given point in the solver's coordinates (a rotated cone's first two entries mapped to their
         * half-sum and half-difference times sqrt 2, x / (E_c s_b), y / (D_r s_c), E_c z / s_c), then wlam of it and
         * 1 - wlam of the cones' identity (the free variables' x: wlam of it, their z: 0); kappa the mean product */
        const double lam = opt->wlam > 0 && opt->wlam <= 1 ? opt->wlam : 0.99, rs2 = sqrt(0.5);
        for (int j = 0; j < n; j++) { S->x[j] = opt->wx[j]; S->z[j] = opt->wz ? opt->wz[j] : 0.0; }
        for (int k = P->nq; k < S->nq; k++) {
            const int j = S->qs[k]; const double g = rgam ? rgam[k] : 1.0;
            const double a = g * S->x[j], b = S->x[j + 1] / g, c = S->z[j] / g, d = g * S->z[j + 1];     /* (u' = gamma u, v' = v / gamma; z inversely) */
            S->x[j] = rs2 * (a + b); S->x[j + 1] = rs2 * (a - b); S->z[j] = rs2 * (c + d); S->z[j + 1] = rs2 * (c - d);
        }
        for (int j = 0; j < n; j++) { S->x[j] /= S->Ec[j] * S->sb; S->z[j] *= S->Ec[j] / S->scl; }
        for (int i = 0; i < m; i++) S->y[i] = opt->wy ? lam * opt->wy[i] / (S->Dr[i] * S->scl) : 0.0;
        if (rgam && opt->wy) for (int k = P->nq; k < S->nq; k++) if (rrow[k] >= 0) S->y[rrow[k]] /= rrho[k];     /* (y' = y / rho on a fixed side's row) */
        for (int j = 0; j < S->nf; j++) { S->x[j] *= lam; S->z[j] = 0.0; }
        for (int j = S->nf; j < n; j++) { S->x[j] *= lam; S->z[j] *= lam; }
        for (int i = S->nf; i < S->nf + S->nl; i++) { S->x[i] += 1.0 - lam; S->z[i] += 1.0 - lam; }
        for (int k = 0; k < S->nq; k++) { S->x[S->qs[k]] += 1.0 - lam; S->z[S->qs[k]] += 1.0 - lam; }
        S->tau = 1.0;
        double xz = 0; for (int j = S->nf; j < n; j++) xz += S->x[j] * S->z[j];
        S->kap = fmax(xz / fmax(1.0, (double)(S->nl + S->nq)), 1e-8);
    }
    const double nu = (double)(S->nl + S->nq + 1);
    double *dbg_x0 = NULL, *dbg_z0 = NULL;
    if (dbg_on() && S->nl > 0) {     /* (21 October 2026, LP/SOCP: the start's scale against the solution's, printed at the end) */
        double sx = 0, sz = 0; for (int i = S->nf; i < S->nf + S->nl; i++) { sx += S->x[i]; sz += S->z[i]; }
        dbg_x0 = (double *)malloc(sizeof(double) * (size_t)n); dbg_z0 = (double *)malloc(sizeof(double) * (size_t)n);
        memcpy(dbg_x0, S->x, sizeof(double) * (size_t)n); memcpy(dbg_z0, S->z, sizeof(double) * (size_t)n);
        socp_printf("  [start, linear part: mean x %.3e, mean z %.3e; tau %.2e kappa %.2e]\n", sx / S->nl, sz / S->nl, S->tau, S->kap);
    }
    int status = SOCP_MAXIT, it;
    if (S->verbose > 0) {
        socp_printf("BRISK second-order cone solver: m = %d, n = %d (free %d, linear %d, cones %d", m, n, S->nf, S->nl, S->nq);
        if (S->nq) { int mx = 0; for (int k = 0; k < S->nq; k++) if (S->q[k] > mx) mx = S->q[k]; socp_printf(", largest %d", mx); }
        if (S->kkt_defer) socp_printf("), nnz(A) = %d, normal equations of %d rows, nnz(L) = %ld\n", S->Ap[n], m, S->ne_lnz);
        else socp_printf("), nnz(A) = %d, factor: %d unknowns, nnz(L) = %ld\n", S->Ap[n], S->nk, S->F.lnz);
        if (dbg_on() && S->nf > 0) {     /* (the rows that hold free columns only: a saddle block with no cone in it) */
            unsigned char *rk_ = (unsigned char *)calloc((size_t)m + 1, 1); long nfo = 0, nfz = 0;
            for (int j = S->nf; j < n; j++) for (int p = S->Ap[j]; p < S->Ap[j + 1]; p++) rk_[S->Ai[p]] = 1;
            for (int i = 0; i < m; i++) if (!rk_[i]) nfo++;
            for (int j = 0; j < S->nf; j++) for (int p = S->Ap[j]; p < S->Ap[j + 1]; p++) if (!rk_[S->Ai[p]]) nfz++;
            socp_printf("  [%ld rows hold free columns only (%ld entries)]\n", nfo, nfz); free(rk_);
        }
        {   extern int schol_panel_threads(void); const int pt = schol_panel_threads();   /* (17 October 2026: the large blocks of the factorization) */
            if (pt > 1) socp_printf("Number of threads: %d (the large blocks of the factorization; the rest is sequential)\n", pt);
            else socp_printf("Number of threads: 1\n"); }
    }
    if (S->verbose > 1) socp_printf("  it      pobj             dobj         pinf     dinf     gap      mu       tau    step  sigma  ref\n");
    double pobj = 0, dobj = 0, e_p = 1, e_d = 1, e_g = 1, e_c = 1;
    static double ne_k = -1; if (ne_k < 0) { const char *e_ = getenv("BRISK_CONENEK"); ne_k = e_ ? atof(e_) : 3.0; }   /* (test: the GMRES steps per solve above which the normal equations are left) */
    /* (18 October 2026, LP/SOCP: an early tolerance of 1e-7 or 1e-8 instead of 1e-9 was measured and not kept: a third
     * fewer solves and the DIMACS sched files better, but the end phase of TV denoising and of the clustering problem
     * is on a knife edge (replaced pivots, one large cone far from the central path), and any change of the path flips
     * one of them into the second attempt: socp_tv_200 19 -> 52 iterations at 1e-7, socp_sonclust 20 -> 49 at 1e-8;
     * ISSUES 207) */
    /* (18 October 2026, the night: 1e-8 now that the rotated cones are balanced, which removed the knife edge: 79 SOCP
     * files 3.3 % faster in the geometric mean, the DIMACS sched files better (one REDUCED -> OPTIMAL, one NUMERICAL
     * DIFFICULTIES -> REDUCED, sched_50_50_orig 50 -> 23 iterations), socp_sonclust 19 iterations either way; 1e-7 took
     * the powered descent from 35 to 82 iterations) */
    double gm_switch = 1e-4, gm_tol_early = 1e-8, gm_tol_late = 1e-12;
    { const char *e_ = getenv("BRISK_CONEGMLATE"); if (e_) gm_tol_late = atof(e_); }   /* (test: the late tolerance) */
    { const char *e = getenv("BRISK_CONEGMTOL"); if (e) { sscanf(e, "%lf,%lf,%lf", &gm_tol_early, &gm_tol_late, &gm_switch); } }
    const int mcc_ref = 10, mcc_max = getenv("BRISK_CONEMCC") ? atoi(getenv("BRISK_CONEMCC")) : 3;
    const double mcc_da = getenv("BRISK_CONEMCCDA") ? atof(getenv("BRISK_CONEMCCDA")) : 0.2, mcc_bmin = getenv("BRISK_CONEMCCB") ? atof(getenv("BRISK_CONEMCCB")) : 0.1, mcc_bmax = 1.0 / mcc_bmin, mcc_acc = getenv("BRISK_CONEMCCACC") ? atof(getenv("BRISK_CONEMCCACC")) : 0.1, step_frac = opt->step > 0 && opt->step < 1 ? opt->step : 0.99;
    double *dxa = (double *)xcalloc((size_t)n, sizeof(double)), *dya = (double *)xcalloc((size_t)m + 1, sizeof(double)), *dza = (double *)xcalloc((size_t)n, sizeof(double));
    double *ctv = (double *)xcalloc((size_t)n, sizeof(double)), *dxs = (double *)xcalloc((size_t)n, sizeof(double)), *dzs = (double *)xcalloc((size_t)n, sizeof(double));
    /* (20 October 2026, LP/SOCP, a test: BRISK_CONEARC=1, the arc search after a short Mehrotra step, as the LP method's
     * BRISK_LPARC=2: Mehrotra's direction is d1 + d2 / 2 of the arc's two, d2 the solve for -2 (W dx_a) o (W^-1 dz_a)
     * alone; the point moves on p + sin(a) d1 + (1 - cos a) d2 when sin(a) beats the line's step; BRISK_CONEARCT the step
     * below which it is tried) */
    static int arcq = -1; static double arct = 0.5;
    /* (20 October 2026, LP/SOCP, tests on the centering: BRISK_CONESIGP the power of Mehrotra's sigma = (1 - alpha_aff)^p
     * (3), BRISK_CONESIGMIN a floor on sigma (0), BRISK_CONECORRW the weight of the second-order term in the corrector
     * (1; -1: the affine step itself)) */
    static double sigp = -1, sigmin = 0, corrw = 1;
    if (sigp < 0) { const char *e_ = getenv("BRISK_CONESIGP"); sigp = e_ ? atof(e_) : 3.0; e_ = getenv("BRISK_CONESIGMIN"); sigmin = e_ ? atof(e_) : 0.0; e_ = getenv("BRISK_CONECORRW"); corrw = e_ ? atof(e_) : 1.0; }
    if (arcq < 0) { const char *e_ = getenv("BRISK_CONEARC"); arcq = e_ ? atoi(e_) : 0; const char *t_ = getenv("BRISK_CONEARCT"); if (t_) arct = atof(t_); }
    double *axa = NULL, *aza = NULL, *ay1 = NULL, *alx = NULL, *alz = NULL, *aly = NULL; double atau = 0, akap = 0, altau = 0, alkap = 0; long narc = 0;
    if (arcq) { axa = (double *)xcalloc((size_t)n, sizeof(double)); aza = (double *)xcalloc((size_t)n, sizeof(double)); ay1 = (double *)xcalloc((size_t)m + 1, sizeof(double));
                alx = (double *)xcalloc((size_t)n, sizeof(double)); alz = (double *)xcalloc((size_t)n, sizeof(double)); aly = (double *)xcalloc((size_t)m + 1, sizeof(double)); }
    double mu0 = 1.0;
    /* up to mcc_max correctors in every iteration (a rule by the cost of a factorization against a
     * solve was tried: the iterations saved are worth the two solves of a corrector on all the sets) */
    const int mcc_now = mcc_max;
    {
        double tf, tsv; kkt_model(S, &tf, &tsv);
        S->rr_est = tf / tsv;
    }
    int ddbump = 0;
    int nfeas = 0, feas_pending = 0, nsmall = 0, nf_retry = 0, nf_restore = 0; double nf_sv[4] = { 0, 0, 0, 1 }; const int feas_on = getenv("BRISK_CONENOFEAS") == NULL;
    double feas_prev = 0;
    double *sv_x = (double *)xcalloc((size_t)n, sizeof(double)), *sv_z = (double *)xcalloc((size_t)n, sizeof(double)), *sv_y = (double *)xcalloc((size_t)m + 1, sizeof(double));
    double c_sw0 = 0;   /* (the modelled cost of the iteration that switched the small cones to their eigenbasis) */
    double t_chk = 0;   /* (the time of the last check of the time limit, since t0) */
    double tq[6] = {0, 0, 0, 0, 0, 0}, tq_last = -1;   /* (debug: the iteration's time by part: residuals, errors, rays, scaling and factorization, the rest) */
    for (it = 0; ; it++) {
        if (tq_last >= 0) tq[4] += sc_time() - tq_last;
        tq_last = sc_time();
        /* ---- residuals of the homogeneous model (scaled problem) */
        for (int i = 0; i < m; i++) rP[i] = S->b[i] * S->tau;
        A_mul(S, -1.0, S->x, rP);                                        /* b tau - A x */
        for (int j = 0; j < n; j++) rD[j] = S->c[j] * S->tau - S->z[j];
        At_mul(S, -1.0, S->y, rD);                                       /* c tau - A'y - z */
        const double cx = dotv(n, S->c, S->x), byv = dotv(m, S->b, S->y);
        const double rG = byv - cx - S->kap;
        double xz = 0; for (int j = S->nf; j < n; j++) xz += S->x[j] * S->z[j];
        const double mu = (xz + S->tau * S->kap) / nu;
        tq[0] += sc_time() - tq_last; tq_last = sc_time();
        /* ---- errors of the problem as given */
        {
            const double f = 1.0 / S->tau;
            for (int j = 0; j < n; j++) { xo[j] = S->Ec[j] * S->x[j] * f * S->sb; zo[j] = S->z[j] * f * S->scl / S->Ec[j]; }
            cone_unboost(S, xo, zo);
            for (int i = 0; i < m; i++) yo[i] = S->Dr[i] * S->y[i] * f * S->scl;
            if (rgam) rot_ungamma(P->nq, S->nq, S->qs, rgam, rrho, rrow, xo, yo, zo);
            if (rgam && rotrow1) rot_rowdual(P->nq, S->nq, S->qs, rsd, rrow, rae, P->c, yo, zo);
            double *ra = S->w1, *rb = S->w2;
            /* (19 October 2026, LP/SOCP: on the threads with the row copy, each row's sum in the order of the columns
             * as by columns, so the same bits) */
            if (S->R0x && cone_apar(S)) {
                #pragma omp parallel for schedule(static) num_threads(g_cone_nt)
                for (int i = 0; i < m; i++) { double v = -b0[i]; for (int q = S->Rp[i]; q < S->Rp[i + 1]; q++) { const double xj = xo[S->Ri[q]]; if (xj != 0.0) v += S->R0x[q] * xj; } ra[i] = v; }
            } else {
                for (int i = 0; i < m; i++) ra[i] = -b0[i];
                for (int j = 0; j < n; j++) { const double xj = xo[j]; if (xj != 0.0) for (int p = S->Ap[j]; p < S->Ap[j + 1]; p++) ra[S->Ai[p]] += A0x[p] * xj; }
            }
            #pragma omp parallel for schedule(static) num_threads(g_cone_nt) if (S->R0x && cone_apar(S))
            for (int j = 0; j < n; j++) { double s = 0; for (int p = S->Ap[j]; p < S->Ap[j + 1]; p++) s += A0x[p] * yo[S->Ai[p]]; rb[j] = c0[j] - s - zo[j]; }
            pobj = dotv(n, c0, xo); dobj = dotv(m, b0, yo);
            e_p = nrm2(m, ra) / (1.0 + nb0); e_d = nrm2(n, rb) / (1.0 + nc0);
            /* (22 October 2026) the cutoff: c'x >= b'y + r'x >= b'y - |r| |x| for a feasible x, r = c - A'y - z, z in the dual
             * cone (the iterate's is interior); |x| estimated by the iterate's */
            if (opt->cutoff < INFINITY && S->tau > 0) {
                const double lb = dobj - nrm2(n, rb) * fmax(1.0, nrm2(n, xo));
                if (lb > opt->cutoff) { R->dobj = lb; R->pobj = pobj; status = SOCP_CUTOFF; break; }
            }
            /* (22 October 2026, LP/SOCP) each row also against its own scale: |r_i| / (1 + |b_i| + sum_j |a_ij x_j|). The
             * DIMACS measure divides by 1 + max|b|, and in an LP's cone form rows of bounds near 1e7 made a residual of
             * 0.04 in small rows pass as 4e-9: piperout-08 (LP) OPTIMAL with its rows as read violated by 1.2e-2 of their
             * activity and the objective 98,999 against 99,129 (ISSUES 236, from the simplex developer's ISSUES 418). It counts at a
             * point the DIMACS measure accepts, when it exceeds 1e-4 (piperout-08: 2.3e-3). BRISK_CONENOROWERR=1: the
             * DIMACS measure alone */
            { static int rowerr = -1; if (rowerr < 0) rowerr = getenv("BRISK_CONENOROWERR") == NULL;
              if (rowerr && m > 0) {
                if (!g_ract || g_ract_m < m) { free(g_ract); g_ract = (double *)malloc(sizeof(double) * ((size_t)m + 1)); g_ract_m = m; }
                double *act = g_ract;
                for (int i = 0; i < m; i++) act[i] = 0.0;
                for (int j = 0; j < n; j++) { const double xj = xo[j]; if (xj != 0.0) for (int p = S->Ap[j]; p < S->Ap[j + 1]; p++) act[S->Ai[p]] += fabs(A0x[p] * xj); }
                double er = 0;
                for (int i = 0; i < m; i++) { const double v = fabs(ra[i]) / (1.0 + fabs(b0[i]) + act[i]); if (v > er) er = v; }
                if (dbg_on() > 1) socp_printf("      [the rows' own primal error: largest %.1e]\n", er);
                if (e_p <= opt->tol && er > 1e-4) e_p = er;     /* (a guard at a point the DIMACS measure would accept: a row violated by more than 1e-4 of its own size; the iteration's path is not changed. Counted at 1e-8 or 1e-6 always, socp_rocket_4000 (SOCP) ended at 1.3e-6 a row against 3e-9 and went 35 -> 78 iterations, REDUCED) */
              } }
            if (dbg_on() > 1) {          /* (18 October 2026: the dual residual by kind of column, its largest entry) */
                double sf = 0, sl = 0, sq = 0, mx = 0; int jm = -1;
                for (int j = 0; j < n; j++) { const double r2 = rb[j] * rb[j]; if (j < S->nf) sf += r2; else if (j < S->nf + S->nl) sl += r2; else sq += r2; if (fabs(rb[j]) > mx) { mx = fabs(rb[j]); jm = j; } }
                socp_printf("      [dual residual: free %.1e linear %.1e cones %.1e; largest %.1e at column %d (%s), |y| %.1e, x %.1e z %.1e]\n", sqrt(sf) / (1.0 + nc0), sqrt(sl) / (1.0 + nc0), sqrt(sq) / (1.0 + nc0), mx, jm, jm < S->nf ? "free" : jm < S->nf + S->nl ? "linear" : "cone", nrm2(m, yo), jm >= 0 ? xo[jm] : 0.0, jm >= 0 ? zo[jm] : 0.0);
            }
            e_g = (pobj - dobj) / (1.0 + fabs(pobj) + fabs(dobj));
            double xzo = 0; for (int j = S->nf; j < n; j++) xzo += xo[j] * zo[j];
            e_c = xzo / (1.0 + fabs(pobj) + fabs(dobj));
        }
        if (feas_pending) {          /* a step on the equations that did not reduce the residuals is taken back */
            feas_pending = 0;
            if (!(fmax(e_p, e_d) < feas_prev)) {
                memcpy(S->x, sv_x, sizeof(double) * (size_t)n); memcpy(S->z, sv_z, sizeof(double) * (size_t)n); memcpy(S->y, sv_y, sizeof(double) * (size_t)m);
                nfeas = 1000; it--;
                continue;
            }
        }
        /* (21 October 2026, LP/SOCP) the dual point measured by its own cone membership: when the primal residual and the
         * gap meet the tolerance and the dual residual c - A'y - z does not, z is taken as the projection of c - A'y on
         * the dual cone (the free columns: 0), so that the dual residual is the distance of c - A'y from the cone; the
         * point is taken when every error, the complementarity with the new z included, meets the tolerance. At the end
         * of TV denoising and the clustering problem the step's z = r3 - H dx misses the dual equation by the error of
         * H dx at extreme scaling (replaced pivots, the dual residual 4e-8 -> 1e-6 while the solves are within 1e-14);
         * y is good and c - A'y lies within 1e-9 of the cone (MOSEK reports this z). BRISK_CONENOZPROJ=1: off. */
        { static int zpj = -1; if (zpj < 0) zpj = getenv("BRISK_CONENOZPROJ") == NULL;
          if (zpj && S->tau > 0 && e_p <= opt->tol && fabs(e_g) <= opt->tol && e_d > opt->tol) {
            double *w = S->w2, *zp = S->w3;      /* (w2 holds c0 - A0'yo - zo: rb above) */
            for (int j = 0; j < n; j++) w[j] += zo[j];
            for (int j = 0; j < S->nf; j++) zp[j] = 0.0;
            for (int j = S->nf; j < S->nf + S->nl; j++) zp[j] = w[j] > 0 ? w[j] : 0.0;
            for (int k = 0; k < S->nq; k++) {
                const int j = S->qs[k], d = S->q[k]; double u = 0;
                for (int i = 1; i < d; i++) u += w[j + i] * w[j + i];
                u = sqrt(u); const double t = w[j];
                if (u <= t) for (int i = 0; i < d; i++) zp[j + i] = w[j + i];
                else if (u <= -t) for (int i = 0; i < d; i++) zp[j + i] = 0.0;
                else { const double a = 0.5 * (t + u); zp[j] = a; for (int i = 1; i < d; i++) zp[j + i] = a * w[j + i] / u; }
            }
            double r2 = 0, xz2 = 0;
            for (int j = 0; j < n; j++) { const double r_ = w[j] - zp[j]; r2 += r_ * r_; }
            for (int j = S->nf; j < n; j++) xz2 += xo[j] * zp[j];
            const double e_d2 = sqrt(r2) / (1.0 + nc0), e_c2 = xz2 / (1.0 + fabs(pobj) + fabs(dobj));
            if (dbg_on()) socp_printf("      [the dual point by cone membership: dual residual %.1e -> %.1e, complementarity %.1e -> %.1e]\n", e_d, e_d2, e_c, e_c2);
            if (fmax(e_d2, e_c2) <= opt->tol) { memcpy(zo, zp, sizeof(double) * (size_t)n); e_d = e_d2; e_c = e_c2; S->zproj_used++; }
          } }
        const double err = fmax(fmax(e_p, e_d), fmax(fabs(e_g), e_c));
        if (opt->wsave > 0 && !R->cx && S->tau > 0 && fabs(e_g) <= opt->wsave && fmax(e_p, e_d) <= 1e-2) {
            R->cx = (double *)malloc(sizeof(double) * ((size_t)n + 1)); R->cy = (double *)malloc(sizeof(double) * ((size_t)m + 1)); R->cz = (double *)malloc(sizeof(double) * ((size_t)n + 1));
            memcpy(R->cx, xo, sizeof(double) * (size_t)n); memcpy(R->cy, yo, sizeof(double) * (size_t)m); memcpy(R->cz, zo, sizeof(double) * (size_t)n);
        }
        if (err < best && S->tau > 0) { best = err; memcpy(bx, xo, sizeof(double) * (size_t)n); memcpy(by, yo, sizeof(double) * (size_t)m); memcpy(bz, zo, sizeof(double) * (size_t)n); best_e[0] = e_p; best_e[1] = e_d; best_e[2] = e_g; best_e[3] = e_c; R->pobj = pobj; R->dobj = dobj; }
        if (S->verbose > 1 && it == 0) socp_printf("%4d %+.9e %+.9e %8.1e %8.1e %8.1e %8.1e %8.1e\n", it, pobj, dobj, e_p, e_d, e_g, mu, S->tau);
        if (dbg_x0) {    /* the balance z0'x / x0'z over the linear part (at the solution: z0'x* / x0'z*) */
            double a_ = 0, b_ = 0; for (int i = S->nf; i < S->nf + S->nl; i++) { a_ += dbg_z0[i] * S->x[i]; b_ += dbg_x0[i] * S->z[i]; }
            double mx_ = 0, mz_ = 0; for (int i = S->nf; i < S->nf + S->nl; i++) { mx_ += S->x[i]; mz_ += S->z[i]; }
            socp_printf("      [balance z0'x / x0'z (linear part) %.3e; mean x / tau %.3e, mean z / tau %.3e]\n", b_ > 0 ? a_ / b_ : 0.0, mx_ / S->nl / S->tau, mz_ / S->nl / S->tau);
        }
        if (err <= opt->tol) {
            if (dbg_on() && S->nl > 0) {
                double sx = 0, sz = 0, mx = 0, mz = 0; for (int i = S->nf; i < S->nf + S->nl; i++) { sx += S->x[i]; sz += S->z[i]; if (S->x[i] > mx) mx = S->x[i]; if (S->z[i] > mz) mz = S->z[i]; }
                socp_printf("  [solution / tau, linear part: mean x %.3e (max %.2e), mean z %.3e (max %.2e); tau %.2e]\n", sx / S->nl / S->tau, mx / S->tau, sz / S->nl / S->tau, mz / S->tau, S->tau);
            }
            status = SOCP_OPTIMAL; break;
        }
        tq[1] += sc_time() - tq_last; tq_last = sc_time();
        /* ---- infeasibility: improving rays of the scaled problem (the scaling keeps them rays) */
        {
            const double nx_ = nrmi(n, S->x), ny_ = fmax(nrmi(m, S->y), nrmi(n, S->z));
            const int cert_proj = getenv("BRISK_CONENOCERTPROJ") == NULL;
            const double cert_tau = 1e-3, cert_gate = 1e-2;      /* (earlier attempts, tau < 1e-1 kappa or a gate of 1e-1: no earlier certificate on mpc_1500) */
            if (byv > 1e-12 * ny_ && S->tau < cert_tau * fmax(1.0, S->kap)) {
                double *w = S->w1; memcpy(w, S->z, sizeof(double) * (size_t)n); At_mul(S, 1.0, S->y, w);      /* A'y + z */
                if (nrmi(n, w) <= opt->tol * byv) { status = SOCP_PINF; break; }
                if (cert_proj && nrmi(n, w) <= cert_gate * byv && cert_project_primal(S, opt->tol, byv)) { if (S->verbose > 1) socp_printf("      [primal infeasibility certificate by projection]\n"); status = SOCP_PINF; break; }
            }
            if (cx < -1e-12 * nx_ && S->tau < cert_tau * fmax(1.0, S->kap)) {
                double *w = S->w1; for (int i = 0; i < m; i++) w[i] = 0.0; A_mul(S, 1.0, S->x, w);
                if (nrmi(m, w) <= opt->tol * (-cx)) { status = SOCP_DINF; break; }
                if (cert_proj && nrmi(m, w) <= cert_gate * (-cx) && cert_project_dual(S, opt->tol, dx, dy)) { if (S->verbose > 1) socp_printf("      [dual infeasibility certificate by projection]\n"); status = SOCP_DINF; break; }
            }
        }
        if (it >= opt->maxit) { status = SOCP_MAXIT; break; }
        /* (24 October 2026, the SOCP developer) an iteration is not started when the last one's duration says it would end
         * beyond the limit: the clock is read here once an iteration, so a problem whose iterations take tens of seconds
         * overran it by most of one (socp_svmsparse_200kx20k, SOCP: no result 30 s after -timelimit 600) */
        if (opt->timelimit > 0) {
            const double tnow = sc_time() - t0, dit = t_chk > 0 ? tnow - t_chk : 0.0; t_chk = tnow;
            if (tnow > opt->timelimit || tnow + dit > opt->timelimit) { status = SOCP_TIME; break; }
        }
        if (*socp_stop || (socp_stop_call && *socp_stop_call)) { status = SOCP_TIME; break; }
        /* the accuracy of the solves: that of the regularized factorization while the errors are large,
         * the full one near the end */
        if (it == 0) mu0 = mu;
        S->gm_tol = err > gm_switch && mu > 1e-5 * mu0 && S->tau > 1e-3 * fmax(1.0, S->kap) ? gm_tol_early : gm_tol_late;      /* (also full accuracy on the way to an infeasibility certificate) */
        /* (19 October 2026, LP/SOCP, a test, off: the late tolerance k times the error, as inexact Newton methods do,
         *  BRISK_CONEGMREL=k. CBLIB's structural-design SOCPs at k = 1e-3: db-plane-strain-prism 33 -> 27 s, joint_FC_7
         *  173 -> 139 s; but on the 79 SOCP files and 43 CBLIB files 1.12 times the time over 5 s, qssp180old OPTIMAL ->
         *  REDUCED, nql180old 19 -> 74 iterations: those classes need the tight solves; ISSUES 217) */
        { static double gmrel = -1; if (gmrel < 0) { const char *e_ = getenv("BRISK_CONEGMREL"); gmrel = e_ ? atof(e_) : 0; }
          if (gmrel > 0 && S->gm_tol == gm_tol_late) S->gm_tol = fmax(gm_tol_late, fmin(gm_tol_early, gmrel * err)); }
        /* (26 October 2026, the SOCP developer, a test: BRISK_CONEREFMU=1 in the early phase, 2 throughout) the refinement to
         * the step's needs, as the QP method's (BRISK_QPREFMU): in the reduced system [-H A'; A 0] a solve's error in the first
         * block becomes the dual residual after the step, in the second the primal one; they need only be a tenth of the
         * present residuals, which along the path fall with mu (|r_k| / mu_k <= |r_0| / mu_0), and at least 1e-2 of the
         * tolerance; the fixed relative tolerance stays as the bound of the work */
        { static int refmu = -1; if (refmu < 0) { const char *e_ = getenv("BRISK_CONEREFMU"); refmu = e_ ? atoi(e_) : 0; }
          static _Thread_local double rp0 = 0, rd0 = 0;
          S->gm_abs = 0.0; S->tg_d = S->tg_p = S->tg_g = 0.0; S->tg_w = 0; S->tg_f = 1.0; S->tg_fl = S->tg_fd = S->tg_fp = 0.0;
          static int rend = -1; if (rend < 0) rend = getenv("BRISK_CONEREFMUEND") != NULL;   /* (test: the targets only away from the end, the error above 100 times the tolerance) */
          if (refmu == 6 && rend && !(err > 100.0 * opt->tol)) { }
          else if (refmu == 7) {      /* (both: the targets of 5 in the measure of the problem as given, times sigma as in 6) */
              double ip = 0, id = 0; for (int i = 0; i < m; i++) { const double u = rP[i] / S->Dr[i]; ip += u * u; } for (int j = 0; j < n; j++) { const double u = rD[j] / S->Ec[j]; id += u * u; }
              const double ft = 1e-2 * opt->tol * fmax(S->tau, 1e-300);
              S->tg_w = 1; S->tg_fl = ft; S->tg_fp = ft * (1.0 + nb0) / S->sb; S->tg_fd = ft * (1.0 + nc0) / S->scl;
              S->tg_p = 0.1 * sqrt(ip); S->tg_d = 0.1 * sqrt(id); S->tg_g = 0.1 * (xz + S->tau * S->kap); S->tg_f = 0.0;
          } else if (refmu == 6) {      /* (the targets of 4 times the fraction of the residuals the step aims at: after a full step the
                                  * residuals are sigma r + the solve's error, so an error of a tenth of r (4) leaves them at
                                  * (sigma + 0.1) r while mu falls to sigma mu, and their ratio to mu grows each iteration; the
                                  * target is a tenth of sigma r: the predictor (which only chooses sigma) at f = 1, the direction
                                  * and its correctors at f = sigma, the solve of [c; b] (its error times dtau) refined in full) */
              double ip = 0, id = 0; for (int i = 0; i < m; i++) if (fabs(rP[i]) > ip) ip = fabs(rP[i]); for (int j = 0; j < n; j++) if (fabs(rD[j]) > id) id = fabs(rD[j]);
              S->tg_fl = 1e-2 * opt->tol * fmax(S->tau, 1e-300);
              S->tg_p = 0.1 * ip; S->tg_d = 0.1 * id; S->tg_g = 0.1 * (xz + S->tau * S->kap); S->tg_f = 0.0;
          } else if (refmu == 5) {      /* (the targets in the measure the iteration's errors use: the 2-norms of the residuals of the problem
                                  * as given, the equilibration undone row by row (1/Dr) and column by column (1/Ec); the floor is a
                                  * hundredth of the tolerance in that measure) */
              double ip = 0, id = 0; for (int i = 0; i < m; i++) { const double u = rP[i] / S->Dr[i]; ip += u * u; } for (int j = 0; j < n; j++) { const double u = rD[j] / S->Ec[j]; id += u * u; }
              const double ft = 1e-2 * opt->tol * fmax(S->tau, 1e-300);
              S->tg_w = 1;
              S->tg_p = 0.1 * fmax(sqrt(ip), ft * (1.0 + nb0) / S->sb); S->tg_d = 0.1 * fmax(sqrt(id), ft * (1.0 + nc0) / S->scl); S->tg_g = 0.1 * fmax(xz + S->tau * S->kap, ft);
          } else if (refmu == 4) {      /* (the targets checked on the preconditioned solve's residual, kkt_solve) */
              double ip = 0, id = 0; for (int i = 0; i < m; i++) if (fabs(rP[i]) > ip) ip = fabs(rP[i]); for (int j = 0; j < n; j++) if (fabs(rD[j]) > id) id = fabs(rD[j]);
              const double fl = 1e-2 * opt->tol * fmax(S->tau, 1e-300);
              S->tg_p = 0.1 * fmax(ip, fl); S->tg_d = 0.1 * fmax(id, fl); S->tg_g = 0.1 * fmax(xz + S->tau * S->kap, fl);
          } else if (refmu > 0) {
              const double rpn = nrm2(m, rP), rdn = nrm2(n, rD);
              if (it == 0) { rp0 = rpn; rd0 = rdn; }
              if ((refmu == 2 || (refmu == 1 && S->gm_tol == gm_tol_early) || (refmu == 3 && err > 100.0 * opt->tol))) {     /* (3: away from the end, where the error is above 100 times the tolerance) */
                  const double fl = 1e-2 * opt->tol * fmax(S->tau, 1e-300) * 2.0, rel = mu0 > 0 ? mu / mu0 : 1.0;
                  static double rf_ = -1; if (rf_ < 0) { const char *e_ = getenv("BRISK_CONEREFMUF"); rf_ = e_ ? atof(e_) : 0.1; }   /* (test: the fraction of the residuals) */
                  static int nrel = -1; if (nrel < 0) nrel = getenv("BRISK_CONEREFMUNOREL") != NULL;   /* (test: without the path's relaxation) */
                  const double tp = rf_ * fmax(fmax(rpn, fl), nrel ? 0.0 : rp0 * rel), td = rf_ * fmax(fmax(rdn, fl), nrel ? 0.0 : rd0 * rel);
                  S->gm_abs = fmin(tp, td);
              }
          } }
        tq[2] += sc_time() - tq_last; tq_last = sc_time();
        /* ---- scaling and factorization */
        S->it_maxres = 0.0;
        if (S->bth && !S->ne && !S->use_dd) for (int k = S->nq - S->nr_rot; k < S->nq; k++) {      /* the rotated cones balanced */
            if (S->bth[k] != S->bth[k] || (!S->bpos[k] && !(S->rot && S->rot[k] && S->rG[k]))) continue;   /* (23 October 2026, the SOCP developer: the cones in the eigenbasis of their scaling too) */
            const double *x = S->x + S->qs[k], a = sqrt(0.5) * (x[0] + x[1]), b = sqrt(0.5) * (x[0] - x[1]);
            if (!(a > 0 && b > 0)) continue;
            const double r = a / b;
            /* (any boost is exact; it is applied only to a pair out of balance by more than 100, where the second-order form's
             * scaling cancels: on TV denoising the replaced pivots of the end phase came from that cone, 30 -> 19 iterations) */
            if (r > 100.0 || r < 0.01) {
                if (dbg_on()) socp_printf("      [rotated cone %d: a %.2e b %.2e, boosted by %.3g]\n", k, a, b, sqrt(r));
                const double th_ = 0.5 * log(r);
                cone_boost(S, k, th_);
                /* (20 October 2026, LP/SOCP: the dual residual, computed before the boost, in the boosted coordinates
                 * too (it transforms as z; the primal residual, the gap and mu are invariant); without it the direction
                 * aimed at the old residual on the pair and the dual residual jumped after each boost) */
                { static int rdb = -1; if (rdb < 0) rdb = getenv("BRISK_CONEBOOSTRD0") == NULL;
                  if (rdb) { const int j = S->qs[k]; const double ch = cosh(th_), sh = sinh(th_);
                    const double a = ch * rD[j] + sh * rD[j + 1], b = sh * rD[j] + ch * rD[j + 1]; rD[j] = a; rD[j + 1] = b; } }
            }
        }
        if (nt_update(S)) { status = SOCP_NUMERR; break; }
        /* (27 October 2026, the SOCP developer, experiments: BRISK_CONEDUMP=prefix BRISK_CONEDUMPIT=k1,k2,..: the scaled problem's A
         * (once) and the iteration's scaling (h of the linear columns, eta and wbar of the cones) written to prefix.A and
         * prefix.<k>, for a study of iterative solves of the normal equations; bench/socp_r6/pcg_study.py reads them) */
        { static const char *dp = (const char *)1; static char dit[256];
          if (dp == (const char *)1) { dp = getenv("BRISK_CONEDUMP"); const char *e_ = getenv("BRISK_CONEDUMPIT"); snprintf(dit, sizeof dit, ",%s,", e_ ? e_ : ""); }
          char key[32]; snprintf(key, sizeof key, ",%d,", it);
          if (dp && strstr(dit, key)) {
            char fn[512]; FILE *fd;
            snprintf(fn, sizeof fn, "%s.A", dp);
            if ((fd = fopen(fn, "rb")) == NULL && (fd = fopen(fn, "wb")) != NULL) {
                int hd[5] = { S->m, S->n, S->nf, S->nl, S->nq }; fwrite(hd, sizeof(int), 5, fd); fwrite(S->q, sizeof(int), (size_t)S->nq, fd);
                fwrite(S->Ap, sizeof(int), (size_t)S->n + 1, fd); fwrite(S->Ai, sizeof(int), (size_t)S->Ap[S->n], fd); fwrite(S->Ax, sizeof(double), (size_t)S->Ap[S->n], fd);
                fclose(fd);
            } else if (fd) fclose(fd);
            snprintf(fn, sizeof fn, "%s.%d", dp, it);
            if ((fd = fopen(fn, "wb")) != NULL) {
                fwrite(S->h, sizeof(double), (size_t)S->nl, fd); fwrite(S->eta, sizeof(double), (size_t)S->nq, fd); fwrite(S->wbar, sizeof(double), (size_t)S->n, fd);
                const double mu_ = mu; fwrite(&mu_, sizeof(double), 1, fd); fclose(fd);
            }
          } }
        { static int ddat = -2; if (ddat == -2) { const char *e_ = getenv("BRISK_CONEDDAT"); ddat = e_ ? atoi(e_) : -1; }   /* (test: double-double from iteration k) */
          if (ddat >= 0 && it >= ddat && !S->use_dd) S->want_dd = 1; }
        /* (22 October 2026, LP/SOCP, a test: BRISK_CONEDDCOST=f) double-double when the previous iteration's GMRES steps cost
         * more than f times one double-double factorization (estimated at 3e8 flops a second; a step at the estimated
         * solve cost tsv); ISSUES 238 */
        { static double ddc = -1; static long refp = 0; if (ddc < 0) { const char *e_ = getenv("BRISK_CONEDDCOST"); ddc = e_ ? atof(e_) : 0; }
          if (ddc > 0) {
            const double lz = (double)(S->sc ? S->slnz : S->F.lnz), tsv = lz * (S->sc ? 3e-9 : 1.5e-9) + (double)S->Ap[n] * 3e-9 + (double)(n + m) * 2e-8;
            const long g = it > 0 ? S->refine_total - refp : 0; refp = S->refine_total;
            if (S->dd_on && !S->use_dd && it >= 2 && S->cflops <= 5e8 && (double)g * tsv > ddc * S->cflops / 3e8) {
                if (dbg_on()) socp_printf("      [GMRES steps %ld of the last iteration (%.3fs estimated) against a double-double factorization (%.3fs): double-double]\n", g, g * tsv, S->cflops / 3e8);
                S->want_dd = 1; } } }
        { static double mureg_c = -1; if (mureg_c < 0) { const char *e_ = getenv("BRISK_CONEMUREG"); mureg_c = e_ ? atof(e_) : 0; }   /* (test: the regularization follows mu) */
          if (mureg_c > 0 && !S->use_dd && S->bump == 1.0) { const double r_ = fmin(1e-6, fmax(1e-10, mureg_c * mu / fmax(mu0, 1e-300))); S->sreg = r_; S->sregx = r_; S->sregf = fmax(r_, 1e-8); } }
        S->fp_rd = nrmi(n, rD);
        /* (27 October 2026, the SOCP developer, a test: BRISK_CONEREGSCALE=1) the regularization of the factored matrix made
         * independent of the relative scale of b and c: with b / f the solution's x is x / f and H is f H, and the symmetric
         * scaling diag(f^-1/2 I, f^1/2 I) maps the factored matrix back to the original one with the primal blocks'
         * regularization divided by f and the rows' multiplied by f (joint_FC_5, SOCP: b / 10 143 -> 97 s at the same
         * iterations; db-shear-wall the other way). Scale-free: the primal blocks' regularization in proportion to the
         * size of H, the rows' in inverse proportion; the size is the geometric mean of H's diagonal (the linear h, the
         * cones' eta^2, each by its dimension), 1 at the start, within [1e-4, 1e4] */
        { static int rsc = -1; if (rsc < 0) { const char *e_ = getenv("BRISK_CONEREGSCALE"); rsc = e_ ? atoi(e_) : 0; }
          if (rsc && !S->use_dd && S->bump == 1.0) {
              static _Thread_local double r0d = 0, r0x = 0, r0f = 0; if (it == 0 || r0d == 0) { r0d = S->sreg0; r0x = S->sregx; r0f = S->sregf; }
              double lg = 0.0; long cnt = 0;
              for (int i = 0; i < S->nl; i++) if (S->h[i] > 0) { lg += log(S->h[i]); cnt++; }
              for (int k = 0; k < S->nq; k++) { const double e2 = S->eta[k] * S->eta[k]; if (e2 > 0) { lg += S->q[k] * log(e2); cnt += S->q[k]; } }
              double g = cnt ? fmin(1e4, fmax(1e-4, exp(lg / (double)cnt))) : 1.0;
              if (rsc == 2 && g > 1.0) g = 1.0;      /* (2: one-sided, only towards more on the rows and less on the primal blocks) */
              S->sreg0 = r0d / g; S->sreg = S->sreg0; S->sregx = r0x * g; S->sregf = r0f * g;
              if (dbg_on()) socp_printf("      [the regularization by the size of H (%.2e): rows %.1e, cones %.1e, free %.1e]\n", g, S->sreg, S->sregx, S->sregf);
          } }
        kkt_fill(S);
        kkt_factor_reg(S);
        tq[3] += sc_time() - tq_last; tq_last = sc_time();
        /* the objective is at the tolerance and a residual is not: Newton steps on the equations alone,
         * with the complementarity products kept (dz = -H dx) and tau fixed */
        const double e_obj = fmax(fabs(e_g), e_c), e_inf = fmax(e_p, e_d);
        if (feas_on && e_inf > opt->tol && (e_obj <= opt->tol || (e_obj <= 1e-6 && e_obj <= 1e-3 * e_inf)) && nfeas < 12) {
            kkt_solve(S, rD, rP, dx, dy, S->refine_max);
            H_apply(S, dx, t2);
            for (int j = 0; j < n; j++) dz[j] = j < S->nf ? 0.0 : -t2[j];
            double al = 1e300;
            al = cone_step(S, S->x, dx, al); al = cone_step(S, S->z, dz, al);
            const double a = fmin(1.0, 0.99 * al);
            int okf = a > 1e-3;
            for (int j = 0; j < n && okf; j++) if (!isfinite(dx[j]) || !isfinite(dz[j])) okf = 0;
            if (okf) {
                memcpy(sv_x, S->x, sizeof(double) * (size_t)n); memcpy(sv_z, S->z, sizeof(double) * (size_t)n); memcpy(sv_y, S->y, sizeof(double) * (size_t)m);
                feas_pending = 1; feas_prev = e_inf;
                for (int j = 0; j < n; j++) { S->x[j] += a * dx[j]; S->z[j] += a * dz[j]; }
                for (int i = 0; i < m; i++) S->y[i] += a * dy[i];
                nfeas++;
                if (S->verbose > 1) socp_printf("%4d %+.9e %+.9e %8.1e %8.1e %8.1e %8.1e %8.1e %5.3f  equations\n", it + 1, pobj, dobj, e_p, e_d, e_g, mu, S->tau, a);
                continue;
            }
        }
        /* p = K^{-1} [c; b] */
        kkt_solve(S, S->c, S->b, px, py, S->refine_max);
        if (S->want_dd && S->dd_on && !S->use_dd) {       /* replaced pivots and a solve that did not refine: this matrix again, in double-double */
            kkt_fill(S); kkt_factor_reg(S);
            kkt_solve(S, S->c, S->b, px, py, S->refine_max);
        }
        const double denom0 = dotv(m, S->b, py) - dotv(n, S->c, px);       /* + kappa / tau */
        double sigma = 0.0, alpha = 0.0, aaff = 0.0;
        double dtau = 0, dkap = 0;
        const long ref0 = S->refine_total;
        const long sol0 = S->solves;
        for (int pass = 0; pass < 2; pass++) {
            const double etaf = 1.0 - sigma;
            if (S->tg_fl > 0) S->tg_f = pass == 0 ? 1.0 : sigma;
            const double cw = pass == 0 ? 0.0 : corrw >= 0 ? corrw : fmin(1.0, aaff);   /* (test: the weight of the second-order term) */
            /* r3 = W (lambda \ d_s);  d_s = sigma mu e - lambda o lambda - (W dx_a) o (W^-1 dz_a) */
            for (int i = 0; i < S->nf; i++) r3[i] = 0.0;
            for (int i = S->nf; i < S->nf + S->nl; i++) {
                double d = sigma * mu - S->x[i] * S->z[i];
                if (pass) d -= cw * dx[i] * dz[i];
                r3[i] = d / S->x[i];
            }
            for (int k = 0; k < S->nq; k++) {
                const int d = S->q[k], s = S->qs[k]; const double *l = S->lam + s;
                double *dd = ds + s;
                double ll = 0; for (int i = 0; i < d; i++) ll += l[i] * l[i];
                dd[0] = sigma * mu - ll;
                for (int i = 1; i < d; i++) dd[i] = -2.0 * l[0] * l[i];
                if (pass) {
                    W_apply(S, k, 0, dx + s, t1 + s); W_apply(S, k, 1, dz + s, t2 + s);
                    double uv = 0; for (int i = 0; i < d; i++) uv += t1[s + i] * t2[s + i];
                    dd[0] -= cw * uv;
                    for (int i = 1; i < d; i++) dd[i] -= cw * (t1[s] * t2[s + i] + t2[s] * t1[s + i]);
                }
                /* p = lambda \ dd */
                double det = l[0] * l[0], ld = l[0] * dd[0];
                for (int i = 1; i < d; i++) { det -= l[i] * l[i]; ld -= l[i] * dd[i]; }
                        double *pp = t1 + s;
                pp[0] = ld / det;
                for (int i = 1; i < d; i++) pp[i] = (dd[i] - pp[0] * l[i]) / l[0];
                W_apply(S, k, 0, pp, r3 + s);
            }
            double r5 = sigma * mu - S->tau * S->kap;
            if (pass) r5 -= cw * dtau * dkap;
            /* K q = [eta rD - r3 ; eta rP] */
            for (int j = 0; j < n; j++) rhsx[j] = etaf * rD[j] - r3[j];
            for (int i = 0; i < m; i++) rhsy[i] = etaf * rP[i];
            kkt_solve(S, rhsx, rhsy, qx, qy, pass == 0 ? S->ref_pred : S->refine_max);
            const double r4 = -etaf * rG;
            dtau = (r4 + r5 / S->tau - dotv(m, S->b, qy) + dotv(n, S->c, qx)) / (denom0 + S->kap / S->tau);
            for (int j = 0; j < n; j++) dx[j] = qx[j] + dtau * px[j];
            for (int i = 0; i < m; i++) dy[i] = qy[i] + dtau * py[i];
            dkap = (r5 - S->kap * dtau) / S->tau;
            /* dz = r3 - H dx  (free part: no z). H dx of a cone at extreme scaling is computed with an
             * error of eps lambda_max(H) |dx|, and dx = q + dtau p is a sum: the direction then misses the
             * dual equation dz = eta rD + c dtau - A'dy by that much (near an infeasibility certificate
             * the dual residual grew by a factor 20 per iteration). When the miss is more than the
             * reduction asked of the dual residual, and more than a tenth of its tolerance, the
             * direction is refined against it: one more solve with the miss as right-hand side. */
            H_apply(S, dx, t2);
            for (int j = 0; j < n; j++) dz[j] = j < S->nf ? 0.0 : r3[j] - t2[j];
            dual_refine(S, opt->tol, etaf, rD, dtau, byv, cx, ncs, dx, dy, dz, t2);
            /* step length */
            double al = 1e300;
            al = cone_step(S, S->x, dx, al); al = cone_step(S, S->z, dz, al);
            if (dtau < 0) al = fmin(al, -S->tau / dtau);
            if (dkap < 0) al = fmin(al, -S->kap / dkap);
            if (pass == 0) {
                const double aa = fmin(1.0, al);
                aaff = aa;
                if (dbg_x0) {     /* (27 October 2026, debug) the balance at the full affine step's point, the linear part, the negative parts left out */
                    double a_ = 0, b_ = 0; for (int i = S->nf; i < S->nf + S->nl; i++) { a_ += dbg_z0[i] * fmax(0.0, S->x[i] + dx[i]); b_ += dbg_x0[i] * fmax(0.0, S->z[i] + dz[i]); }
                    socp_printf("      [the affine step's balance %.3e (its step %.3f)]\n", b_ > 0 ? a_ / b_ : 0.0, aa);
                }
                sigma = sigp == 3.0 ? (1.0 - aa) * (1.0 - aa) * (1.0 - aa) : pow(1.0 - aa, sigp);
                /* (27 October 2026, the SOCP developer, a test: BRISK_CONESIGMU=cap) Mehrotra's own rule, as the LP method has it:
                 * sigma = (mu_aff / mu)^3 from the complementarity at the end of the predictor's step, capped at cap; the
                 * shortcut (1 - alpha_aff)^3 ignores how much the step itself reduces the products (lp_l1svm_8000x80, LP, by the
                 * SOCP solver: sigma 0.2-0.4 where the LP method's rule gives 0.04-0.1, 27 against 19 iterations) */
                static double sigmu = -1; if (sigmu < 0) { const char *e_ = getenv("BRISK_CONESIGMU"); sigmu = e_ ? atof(e_) : 0.0; }
                if (sigmu > 0) {
                    double xza = 0.0;
                    for (int j = S->nf; j < n; j++) xza += (S->x[j] + aa * dx[j]) * (S->z[j] + aa * dz[j]);
                    xza += (S->tau + aa * dtau) * (S->kap + aa * dkap);
                    const double r_ = fmax(0.0, xza / nu) / mu;
                    sigma = fmin(sigmu, r_ * r_ * r_);
                }
                if (sigma < sigmin) sigma = sigmin;
                if (sigma < 1e-12) sigma = 1e-12;
                if (arcq) { memcpy(axa, dx, sizeof(double) * (size_t)n); memcpy(aza, dz, sizeof(double) * (size_t)n); atau = dtau; akap = dkap; }
            } else alpha = fmin(1.0, step_frac * al);
        }
        int arc_taken = 0;
        if (arcq == 2) { memcpy(alx, dx, sizeof(double) * (size_t)n); memcpy(alz, dz, sizeof(double) * (size_t)n); memcpy(aly, dy, sizeof(double) * (size_t)m); altau = dtau; alkap = dkap; }
        if (arcq == 1 && alpha < arct && !S->use_dd) {
            double *tv = ctv;
            for (int i = 0; i < S->nf; i++) tv[i] = 0.0;
            for (int i = S->nf; i < S->nf + S->nl; i++) tv[i] = -2.0 * axa[i] * aza[i];
            for (int k = 0; k < S->nq; k++) {
                const int d = S->q[k], s = S->qs[k];
                W_apply(S, k, 0, axa + s, t1 + s); W_apply(S, k, 1, aza + s, t2 + s);
                double uv = 0; for (int i = 0; i < d; i++) uv += t1[s + i] * t2[s + i];
                tv[s] = -2.0 * uv;
                for (int i = 1; i < d; i++) tv[s + i] = -2.0 * (t1[s] * t2[s + i] + t2[s] * t1[s + i]);
            }
            const double ttk = -2.0 * atau * akap;
            DirCtx C = { S, px, py, denom0, r3, t1, t2, rhsx, rhsy, qx, qy };
            double dt2, dk2;
            corr_solve(&C, tv, ttk, dxa, dya, dza, &dt2, &dk2, S->refine_max);     /* d2 */
            for (int j = 0; j < n; j++) { dxs[j] = dx[j] - 0.5 * dxa[j]; dzs[j] = dz[j] - 0.5 * dza[j]; }   /* d1 = d_line - d2 / 2 */
            for (int i = 0; i < m; i++) ay1[i] = dy[i] - 0.5 * dya[i];
            const double dt1 = dtau - 0.5 * dt2, dk1 = dkap - 0.5 * dk2;
            int fin = 1; for (int j = 0; j < n && fin; j++) if (!isfinite(dxa[j]) || !isfinite(dza[j])) fin = 0;
            if (fin) {
                double lo = 0, hi = M_PI / 2;
                if (arc_inside(S, hi, dxs, dzs, dt1, dk1, dxa, dza, dt2, dk2)) lo = hi;
                else for (int b = 0; b < 40; b++) { const double mid = 0.5 * (lo + hi); if (arc_inside(S, mid, dxs, dzs, dt1, dk1, dxa, dza, dt2, dk2)) lo = mid; else hi = mid; }
                const double aa = step_frac * lo;
                if (sin(aa) > alpha) {
                    const double sa = sin(aa), ca = 1.0 - cos(aa);
                    for (int j = 0; j < n; j++) { dx[j] = sa * dxs[j] + ca * dxa[j]; dz[j] = sa * dzs[j] + ca * dza[j]; }
                    for (int i = 0; i < m; i++) dy[i] = sa * ay1[i] + ca * dya[i];
                    dtau = sa * dt1 + ca * dt2; dkap = sa * dk1 + ca * dk2;
                    if (dbg_on()) socp_printf("      [arc: the line's step %.3f, the arc's angle %.3f (sin %.3f)]\n", alpha, aa, sa);
                    alpha = 1.0; arc_taken = 1; narc++;
                }
            }
        }
        /* centrality correctors (Gondzio): at a longer trial step the complementarity products (the
         * eigenvalues of the scaled Jordan products for the cones) are moved into [bmin, bmax] x the
         * target; the correction is kept when it lengthens the step */
        int ncor = 0;
        for (int kc_ = 0; kc_ < mcc_now && alpha < 0.999; kc_++) {
            DirCtx C = { S, px, py, denom0, r3, t1, t2, rhsx, rhsy, qx, qy };
            const double at = fmin(1.0, alpha + mcc_da), mut = sigma * mu;
            const double lo = mcc_bmin * mut, hi = mcc_bmax * mut;
            double *tv = ctv;                     /* the correction term */
            for (int i = 0; i < S->nf; i++) tv[i] = 0.0;
            for (int i = S->nf; i < S->nf + S->nl; i++) {
                const double v = (S->x[i] + at * dx[i]) * (S->z[i] + at * dz[i]);
                double t = v < lo ? lo - v : v > hi ? hi - v : 0.0;
                if (t < -hi) t = -hi;
                tv[i] = t;
            }
            for (int k = 0; k < S->nq; k++) {
                const int d = S->q[k], s0 = S->qs[k];
                for (int i = 0; i < d; i++) { dxs[s0 + i] = S->x[s0 + i] + at * dx[s0 + i]; dzs[s0 + i] = S->z[s0 + i] + at * dz[s0 + i]; }
                W_apply(S, k, 0, dxs + s0, t1 + s0); W_apply(S, k, 1, dzs + s0, t2 + s0);
                double w0 = 0; for (int i = 0; i < d; i++) w0 += t1[s0 + i] * t2[s0 + i];
                double wn = 0;
                for (int i = 1; i < d; i++) { tv[s0 + i] = t1[s0] * t2[s0 + i] + t2[s0] * t1[s0 + i]; wn += tv[s0 + i] * tv[s0 + i]; }
                wn = sqrt(wn);
                const double l1 = w0 + wn, l2 = w0 - wn;
                double c1 = l1 < lo ? lo - l1 : l1 > hi ? hi - l1 : 0.0, c2 = l2 < lo ? lo - l2 : l2 > hi ? hi - l2 : 0.0;
                if (c1 < -hi) c1 = -hi;
                if (c2 < -hi) c2 = -hi;
                const double f = wn > 0 ? 0.5 * (c1 - c2) / wn : 0.0;
                for (int i = 1; i < d; i++) tv[s0 + i] *= f;
                tv[s0] = 0.5 * (c1 + c2);
            }
            double ttk;
            { const double v = (S->tau + at * dtau) * (S->kap + at * dkap); ttk = v < lo ? lo - v : v > hi ? hi - v : 0.0; if (ttk < -hi) ttk = -hi; }
            double tc_, kc2;
            corr_solve(&C, tv, ttk, dxa, dya, dza, &tc_, &kc2, mcc_ref);
            for (int j = 0; j < n; j++) { dxa[j] += dx[j]; dza[j] += dz[j]; }
            const double dt2 = dtau + tc_, dk2 = dkap + kc2;
            double a2 = 1e300;
            a2 = cone_step(S, S->x, dxa, a2); a2 = cone_step(S, S->z, dza, a2);
            if (dt2 < 0) a2 = fmin(a2, -S->tau / dt2);
            if (dk2 < 0) a2 = fmin(a2, -S->kap / dk2);
            const double al2 = fmin(1.0, step_frac * a2);
            if (!(al2 >= alpha + mcc_acc * mcc_da)) break;
            memcpy(dx, dxa, sizeof(double) * (size_t)n); memcpy(dz, dza, sizeof(double) * (size_t)n);
            for (int i = 0; i < m; i++) dy[i] += dya[i];
            dtau = dt2; dkap = dk2; alpha = al2; ncor++;
        }
        S->ncor_total += ncor;
        /* (20 October 2026, LP/SOCP, a test: BRISK_CONENBH=beta, a wide neighbourhood of the central path: the step is
         * shortened (by 0.9 at a time, at most eight times) while the smallest complementarity product at the new point
         * (linear pairs, cones' eigenvalues of the scaled product, tau kappa) is below beta times the new mu; db-shear-wall's
         * endgame squeezed one cone to 1 % of its boundary distance at each step, which blocked the next) */
        static double nbh = -1; if (nbh < 0) { const char *e_ = getenv("BRISK_CONENBH"); nbh = e_ ? atof(e_) : 0.0; }
        if (nbh > 0) {
            for (int tr = 0; tr < 8; tr++) {
                double mn = 1e300, tot = 0.0;
                for (int i = S->nf; i < S->nf + S->nl; i++) { const double v = (S->x[i] + alpha * dx[i]) * (S->z[i] + alpha * dz[i]); tot += v; if (v < mn) mn = v; }
                for (int k = 0; k < S->nq; k++) {
                    const int d = S->q[k], s0 = S->qs[k];
                    for (int i = 0; i < d; i++) { dxs[s0 + i] = S->x[s0 + i] + alpha * dx[s0 + i]; dzs[s0 + i] = S->z[s0 + i] + alpha * dz[s0 + i]; }
                    W_apply(S, k, 0, dxs + s0, t1 + s0); W_apply(S, k, 1, dzs + s0, t2 + s0);
                    double w0 = 0; for (int i = 0; i < d; i++) w0 += t1[s0 + i] * t2[s0 + i];
                    double wn = 0; for (int i = 1; i < d; i++) { const double v = t1[s0] * t2[s0 + i] + t2[s0] * t1[s0 + i]; wn += v * v; }
                    wn = sqrt(wn); tot += w0; if (w0 - wn < mn) mn = w0 - wn;
                }
                { const double v = (S->tau + alpha * dtau) * (S->kap + alpha * dkap); tot += v; if (v < mn) mn = v; }
                const double mup = tot / nu;
                if (mn >= nbh * mup) break;
                if (dbg_on()) socp_printf("      [neighbourhood: the smallest product %.1e of mu %.1e at step %.3f: shortened]\n", mn, mup, alpha);
                alpha *= 0.9;
            }
        }
        if (arcq == 2 && alpha < arct && !S->use_dd) {        /* (the arc from the line before the correctors, judged against the corrected step) */
            double *tv = ctv;
            for (int i = 0; i < S->nf; i++) tv[i] = 0.0;
            for (int i = S->nf; i < S->nf + S->nl; i++) tv[i] = -2.0 * axa[i] * aza[i];
            for (int k = 0; k < S->nq; k++) {
                const int d = S->q[k], s = S->qs[k];
                W_apply(S, k, 0, axa + s, t1 + s); W_apply(S, k, 1, aza + s, t2 + s);
                double uv = 0; for (int i = 0; i < d; i++) uv += t1[s + i] * t2[s + i];
                tv[s] = -2.0 * uv;
                for (int i = 1; i < d; i++) tv[s + i] = -2.0 * (t1[s] * t2[s + i] + t2[s] * t1[s + i]);
            }
            const double ttk = -2.0 * atau * akap;
            DirCtx C = { S, px, py, denom0, r3, t1, t2, rhsx, rhsy, qx, qy };
            double dt2, dk2;
            corr_solve(&C, tv, ttk, dxa, dya, dza, &dt2, &dk2, S->refine_max);
            for (int j = 0; j < n; j++) { dxs[j] = alx[j] - 0.5 * dxa[j]; dzs[j] = alz[j] - 0.5 * dza[j]; }
            for (int i = 0; i < m; i++) ay1[i] = aly[i] - 0.5 * dya[i];
            const double dt1 = altau - 0.5 * dt2, dk1 = alkap - 0.5 * dk2;
            int fin = 1; for (int j = 0; j < n && fin; j++) if (!isfinite(dxa[j]) || !isfinite(dza[j])) fin = 0;
            if (fin) {
                double lo = 0, hi = M_PI / 2;
                if (arc_inside(S, hi, dxs, dzs, dt1, dk1, dxa, dza, dt2, dk2)) lo = hi;
                else for (int b = 0; b < 40; b++) { const double mid = 0.5 * (lo + hi); if (arc_inside(S, mid, dxs, dzs, dt1, dk1, dxa, dza, dt2, dk2)) lo = mid; else hi = mid; }
                const double aa = step_frac * lo;
                if (sin(aa) > alpha) {
                    const double sa = sin(aa), ca = 1.0 - cos(aa);
                    for (int j = 0; j < n; j++) { dx[j] = sa * dxs[j] + ca * dxa[j]; dz[j] = sa * dzs[j] + ca * dza[j]; }
                    for (int i = 0; i < m; i++) dy[i] = sa * ay1[i] + ca * dya[i];
                    dtau = sa * dt1 + ca * dt2; dkap = sa * dk1 + ca * dk2;
                    if (dbg_on()) socp_printf("      [arc: the corrected line's step %.3f, the arc's angle %.3f (sin %.3f)]\n", alpha, aa, sa);
                    alpha = 1.0; arc_taken = 1; narc++;
                }
            }
        }
        /* the correctors' directions were added: the same refinement of the sum, and its step length */
        if (!arc_taken && dual_refine(S, opt->tol, 1.0 - sigma, rD, dtau, byv, cx, ncs, dx, dy, dz, t2)) {
            double a2 = 1e300;
            a2 = cone_step(S, S->x, dx, a2); a2 = cone_step(S, S->z, dz, a2);
            if (dtau < 0) a2 = fmin(a2, -S->tau / dtau);
            if (dkap < 0) a2 = fmin(a2, -S->kap / dkap);
            alpha = fmin(alpha, fmin(1.0, step_frac * a2));
        }
        if (dbg_on()) {      /* (diagnostics: the step to the boundary of each part) */
            double axl = 1e300, azl = 1e300, axq = 1e300, azq = 1e300; int ixl = -1, izl = -1;
            for (int i = S->nf; i < S->nf + S->nl; i++) { if (dx[i] < 0 && -S->x[i] / dx[i] < axl) { axl = -S->x[i] / dx[i]; ixl = i; } if (dz[i] < 0 && -S->z[i] / dz[i] < azl) { azl = -S->z[i] / dz[i]; izl = i; } }
            int kxb = -1, kzb = -1;
            for (int k = 0; k < S->nq; k++) { const double a1 = soc_step(S->q[k], S->x + S->qs[k], dx + S->qs[k], axq), a2 = soc_step(S->q[k], S->z + S->qs[k], dz + S->qs[k], azq); if (a1 < axq) { axq = a1; kxb = k; } if (a2 < azq) { azq = a2; kzb = k; } }
            if (kxb >= 0 || kzb >= 0) socp_printf("      [cones blocking: x cone %d (dimension %d), z cone %d (dimension %d)]\n", kxb, kxb >= 0 ? S->q[kxb] : 0, kzb, kzb >= 0 ? S->q[kzb] : 0);
            const double at = dtau < 0 ? -S->tau / dtau : 1e300, ak = dkap < 0 ? -S->kap / dkap : 1e300;
            int nsmall = 0; double pmin = 1e300, pmax = 0, lmin = 1e300, lmax = 0;
            for (int i = S->nf; i < S->nf + S->nl; i++) { const double p = S->x[i] * S->z[i]; if (p < 1e-2 * mu) nsmall++; pmin = fmin(pmin, p); pmax = fmax(pmax, p); }
            int kmax = -1, kmin = -1;
            for (int k = 0; k < S->nq; k++) { const double *l = S->lam + S->qs[k]; double ln = 0; for (int i = 1; i < S->q[k]; i++) ln += l[i] * l[i]; ln = sqrt(ln);
                if ((l[0] - ln) * (l[0] - ln) < lmin) { lmin = (l[0] - ln) * (l[0] - ln); kmin = k; } if ((l[0] + ln) * (l[0] + ln) > lmax) { lmax = (l[0] + ln) * (l[0] + ln); kmax = k; } }
            if (kmax >= 0) socp_printf("      [cones: the largest eigenvalue in cone %d (dimension %d%s), the smallest in cone %d (dimension %d)]\n", kmax, S->q[kmax], S->rot && S->rot[kmax] ? ", rotated" : "", kmin, S->q[kmin]);
            socp_printf("      [products / mu: linear %.1e .. %.1e, cones (lambda eigenvalues squared) %.1e .. %.1e, tau kap %.1e; sigma %.2f]\n", pmin / mu, pmax / mu, lmin / mu, lmax / mu, S->tau * S->kap / mu, sigma);
            socp_printf("      [steps to the boundary: lin x %.3f (%d: x %.1e z %.1e) lin z %.3f (%d) cone x %.3f cone z %.3f tau %.3f kap %.3f; %d linear products below mu/100]\n",
                        axl, ixl, ixl >= 0 ? S->x[ixl] : 0, ixl >= 0 ? S->z[ixl] : 0, azl, izl, axq, azq, at, ak, nsmall);
        }
        {   /* (18 October 2026, LP/SOCP: a direction that is not finite is not taken: the point stays, the iteration
             * ends as a numerical failure; a branch-flow relaxation's last double-double step had tau = NaN, and the
             * attempt then ended on an earlier point) */
            int fin = isfinite(alpha) && isfinite(dtau) && isfinite(dkap);
            for (int j = 0; j < n && fin; j++) if (!isfinite(dx[j]) || !isfinite(dz[j])) fin = 0;
            for (int i = 0; i < m && fin; i++) if (!isfinite(dy[i])) fin = 0;
            if (!fin) {
                /* the factored matrix's regularization raised, as for a pivot of the wrong sign (GMRES solves the system
                 * without it), and the iteration done again from the same point; at most twice in a solve (a cap per step: the iterate drifted,
                 * the raised level spoils the preconditioner, 70-100 GMRES steps) */
                /* (22 October 2026, LP/SOCP) first the factorization in double-double when its cap after such a direction
                 * allows it (dd_force: up to four times the base cap) */
                /* (23 October 2026, the SOCP developer: with the column factorization too, which the double-double
                 * factorization is; small problems went straight to the raised regularization and the second attempt) */
                if (S->dd_on && !S->use_dd && !S->dd_force && S->cflops <= 4.0 * 5e8 && !getenv("BRISK_CONENODDFORCE")) {
                    if (dbg_on()) socp_printf("      [the direction is not finite: not taken; the factorization in double-double]\n");
                    S->dd_force = 1; S->want_dd = 1;
                    continue;
                }
                const int retry = nf_retry < 2 && S->bump_on && S->bump < 1e4;
                if (dbg_on()) socp_printf("      [the direction is not finite (alpha %g, dtau %g, dkap %g, denom %g): not taken%s]\n", alpha, dtau, dkap, denom0, retry ? "; the regularization raised" : "");
                if (retry) {
                    if (!nf_restore) { nf_sv[0] = S->sreg; nf_sv[1] = S->sregx; nf_sv[2] = S->sregf; nf_sv[3] = S->bump; }
                    nf_restore = 1;
                    nf_retry++; S->bump *= 100.0; S->nbump++;
                    S->sreg = S->sreg0 * S->bump; S->sregx = S->sreg0 * S->bump; S->sregf = fmax(1e-6, 1e-8 * S->bump);
                    continue;
                }
                status = SOCP_NUMERR; break;
            }
        }
        if (nf_restore) { S->sreg = nf_sv[0]; S->sregx = nf_sv[1]; S->sregf = nf_sv[2]; S->bump = nf_sv[3]; nf_restore = 0; }   /* (the raise for that iteration only) */
        for (int j = 0; j < n; j++) { S->x[j] += alpha * dx[j]; S->z[j] += alpha * dz[j]; }
        if (S->fpr) { if (!S->fp_dxp) S->fp_dxp = (double *)calloc((size_t)S->nf + 1, sizeof(double)); for (int j = 0; j < S->nf; j++) S->fp_dxp[j] = alpha * dx[j]; }
        for (int i = 0; i < m; i++) S->y[i] += alpha * dy[i];
        S->tau += alpha * dtau; S->kap += alpha * dkap;
        /* the step length comes from a quadratic per cone: in a large cone its rounding can leave the
         * new point outside; the step is then shortened */
        for (int bt = 0; bt < 20 && !cone_interior(S); bt++) {
            const double back = 0.2 * alpha;
            for (int j = 0; j < n; j++) { S->x[j] -= back * dx[j]; S->z[j] -= back * dz[j]; }
            for (int i = 0; i < m; i++) S->y[i] -= back * dy[i];
            S->tau -= back * dtau; S->kap -= back * dkap;
            alpha -= back;
        }
        /* the normal equations lose accuracy late on degenerate problems (the preconditioned
         * iteration needs many steps): from then on the full system */
        { static int dropit = -1; if (dropit < 0) { const char *e_ = getenv("BRISK_CONENEDROPTEST"); dropit = e_ ? atoi(e_) : 0; }   /* (a test of the fallback: leave at that iteration) */
          /* (26 October 2026, the SOCP developer) the count rule (more than ne_k refinement steps a solve) weighs no cost: on
           * 2013_wbNRL (SOCP) 20 steps for 6 solves left a system of 3.8e8 flops for one of 4e10 (28 s of analysis and 4 s a
           * factorization, against 0.25 s for the extra steps). The rule by the solver's cost model (the rates of kkt_model):
           * leave when the iteration's steps beyond one per solve cost more than a factorization of the full system (twice
           * that when its analysis is not done yet). BRISK_CONENECOST=0: the count rule */
          static int necost = -1; if (necost < 0) { const char *e_ = getenv("BRISK_CONENECOST"); necost = e_ ? atoi(e_) : 0; }   /* (a test, as BRISK_CONENEREG: BRISK_CONENECOST=1) */
          int leave_ne;
          if ((necost || S->ne_tinykept) && S->ne) {      /* (by cost also once the normal equations are kept past tiny pivots: ne_factor) */
              const double ex = (double)(S->refine_total - ref0) - (double)(S->solves - sol0);
              const double cstep = 2.0 * (double)S->ne_lnz * 1.5e-9 + 2.0 * (double)S->Ap[n] * 3e-9 + (double)(n + m) * 2e-8;
              const double tfull = S->cflops / 6e9 * (S->kkt_defer ? 2.0 : 1.0);
              leave_ne = ex * cstep > tfull;
          } else leave_ne = S->ne && S->refine_total - ref0 > ne_k * (double)(S->solves - sol0);
          if (S->ne && (leave_ne || (dropit > 0 && it + 1 == dropit))) {
              if (dbg_on()) socp_printf("      [the normal equations left at iteration %d: %ld refinement steps for %ld solves]\n", it + 1, S->refine_total - ref0, S->solves - sol0);
              ne_free(S); S->ne_ok = 0; S->ne_left = it + 1; } }
        /* (23 October 2026, the SOCP developer) the small cones (at most SOCP_DENSE_MAX entries) into the eigenbasis of their
         * scaling when the solves need it: the refinement took at least two GMRES steps a solve in this iteration and the
         * preconditioned solves' residual lay mostly (more than half of its square) in those cones. A cone's d x d block of H
         * has eigenvalues eta^2 (w0 + |w1|)^2 and eta^2 / (w0 + |w1|)^2; factored by scalar pivots the small one is lost to
         * cancellation once they are far apart, and the factorization stops being a preconditioner there; in the eigenbasis
         * the block is diagonal with the exact eigenvalues (ISSUES 259). From the start it costs the union of each cone's
         * column patterns (chainsing-50000-2, SOCP: nnz(L) 1.10 -> 1.50 million, 2.8 -> 3.7 s at the same 13 iterations), so
         * it waits for that sign; the full system is set up again (its pattern, order and analysis), once.
         * BRISK_CONEROT=1: from the start, 0: never */
        /* the iteration's linear algebra in the solver's model: a factorization and its solves, the correction steps
         * included (counted, not timed, so that the path stays the same from run to run) */
        double c_this; { double tf, tsv; kkt_model(S, &tf, &tsv); c_this = tf + (double)(S->solves - sol0 + S->refine_total - ref0) * tsv; }
        /* the iteration after the switch cost more than the one before it: the eigenbasis's fill costs more than the
         * correction steps it saves (socp_snl_20000, SOCP: nnz(L) 2.85 -> 5.83 million, the steps per iteration only 6 -> 4);
         * back to the original basis, for good */
        if (S->rot_switch == it && S->rot_mode == 1 && !S->use_dd && c_this > c_sw0) {
            const double ts = sc_time();
            kkt_teardown(S); S->rot_mode = 0; S->kkt_defer = 0; S->rot_back = it + 1;
            const int rc_ = kkt_setup(S);
            S->t_setup += sc_time() - ts;
            if (S->verbose > 1) socp_printf("      [the small cones back in their original basis (the iteration %.3g against %.3g before the eigenbasis, in the cost model; %.3fs)]\n", c_this, c_sw0, sc_time() - ts);
            if (rc_) { status = SOCP_NUMERR; break; }
        }
        if (S->rot_mode == 2 && !S->rot_switch && !S->anyrot && !S->ne && !S->use_dd
            && S->solves - sol0 > 0 && S->refine_total - ref0 >= 2 * (S->solves - sol0) && S->it_rs2 > 0.5 * S->it_rt2) {
            const long lnz0 = S->F.lnz; const double cf0 = S->cflops, sv = (double)(S->solves - sol0), gs = (double)(S->refine_total - ref0) / sv;
            const double ts = sc_time();
            S->rot_switch = it + 1; S->rot_mode = 1; kkt_teardown(S); S->kkt_defer = 0; c_sw0 = c_this;
            const int rc_ = kkt_setup(S);
            S->t_setup += sc_time() - ts;
            if (getenv("BRISK_ROTDBG")) socp_printf("      [the switch: the iteration %.3g in the cost model; solves %.0f, steps per solve %.2f; nnz(L) %ld -> %ld, flops %.3g -> %.3g]\n", c_this, sv, gs, lnz0, S->F.lnz, cf0, S->cflops);
            if (S->verbose > 1) socp_printf("      [the small cones in the eigenbasis of their scaling from here (the solves' residual %.0f %% in them; %.3fs)]\n", 100.0 * S->it_rs2 / fmax(S->it_rt2, 1e-300), sc_time() - ts);
            if (rc_) { status = SOCP_NUMERR; break; }
        }
        S->it_rs2 = S->it_rt2 = 0;
        if (S->verbose > 1) socp_printf("%4d %+.9e %+.9e %8.1e %8.1e %8.1e %8.1e %8.1e %5.3f %6.0e %3ld%s\n", it + 1, pobj, dobj, e_p, e_d, e_g, mu, S->tau, alpha, sigma, S->refine_total - ref0, S->F.ndyn ? (S->F.dd ? " dd*" : " *") : S->F.dd ? " dd" : "");
        if (dbg_on()) { static double tprev = -1; const double tn = sc_time(); socp_printf("      [iteration %d: %.3fs (factorizations %.3fs, solves %.3fs, products %.3fs so far; %ld solves, %ld with the factor)]\n", it + 1, tprev < 0 ? 0.0 : tn - tprev, S->t_factor, S->t_solve, S->t_mul, S->solves, S->nback); tprev = tn; }
        if (dbg_on()) socp_printf("      [pivots replaced %d, last solve residual %.1e]\n", S->F.ndyn, S->last_res);
        /* double-double with replaced pivots and no step: the matrix is singular at this
         * regularization; two more levels before giving up */
        if (alpha < 1e-3 && S->use_dd && S->F.ndyn > 0 && ddbump < 2) { ddbump++; S->sreg *= 100.0; S->sregx *= 100.0; continue; }
        /* end phase: the iterate is within 10x the tolerance and a solve of this iteration has failed
         * outright (GMRES residual above 1e-3: the double factorization of a matrix with pivots from
         * 1e-13 to 1e13 is no preconditioner any more; nql180 ends on a knife edge at 1e-8 here, and
         * the second attempt costs the whole solve again). The next factorizations are double-double
         * when one is affordable (work <= 5e9: about 3 s on nql180); once per solve. */
        if (!S->use_dd && S->dd_on && !S->want_dd && !S->dd_polish && S->it_maxres > 1e-3 && best <= 10.0 * opt->tol && S->cflops <= 5e9 && !getenv("BRISK_CONENODDEND")) {
            S->dd_polish = 1; S->want_dd = 1;
            if (S->verbose > 1) socp_printf("      [solve residual %.1e within 10x the tolerance: double-double from here]\n", S->it_maxres);
            if (alpha < 1e-3) { nsmall = 0; continue; }
        }
        /* (23 October 2026) the solves of two iterations in a row failed outright (GMRES residual above 1e-3): the double
         * factorization no longer gives directions, whatever the error; double-double from here under the same work
         * limit (nql180: from iteration 17 the residuals are 1e-2 and the steps 0.2-0.5 for ten iterations, while the
         * complementarity, 1.8e-7, keeps the rule above from firing: 28 -> 18 iterations; the other 157 SOCP files of the
         * sets unchanged). BRISK_CONEDDFAIL=k: after k iterations, 0: off */
        { static int ddfail = -1; if (ddfail < 0) { const char *e_ = getenv("BRISK_CONEDDFAIL"); ddfail = e_ ? atoi(e_) : 2; }
          S->nfailit = S->it_maxres > 1e-3 ? S->nfailit + 1 : 0;
          if (ddfail > 0 && S->nfailit >= ddfail && !S->use_dd && S->dd_on && !S->want_dd && !S->dd_polish && S->cflops <= 5e9) {
              S->dd_polish = 1; S->want_dd = 1;
              if (S->verbose > 1) socp_printf("      [the solves of %d iterations in a row failed (residual %.1e): double-double from here]\n", S->nfailit, S->it_maxres);
          } }
        nsmall = alpha < 1e-3 ? nsmall + 1 : 0;
        if (!(alpha > 1e-10) || nsmall >= 2 || !(S->tau > 0) || !(S->tau == S->tau)) { status = SOCP_NUMERR; break; }
    }
    if (tq_last >= 0) tq[4] += sc_time() - tq_last;
    if (dbg_on()) socp_printf("  [the iterations' time by part: residuals %.3f, errors on the problem as given %.3f, rays %.3f, scaling and factorization %.3f, directions and steps %.3f s]\n", tq[0], tq[1], tq[2], tq[3], tq[4]);
    /* ---- result: the point (the best one when the iteration did not end at the tolerance), rotated cones mapped back */
    R->x = (double *)xcalloc((size_t)n, sizeof(double)); R->y = (double *)xcalloc((size_t)m + 1, sizeof(double)); R->z = (double *)xcalloc((size_t)n, sizeof(double));
    if (status == SOCP_PINF || status == SOCP_DINF) {
        /* the certificate, normalised: b'y = 1 (primal infeasible) or c'x = -1 (dual infeasible) */
        for (int j = 0; j < n; j++) { R->x[j] = S->Ec[j] * S->x[j] * S->sb; R->z[j] = S->z[j] * S->scl / S->Ec[j]; }
        cone_unboost(S, R->x, R->z);
        for (int i = 0; i < m; i++) R->y[i] = S->Dr[i] * S->y[i] * S->scl;
        if (rgam) rot_ungamma(P->nq, S->nq, S->qs, rgam, rrho, rrow, R->x, R->y, R->z);
        const double v = status == SOCP_PINF ? dotv(m, b0, R->y) : -dotv(n, c0, R->x);
        if (v > 0) { for (int j = 0; j < n; j++) { R->x[j] = status == SOCP_DINF ? R->x[j] / v : 0.0; R->z[j] = status == SOCP_PINF ? R->z[j] / v : 0.0; } for (int i = 0; i < m; i++) R->y[i] = status == SOCP_PINF ? R->y[i] / v : 0.0; }
        R->pobj = status == SOCP_DINF ? -1.0 : 0.0; R->dobj = status == SOCP_PINF ? 1.0 : 0.0;
    } else {
        memcpy(R->x, bx, sizeof(double) * (size_t)n); memcpy(R->y, by, sizeof(double) * (size_t)m); memcpy(R->z, bz, sizeof(double) * (size_t)n);
        R->err[0] = best_e[0]; R->err[1] = best_e[1]; R->err[2] = best_e[2]; R->err[3] = best_e[3];
        if (status != SOCP_OPTIMAL && best <= 1e-6 && status != SOCP_TIME && status != SOCP_CUTOFF) status = SOCP_REDUCED;
        if (status == SOCP_TIME && best <= opt->tol) status = SOCP_OPTIMAL;
    }
    for (int k = P->nq; k < S->nq; k++) {
        const int j = S->qs[k];
        double a = R->x[j], b_ = R->x[j + 1]; R->x[j] = rs * (a + b_); R->x[j + 1] = rs * (a - b_);
        a = R->z[j]; b_ = R->z[j + 1]; R->z[j] = rs * (a + b_); R->z[j + 1] = rs * (a - b_);
        if (R->cx) { a = R->cx[j]; b_ = R->cx[j + 1]; R->cx[j] = rs * (a + b_); R->cx[j + 1] = rs * (a - b_); a = R->cz[j]; b_ = R->cz[j + 1]; R->cz[j] = rs * (a + b_); R->cz[j + 1] = rs * (a - b_); }
    }
    free(rgam); free(rrho); free(rrow); free(rsd); free(rae); rgam = NULL;
    R->status = status; R->iters = it; R->time = sc_time() - t0; R->maxerr = best; R->lnz = S->kkt_defer ? S->ne_lnz : S->F.lnz;
    const int att = opt->attempt % 10, by_elim = opt->attempt >= 10;      /* attempt + 10: called by the substitution presolve, which prints the final status */
    const int att_line = att == 2 || (att == 1 && (status == SOCP_REDUCED || status == SOCP_NUMERR));   /* the retry wrapper prints the final status */
    if (S->verbose > 0) {
        if (att_line || !by_elim) {
            if (att_line) socp_printf("attempt %d%s: %s", att, att == 2 ? " (shorter steps)" : "", socp_status_line(status));
            else socp_printf("%sstatus: %s", g_stat_prefix, socp_status_line(status));
            if (status != SOCP_PINF && status != SOCP_DINF) socp_printf(" (max rel error %.1e)", best);
            socp_printf("   iterations: %d   time: %.3fs\n", it, R->time);
        }
        if (dbg_on() && S->nd > 0) socp_printf("  [the dense block's products: %ld D x, %ld D'y, %ld both; %.3fs]\n", g_dn_calls[0], g_dn_calls[1], g_dn_calls[2], g_dn_time);
        if (S->verbose > 1 && S->kkt_defer) socp_printf("  time: setup %.3f, factorizations %.3f, %ld solves %.3f + products %.3f (the normal equations alone: the full system not analysed)\n", S->t_setup, S->t_factor, S->nback, S->t_solve, S->t_mul);
        else if (S->verbose > 1) socp_printf("  time: setup %.3f, factorizations %.3f, %ld solves %.3f (%.3f with their permutations) + products %.3f (%s factorization, nnz(L) = %ld%s; %.2e flops by columns)\n", S->t_setup, S->t_factor, S->nback, S->t_solve, S->t_prec, S->t_mul, S->sc ? "supernodal" : "column", S->sc ? S->slnz : S->F.lnz, S->nfallback ? "; some by columns" : "", S->cflops);
        { extern void schol_ftime_print(void); if (getenv("BRISK_FTIME")) schol_ftime_print(); }   /* (the sparse factorization's split, as in the SDP solver) */
        if (dbg_on()) socp_printf("  [factorization / solve: estimated %.1f, measured %.1f]\n", S->rr_est, (S->t_factor / fmax(S->nfact, 1)) / fmax((S->t_solve + S->t_mul) / fmax(S->nback, 1), 1e-12));
        if (S->verbose > 1 && S->ncor_total) socp_printf("  %ld centrality correctors\n", S->ncor_total);
        if (S->verbose > 1 && narc) socp_printf("  %ld steps on the arc\n", narc);
        if (S->verbose > 1 && S->ne_nfact) socp_printf("  %ld factorizations of the normal equations (nnz(L) = %ld, %d tiny pivots%s)\n", S->ne_nfact, S->ne_lnz, S->ne_tiny, S->ne_dense_msg);
        if (S->verbose > 1 && S->nbump) socp_printf("  regularization of the factored matrix raised to %.0e\n", 1e-8 * S->bump);
        if (S->nbump) SNOTE("the regularization of the factored matrix was raised to %.0e", 1e-8 * S->bump);
        if (S->rot_back) SNOTE("the small cones factored in the eigenbasis of their scaling in iteration %d only (its larger factor cost more than the steps it saved)", S->rot_switch);
        else if (S->rot_switch) SNOTE("the small cones factored in the eigenbasis of their scaling from iteration %d", S->rot_switch);
        if (S->verbose > 1 && S->ndd) socp_printf("  %d factorizations in double-double\n", S->ndd);
        if (S->ndd) SNOTE("%d of %d factorizations in double-double arithmetic", S->ndd, S->nfact);
        if (S->verbose > 1 && S->dz_count) socp_printf("  %ld refinements of the direction against the dual equation\n", S->dz_count);
        if (S->verbose > 1 && S->rc_used) socp_printf("  %ld solves started from the previous Krylov space (mean gain %.1f digits)\n", S->rc_used, S->rc_gain / (double)S->rc_used / log(10.0));
        if (!att_line && !by_elim && status != SOCP_PINF && status != SOCP_DINF) socp_printf("  primal obj c'x = %+.12e   dual obj b'y = %+.12e\n  errors: pinf %.1e  dinf %.1e  gap %.1e  x'z %.1e\n", R->pobj, R->dobj, R->err[0], R->err[1], R->err[2], R->err[3]);
    }
    free(px); free(py); free(qx); free(qy); free(dx); free(dy); free(dz); free(rP); free(rD); free(r3); free(rhsx); free(rhsy); free(ds); free(t1); free(t2);
    free(dbg_x0); free(dbg_z0); free(sv_x); free(sv_y); free(sv_z); free(dxa); free(dya); free(dza); free(ctv); free(dxs); free(dzs); free(axa); free(aza); free(ay1); free(alx); free(alz); free(aly);
    free(xo); free(yo); free(zo); free(bx); free(by); free(bz); free(A0x); free(b0); free(c0);
    free(S->fpr); free(S->fpr_si); free(S->fp_dxp); free(S->wdz); free(S->wdx); free(S->wdy); free(S->wdy2);
    socp_free(S);
    return status;
}

void socp_result_free(SocpRes *R) { free(R->x); free(R->y); free(R->z); free(R->cx); free(R->cy); free(R->cz); R->x = R->y = R->z = NULL; R->cx = R->cy = R->cz = NULL; }


/* ---- presolve: free variables eliminated by substitution ----
 * A free variable u with a_iu != 0 in row i is determined by that row:  x_u = (b_i - sum_{j != u} a_ij x_j) / a_iu.
 * Substituting it removes the row and the column: the other rows k with u get row_k - (a_ku/a_iu) row_i,
 * the costs get c_j - (c_u/a_iu) a_ij. Postsolve: x_u from the formula, y_i = c_u/a_iu - sum_k (a_ku/a_iu) y_k,
 * in the reverse order of the eliminations; the slack z of the other variables is unchanged.
 * Only eliminations without fill to speak of and with a pivot that dominates its row are done: a
 * short pivot row (at most ELIM_ROWMAX entries), (len(row) - 1)(len(col) - 1) <= len(row) + len(col),
 * |pivot| >= 0.5 max |row| and >= 0.01 max |col| (0.1 before 23 October 2026). (Eliminating as much as possible was tried: on
 * nql180old the reduced problem was much harder.) */
#define ELIM_ROWMAX 4
/* (28 October 2026, the LP developer) v - d with a cancellation to the rounding level of its two terms taken as an exact
 * zero, as the simplex method's presolve does (simplex_pre.c, cost_sub): the substitutions' cost updates left
 * -0.0307400974 + 0.0307400974 = 1.7e-18 on a free variable of neos-4984115-canche (LP, MIPLIB 2017 collection), its
 * rows went, and the presolve below declared the LP dual infeasible (optimal, -1.0414620716e7, for Gurobi and the simplex
 * method). The right-hand sides' updates likewise (a row left empty with a residue would read as infeasible) */
static inline double elim_sub(double v, double d) { const double r = v - d; return fabs(r) <= 1e-12 * fmax(fabs(v), fabs(d)) ? 0.0 : r; }
typedef struct {
    int m, n, nf;
    int *rlen, *rcap; int **rcol; double **rval; unsigned char *ralive;
    int *fcnt, *flen, *fcap; int **flist;
    unsigned char *cgone;
    double *b, *c;
    long nnz;
    int ne, ecap; int *eu, *ei, *elen, *emn; int **ecol; double **eval; double *epiv, *ebi, *ecu; int **emk; double **emv;
    unsigned long long *hk; int *hp; size_t hmask, hcnt;    /* the index (row, column) -> position in the row */
    unsigned char *rh;                                      /* the row is in the index (else: as read, sorted) */
    double *esr, *esc;                                      /* the row and column factors of the pivot tests */
} Elim;
/* 18 October 2026 (LP/SOCP): a row that a substitution updates enters an open-addressing index (linear probing,
 * deletion by backward shift) that gives the position of an entry, and is updated in place (unsorted from then on),
 * so that a substitution costs O(len(pivot row)) per row it updates, after the row's first update. Rows never
 * updated stay as read (sorted; binary search). Before, each update merged the pivot row into the whole sorted row: with long rows holding many free
 * variables (one cone of dimension 50,001 tied to its free copy, with 50 dense rows) every substitution rewrote
 * 50 rows of length n, O(n^2) in all (426 s). The entries, their values and the reduced problem are the same. */
#define EH_EMPTY (~0ULL)
static inline size_t eh_slot(const Elim *E, unsigned long long key) { return (size_t)((key * 0x9E3779B97F4A7C15ULL) >> 20) & E->hmask; }
static inline unsigned long long eh_key(int i, int col) { return ((unsigned long long)(unsigned)i << 32) | (unsigned)col; }
static int row_find(const Elim *E, int i, int col) {
    if (!E->rh[i]) {
        int lo = 0, hi = E->rlen[i] - 1; const int *rc = E->rcol[i];
        while (lo <= hi) { const int mid = (lo + hi) >> 1; if (rc[mid] == col) return mid; if (rc[mid] < col) lo = mid + 1; else hi = mid - 1; }
        return -1;
    }
    const unsigned long long key = eh_key(i, col);
    for (size_t h = eh_slot(E, key); ; h = (h + 1) & E->hmask) { if (E->hk[h] == key) return E->hp[h]; if (E->hk[h] == EH_EMPTY) return -1; }
}
static void eh_grow(Elim *E);
static void eh_set(Elim *E, int i, int col, int pos) {          /* insert or update */
    if (2 * (E->hcnt + 1) > E->hmask + 1) eh_grow(E);
    const unsigned long long key = eh_key(i, col);
    size_t h = eh_slot(E, key);
    while (E->hk[h] != EH_EMPTY && E->hk[h] != key) h = (h + 1) & E->hmask;
    if (E->hk[h] == EH_EMPTY) { E->hk[h] = key; E->hcnt++; }
    E->hp[h] = pos;
}
static void eh_grow(Elim *E) {
    const size_t oc = E->hmask + 1; unsigned long long *ok = E->hk; int *op = E->hp;
    size_t c = 1024; while (c < 4 * (E->hcnt + 1)) c <<= 1;
    E->hk = (unsigned long long *)malloc(sizeof(unsigned long long) * c); E->hp = (int *)malloc(sizeof(int) * c);
    for (size_t h = 0; h < c; h++) E->hk[h] = EH_EMPTY;
    E->hmask = c - 1; E->hcnt = 0;
    for (size_t h = 0; h < oc && ok; h++) if (ok[h] != EH_EMPTY) eh_set(E, (int)(ok[h] >> 32), (int)(unsigned)(ok[h] & 0xffffffffULL), op[h]);
    free(ok); free(op);
}
static void eh_del(Elim *E, int i, int col) {
    const unsigned long long key = eh_key(i, col);
    size_t h = eh_slot(E, key);
    while (E->hk[h] != key) { if (E->hk[h] == EH_EMPTY) return; h = (h + 1) & E->hmask; }
    for (size_t g = (h + 1) & E->hmask; E->hk[g] != EH_EMPTY; g = (g + 1) & E->hmask) {   /* backward shift */
        const size_t want = eh_slot(E, E->hk[g]);
        if (((g - want) & E->hmask) >= ((g - h) & E->hmask)) { E->hk[h] = E->hk[g]; E->hp[h] = E->hp[g]; h = g; }
    }
    E->hk[h] = EH_EMPTY; E->hcnt--;
}
/* remove the entry at position t of row k: the last entry takes its place */
static void row_del(Elim *E, int k, int t) {
    const int last = --E->rlen[k];
    eh_del(E, k, E->rcol[k][t]);
    if (t != last) { E->rcol[k][t] = E->rcol[k][last]; E->rval[k][t] = E->rval[k][last]; eh_set(E, k, E->rcol[k][t], t); }
}
static void flist_add(Elim *E, int u, int row) {
    if (E->flen[u] == E->fcap[u]) { E->fcap[u] = 2 * E->fcap[u] + 4; E->flist[u] = (int *)realloc(E->flist[u], sizeof(int) * (size_t)E->fcap[u]); }
    E->flist[u][E->flen[u]++] = row;
}
static int elim_run(Elim *E, const SocpProb *P) {
    const int m = P->m, n = P->n, nf = P->nf, nl = P->nl;
    E->m = m; E->n = n; E->nf = nf;
    E->rlen = (int *)calloc((size_t)m + 1, sizeof(int)); E->rcap = (int *)calloc((size_t)m + 1, sizeof(int));
    E->rcol = (int **)calloc((size_t)m + 1, sizeof(int *)); E->rval = (double **)calloc((size_t)m + 1, sizeof(double *));
    E->ralive = (unsigned char *)malloc((size_t)m + 1); memset(E->ralive, 1, (size_t)m + 1);
    E->cgone = (unsigned char *)calloc((size_t)n + 1, 1);
    E->b = (double *)malloc(sizeof(double) * ((size_t)m + 1)); memcpy(E->b, P->b, sizeof(double) * (size_t)m);
    E->c = (double *)malloc(sizeof(double) * ((size_t)n + 1)); memcpy(E->c, P->c, sizeof(double) * (size_t)n);
    for (int p = 0; p < P->Ap[n]; p++) E->rlen[P->Ai[p]]++;
    /* (19 October 2026, LP/SOCP) the long free copies first, by columns: a free u in more than 64 rows (which the
     * substitution by rows below skips) with a row i = {u, v}, v not free and in no other row (a cone's entry tied to a
     * free variable), is substituted by moving its column onto v (no fill), in O(nnz(column)), before the rows are
     * built; by rows it would cost a hash access per entry, 1.3 s for a group lasso's 3,000 coefficients in 1,500 dense
     * rows. Recorded as the row route records an elimination, so the postsolve is the same. The shorter columns stay
     * with the row route and its order: done by columns too, the order changed and a powered-descent SOCP (36 -> 78
     * iterations) and an MPC problem (17 -> 22) became harder. */
    /* (27 October 2026, the SOCP developer; a test, BRISK_CONEELIMSCALE=1, not kept: CBLIB's db-plate-yield-line (SOCP) then kept
     * 3,187 free variables, 38 -> 91 iterations at reduced accuracy, db-shear-wall 62 -> 85; the free variables in one row are
     * substituted without the tests instead, below) the pivot tests below compare entries of different rows
     * and columns, so they are made on the entries scaled by row and column factors computed first (alternate passes that divide
     * each row and each column by the geometric mean of its largest and smallest entries, as threshold pivoting in sparse LU
     * codes is preceded by a scaling): the substitutions then do not depend on how the rows and the variables were scaled in
     * the file. The matrix and the substitutions themselves are not scaled. The independent benchmark's copies with each
     * variable times 10^u, u in [-3, 3]: a control SOCP kept 195 of its 3,848 free variables (the row test), 9 -> 242 iterations */
    static int elsc = -1; if (elsc < 0) elsc = getenv("BRISK_CONEELIMSCALE") ? atoi(getenv("BRISK_CONEELIMSCALE")) : 0;
    double *esr = (double *)malloc(sizeof(double) * ((size_t)m + 1)), *esc = (double *)malloc(sizeof(double) * ((size_t)n + 1));
    for (int i = 0; i < m; i++) esr[i] = 1.0;
    for (int j = 0; j < n; j++) esc[j] = 1.0;
    if (elsc) {
        double *hi = (double *)malloc(sizeof(double) * ((size_t)m + 1)), *lo = (double *)malloc(sizeof(double) * ((size_t)m + 1));
        for (int pass = 0; pass < 8; pass++) {
            for (int j = 0; j < n; j++) {
                double h = 0, l = INFINITY;
                for (int p = P->Ap[j]; p < P->Ap[j + 1]; p++) { const double v = fabs(P->Ax[p]) * esr[P->Ai[p]]; if (v == 0.0) continue; if (v > h) h = v; if (v < l) l = v; }
                if (h > 0) esc[j] = 1.0 / sqrt(h * l);
            }
            for (int i = 0; i < m; i++) { hi[i] = 0.0; lo[i] = INFINITY; }
            for (int j = 0; j < n; j++) for (int p = P->Ap[j]; p < P->Ap[j + 1]; p++) {
                const double v = fabs(P->Ax[p]) * esc[j]; if (v == 0.0) continue;
                const int i = P->Ai[p]; if (v > hi[i]) hi[i] = v; if (v < lo[i]) lo[i] = v;
            }
            for (int i = 0; i < m; i++) if (hi[i] > 0) esr[i] = 1.0 / sqrt(hi[i] * lo[i]);
        }
        free(hi); free(lo);
    }
    E->esr = esr; E->esc = esc;
    int *uof = (int *)malloc(sizeof(int) * ((size_t)n + 1));
    for (int j = 0; j < n; j++) uof[j] = -1;
    int *cpiv = (int *)malloc(sizeof(int) * ((size_t)n + 1));     /* the position of the pivot in column u */
    /* (19 October 2026, LP/SOCP) the entries a cone keeps nonnegative: the first of a second-order cone, the first two of a
     * rotated one. A nonnegative variable u tied to one of them by u = alpha v, alpha > 0 (a row a_iu u + a_iv v = 0,
     * a_iu a_iv < 0), has its bound implied by the cone: it is substituted as a free copy, its slack z_u = 0 (the bound
     * is not active in a solution of the problem without it, which has the same solutions). TV denoising ties each
     * cone's t to a nonnegative t: 22,500 rows of 67,502. */
    unsigned char *impl = (unsigned char *)calloc((size_t)n + 1, 1);
    { int s_ = nf + nl; for (int k = 0; k < P->nq; k++) { impl[s_] = 1; s_ += P->q[k]; } for (int k = 0; k < P->nr; k++) { impl[s_] = 1; impl[s_ + 1] = 1; s_ += P->r[k]; } }
    static int implon = -1; if (implon < 0) implon = getenv("BRISK_CONENOIMPL") == NULL;
    {
        int *rfirst = (int *)malloc(sizeof(int) * ((size_t)m + 1)), *rother = (int *)malloc(sizeof(int) * ((size_t)m + 1));     /* a row of two: its columns */
        for (int i = 0; i < m; i++) rfirst[i] = rother[i] = -1;
        for (int j = 0; j < n; j++) for (int p = P->Ap[j]; p < P->Ap[j + 1]; p++) { const int i = P->Ai[p]; if (E->rlen[i] == 2) { if (rfirst[i] < 0) rfirst[i] = j; else rother[i] = j; } }
        static int fcmin = -1; if (fcmin < 0) { const char *e_ = getenv("BRISK_CONEFCOPY"); fcmin = e_ ? atoi(e_) : 64; }
        for (int u = 0; u < nf + (implon ? nl : 0); u++) {
            cpiv[u] = -1;
            /* (the shorter columns by rows, as before, unless the substitution is a mere renaming: a row a_iu u + a_iv v = 0
             * moves u's entries to v; with |a_iu| = |a_iv| the same values, so the substitutions by rows decide as they
             * would have; a cone of 50,001 entries tied to its free copies in 50 dense rows) */
            const int shortcol = P->Ap[u + 1] - P->Ap[u] <= fcmin;
            double amax = 0; for (int p = P->Ap[u]; p < P->Ap[u + 1]; p++) amax = fmax(amax, fabs(P->Ax[p]) * esr[P->Ai[p]]);
            int best = -1;
            for (int p = P->Ap[u]; p < P->Ap[u + 1]; p++) {
                const int i = P->Ai[p];
                if (E->rlen[i] != 2 || !E->ralive[i]) continue;
                const int v = rfirst[i] == u ? rother[i] : rfirst[i];
                if (v < nf || v == u || P->Ap[v + 1] - P->Ap[v] != 1 || uof[v] >= 0) continue;
                const double av = P->Ax[P->Ap[v]], au = P->Ax[p];
                /* (27 October 2026, the SOCP developer, a test: BRISK_CONEBOXELIM=1) a nonnegative u whose bound a row of two implies
                 * with any nonnegative v (a linear column or a cone's first entry): u = b_i / a_iu - (a_iv / a_iu) v >= 0 for all
                 * v >= 0 when a_iu a_iv < 0 and b_i a_iu >= 0. A lower bound l >= 0 written as its own row u - s = l (beam7, SOCP:
                 * 53,075 variables with both bounds as rows, 106,150 rows where one per box suffices: u = l + s, its upper-bound
                 * row u + t = h becomes s + t = h - l) */
                static int boxel = -1; if (boxel < 0) { const char *e_ = getenv("BRISK_CONEBOXELIM"); boxel = e_ ? atoi(e_) : 0; }
                const int vnn = impl[v] || (boxel && v >= nf && v < nf + nl);
                if (u >= nf && !(vnn && au * av < 0 && (P->b[i] == 0.0 || (boxel && P->b[i] * au > 0.0)))) continue;     /* (a nonnegative u: implied by the cone or by the row) */
                /* (20 October 2026, LP/SOCP: a scaled renaming too, b_i = 0 with any a_iu, a_iv: v is in no other row, so u's
                 * column moves onto v times -a_iv/a_iu with no fill and no sum, an exact rescaling, and the pivot tests,
                 * which guard the sums of a substitution, have nothing to guard. An MPC problem that ties its cone's
                 * entries to sqrt(0.1) times its inputs: 4,800 rows -> 3,000, 18 -> 10 iterations; ISSUES 222.
                 * BRISK_CONENOSCALEREN=1: only |a_iu| = |a_iv|, as before) */
                static int scren = -1; if (scren < 0) scren = getenv("BRISK_CONENOSCALEREN") == NULL;
                const int ren = P->b[i] == 0.0 && (scren || fabs(av) == fabs(au));
                if (shortcol && !ren && !(boxel && u >= nf)) continue;      /* (the row route below takes free columns only) */
                const double sau = fabs(au) * esr[i] * esc[u], sav = fabs(av) * esr[i] * esc[v];     /* (the scaled entries) */
                if (!ren && (sau < 0.5 * fmax(sau, sav) || fabs(au) * esr[i] < 0.1 * amax)) continue;
                if (best < 0 || sau > fabs(P->Ax[best]) * esr[P->Ai[best]] * esc[u]) best = p;
            }
            if (best < 0) continue;
            const int i = P->Ai[best], v = rfirst[i] == u ? rother[i] : rfirst[i];
            uof[v] = u; cpiv[u] = best; E->ralive[i] = 0; E->cgone[u] = 1;
        }
        free(rfirst); free(rother);
    }
    for (int i = 0; i < m; i++) if (!E->ralive[i]) E->rlen[i] = 0;
    for (int v = 0; v < n; v++) if (uof[v] >= 0) {      /* the row lengths after the moves */
        const int u = uof[v];
        for (int p = P->Ap[u]; p < P->Ap[u + 1]; p++) if (p != cpiv[u]) E->rlen[P->Ai[p]]++;
    }
    for (int j = 0; j < n; j++) if (E->cgone[j]) for (int p = P->Ap[j]; p < P->Ap[j + 1]; p++) if (E->ralive[P->Ai[p]]) E->rlen[P->Ai[p]]--;
    for (int i = 0; i < m; i++) { E->rcap[i] = E->rlen[i] + 4; E->rcol[i] = (int *)malloc(sizeof(int) * (size_t)E->rcap[i]); E->rval[i] = (double *)malloc(sizeof(double) * (size_t)E->rcap[i]); E->rlen[i] = 0; }
    E->nnz = 0;
    for (int t = 0; t < n; t++) {       /* the rows in column order */
        if (E->cgone[t]) continue;
        if (uof[t] >= 0) {
            const int u = uof[t], pp = cpiv[u], i = P->Ai[pp]; const double au = P->Ax[pp], av = P->Ax[P->Ap[t]];
            const int nr = P->Ap[u + 1] - P->Ap[u];
            if (E->ne == E->ecap) {
                E->ecap = 2 * E->ecap + 256;
                E->eu = (int *)realloc(E->eu, sizeof(int) * (size_t)E->ecap); E->ei = (int *)realloc(E->ei, sizeof(int) * (size_t)E->ecap);
                E->elen = (int *)realloc(E->elen, sizeof(int) * (size_t)E->ecap); E->emn = (int *)realloc(E->emn, sizeof(int) * (size_t)E->ecap);
                E->ecol = (int **)realloc(E->ecol, sizeof(int *) * (size_t)E->ecap); E->eval = (double **)realloc(E->eval, sizeof(double *) * (size_t)E->ecap);
                E->epiv = (double *)realloc(E->epiv, sizeof(double) * (size_t)E->ecap); E->ebi = (double *)realloc(E->ebi, sizeof(double) * (size_t)E->ecap); E->ecu = (double *)realloc(E->ecu, sizeof(double) * (size_t)E->ecap);
                E->emk = (int **)realloc(E->emk, sizeof(int *) * (size_t)E->ecap); E->emv = (double **)realloc(E->emv, sizeof(double *) * (size_t)E->ecap);
            }
            const int e = E->ne++;
            E->eu[e] = u; E->ei[e] = i; E->elen[e] = 2; E->epiv[e] = au; E->ebi[e] = P->b[i]; E->ecu[e] = P->c[u];
            E->ecol[e] = (int *)malloc(sizeof(int) * 2); E->eval[e] = (double *)malloc(sizeof(double) * 2);
            E->ecol[e][0] = u < t ? u : t; E->eval[e][0] = u < t ? au : av; E->ecol[e][1] = u < t ? t : u; E->eval[e][1] = u < t ? av : au;
            E->emn[e] = nr - 1; E->emk[e] = (int *)malloc(sizeof(int) * (size_t)nr); E->emv[e] = (double *)malloc(sizeof(double) * (size_t)nr);
            E->c[t] = elim_sub(E->c[t], P->c[u] / au * av);
            int mq = 0;
            for (int p = P->Ap[u]; p < P->Ap[u + 1]; p++) {
                if (p == pp) continue;
                const int k = P->Ai[p]; const double mk = P->Ax[p] / au;
                E->emk[e][mq] = k; E->emv[e][mq++] = mk;
                E->b[k] = elim_sub(E->b[k], mk * P->b[i]);
                if (!E->ralive[k]) continue;
                E->rcol[k][E->rlen[k]] = t; E->rval[k][E->rlen[k]++] = -mk * av; E->nnz++;
            }
        } else for (int p = P->Ap[t]; p < P->Ap[t + 1]; p++) { const int i = P->Ai[p]; if (!E->ralive[i]) continue; E->rcol[i][E->rlen[i]] = t; E->rval[i][E->rlen[i]++] = P->Ax[p]; E->nnz++; }
    }
    free(uof); free(cpiv); free(impl);
    E->rh = (unsigned char *)calloc((size_t)m + 1, 1); E->hcnt = 0; E->hmask = 0; E->hk = NULL; E->hp = NULL; eh_grow(E);
    E->fcnt = (int *)calloc((size_t)nf + 1, sizeof(int)); E->flen = (int *)calloc((size_t)nf + 1, sizeof(int)); E->fcap = (int *)calloc((size_t)nf + 1, sizeof(int));
    E->flist = (int **)calloc((size_t)nf + 1, sizeof(int *));
    for (int u = 0; u < nf; u++) if (!E->cgone[u]) for (int p = P->Ap[u]; p < P->Ap[u + 1]; p++) if (E->ralive[P->Ai[p]]) { flist_add(E, u, P->Ai[p]); E->fcnt[u]++; }
    int *rows = NULL; double *avals = NULL; int rwcap = 0;
    int *rmark = (int *)calloc((size_t)m + 1, sizeof(int)), stamp = 0;     /* (the rows of a column, each once) */
    /* (23 October 2026) the pivot's share of its column's largest entry 0.01 (was 0.1), the threshold of sparse LU's threshold
     * pivoting (MA57's default): the substitution's multipliers stay below 100. db-plate-yield-line (SOCP): 73,314 free
     * variables were kept only by the column rule (rows x_u - x_v = b next to column entries of 12-19); with 0.01 all but one
     * go (rows 245,085 -> 171,771, as MOSEK's presolve: 173,647), 73 -> 37 s; on the 158 SOCP files 5 change, all OPTIMAL,
     * 2,682 -> 2,680 iterations. BRISK_CONEELIM=4,1,3,0.1: as before */
    int el_rowmax = ELIM_ROWMAX, el_pass = 3; double el_fill = 1.0, el_col = 0.01;
    static int elsing = -1; if (elsing < 0) elsing = getenv("BRISK_CONEELIMSING") ? atoi(getenv("BRISK_CONEELIMSING")) : 1;
    { const char *e_ = getenv("BRISK_CONEELIM"); if (e_) sscanf(e_, "%d,%lf,%d,%lf", &el_rowmax, &el_fill, &el_pass, &el_col); }   /* (test: row length, fill factor, passes, the pivot's share of its column's largest entry) */
    for (int pass = 0; pass < el_pass; pass++) {
        int done = 0;
        long whyc[8] = {0,0,0,0,0,0,0,0};
        for (int u = 0; u < nf; u++) {
            if (E->cgone[u] || E->fcnt[u] == 0) { if (!E->cgone[u]) whyc[0]++; continue; }
            if (E->flen[u] > rwcap) { rwcap = 2 * E->flen[u] + 8; rows = (int *)realloc(rows, sizeof(int) * (size_t)rwcap); avals = (double *)realloc(avals, sizeof(double) * (size_t)rwcap); }
            int nr = 0; double amax = 0;
            if (++stamp == 0x7fffffff) { memset(rmark, 0, sizeof(int) * (size_t)m); stamp = 1; }     /* (the duplicates by a mark) */
            for (int t = 0; t < E->flen[u]; t++) {
                const int i = E->flist[u][t];
                if (!E->ralive[i] || rmark[i] == stamp) continue;
                const int pos = row_find(E, i, u);
                if (pos < 0) continue;
                rmark[i] = stamp;
                rows[nr] = i; avals[nr] = E->rval[i][pos]; if (fabs(avals[nr]) * esr[i] > amax) amax = fabs(avals[nr]) * esr[i]; nr++;
                if (nr > 64) break;
            }
            if (nr <= 64) { memcpy(E->flist[u], rows, sizeof(int) * (size_t)nr); E->flen[u] = nr; E->fcnt[u] = nr; }
            if (nr == 0 || nr > 64 || !(amax > 0)) { whyc[1]++; continue; }
            int best = -1; int w_len = 0, w_col = 0, w_row = 0, w_fill = 0;
            for (int q = 0; q < nr; q++) {
                const int i = rows[q], li = E->rlen[i];
                if (li > el_rowmax) { w_len++; continue; }
                /* (27 October 2026, the SOCP developer) a free variable in one row (nr = 1) is substituted without the pivot tests:
                 * no other row receives a multiple of the pivot row, the costs change by c_j - (c_u / a_iu) a_ij, which a scaling
                 * of the variable leaves the same, and x_u is recovered from its row. The tests compare a_iu with the other
                 * entries of its row, whose sizes depend on how the other variables are scaled: the independent benchmark's
                 * copy of a control SOCP with each variable times 10^u kept 195 such variables of 3,848 (BRISK_CONEELIMSING=0:
                 * the tests as before) */
                if (nr > 1 || !elsing) {
                    if (fabs(avals[q]) * esr[i] < el_col * amax) { w_col++; continue; }
                    double rmax = 0; for (int t = 0; t < li; t++) rmax = fmax(rmax, fabs(E->rval[i][t]) * esc[E->rcol[i][t]]);
                    if (fabs(avals[q]) * esc[u] < 0.5 * rmax) { w_row++; continue; }
                }
                if ((double)(li - 1) * (nr - 1) > el_fill * (double)(li + nr)) { w_fill++; continue; }
                if (best < 0 || li < E->rlen[rows[best]] || (li == E->rlen[rows[best]] && fabs(avals[q]) * esr[i] > fabs(avals[best]) * esr[rows[best]])) best = q;
            }
            if (best < 0) { whyc[w_fill ? 5 : w_row ? 4 : w_col ? 3 : 2]++; if (getenv("BRISK_ELIMDBG") && whyc[2] + whyc[3] + whyc[4] + whyc[5] < 6) { fprintf(stderr, "  [free %d: %d rows:", u, nr); for (int q = 0; q < nr; q++) fprintf(stderr, " (len %d, a %.2g)", E->rlen[rows[q]], avals[q]); fprintf(stderr, "]\n"); } continue; }
            const int pi = rows[best], li = E->rlen[pi];
            const double piv = avals[best];
            if (E->ne == E->ecap) {
                E->ecap = 2 * E->ecap + 256;
                E->eu = (int *)realloc(E->eu, sizeof(int) * (size_t)E->ecap); E->ei = (int *)realloc(E->ei, sizeof(int) * (size_t)E->ecap);
                E->elen = (int *)realloc(E->elen, sizeof(int) * (size_t)E->ecap); E->emn = (int *)realloc(E->emn, sizeof(int) * (size_t)E->ecap);
                E->ecol = (int **)realloc(E->ecol, sizeof(int *) * (size_t)E->ecap); E->eval = (double **)realloc(E->eval, sizeof(double *) * (size_t)E->ecap);
                E->epiv = (double *)realloc(E->epiv, sizeof(double) * (size_t)E->ecap); E->ebi = (double *)realloc(E->ebi, sizeof(double) * (size_t)E->ecap); E->ecu = (double *)realloc(E->ecu, sizeof(double) * (size_t)E->ecap);
                E->emk = (int **)realloc(E->emk, sizeof(int *) * (size_t)E->ecap); E->emv = (double **)realloc(E->emv, sizeof(double *) * (size_t)E->ecap);
            }
            const int e = E->ne++;
            E->eu[e] = u; E->ei[e] = pi; E->elen[e] = li; E->ecol[e] = E->rcol[pi]; E->eval[e] = E->rval[pi]; E->epiv[e] = piv; E->ebi[e] = E->b[pi]; E->ecu[e] = E->c[u];
            E->emn[e] = nr - 1; E->emk[e] = (int *)malloc(sizeof(int) * (size_t)nr); E->emv[e] = (double *)malloc(sizeof(double) * (size_t)nr);
            if (E->rh[pi]) for (int t = 0; t < li; t++) eh_del(E, pi, E->rcol[pi][t]);
            for (int t = 1; t < li; t++) {      /* (in column order: the postsolve sums the row as before) */
                const int cc = E->rcol[pi][t]; const double vv = E->rval[pi][t]; int q = t - 1;
                while (q >= 0 && E->rcol[pi][q] > cc) { E->rcol[pi][q + 1] = E->rcol[pi][q]; E->rval[pi][q + 1] = E->rval[pi][q]; q--; }
                E->rcol[pi][q + 1] = cc; E->rval[pi][q + 1] = vv;
            }
            const int *ic = E->rcol[pi]; const double *iv = E->rval[pi];
            const double cu = E->c[u] / piv;
            if (cu != 0) for (int t = 0; t < li; t++) if (ic[t] != u) E->c[ic[t]] = elim_sub(E->c[ic[t]], cu * iv[t]);
            int mq = 0;
            for (int q = 0; q < nr; q++) {
                if (q == best) continue;
                const int k = rows[q]; const double mk = avals[q] / piv;
                E->emk[e][mq] = k; E->emv[e][mq++] = mk;
                const int lk = E->rlen[k];
                if (!E->rh[k]) { E->rh[k] = 1; for (int t = 0; t < lk; t++) eh_set(E, k, E->rcol[k][t], t); }
                if (lk + li > E->rcap[k]) { E->rcap[k] = lk + li + (lk + li) / 2 + 4; E->rcol[k] = (int *)realloc(E->rcol[k], sizeof(int) * (size_t)E->rcap[k]); E->rval[k] = (double *)realloc(E->rval[k], sizeof(double) * (size_t)E->rcap[k]); }
                for (int t = 0; t < li; t++) {
                    const int col = ic[t], pos = row_find(E, k, col);
                    if (col == u) { if (pos >= 0) row_del(E, k, pos); continue; }
                    if (pos < 0) {
                        const int w = E->rlen[k]++;
                        E->rcol[k][w] = col; E->rval[k][w] = -mk * iv[t]; eh_set(E, k, col, w);
                        if (col < nf) { E->fcnt[col]++; flist_add(E, col, k); }
                    } else {
                        const double v = E->rval[k][pos] - mk * iv[t];
                        if (v != 0) E->rval[k][pos] = v;
                        else { row_del(E, k, pos); if (col < nf) E->fcnt[col]--; }
                    }
                }
                E->nnz += E->rlen[k] - lk;
                E->b[k] = elim_sub(E->b[k], mk * E->b[pi]);
            }
            for (int t = 0; t < li; t++) if (ic[t] < nf) E->fcnt[ic[t]]--;
            E->ralive[pi] = 0; E->cgone[u] = 1; E->nnz -= li;
            E->rcol[pi] = NULL; E->rval[pi] = NULL; E->rlen[pi] = 0;
            done++;
        }
        if (getenv("BRISK_ELIMDBG")) fprintf(stderr, "  [elimination pass %d: %d done; kept: no rows %ld, rows > 64 %ld, rows too long %ld, pivot small in its column %ld, in its row %ld, fill %ld]\n", pass, done, whyc[0], whyc[1], whyc[2], whyc[3], whyc[4], whyc[5]);
        if (!done) break;
    }
    free(rows); free(avals); free(rmark);
    return E->ne;
}
static void elim_free(Elim *E) {
    for (int i = 0; i < E->m; i++) { free(E->rcol[i]); free(E->rval[i]); }
    for (int u = 0; u < E->nf; u++) free(E->flist[u]);
    for (int e = 0; e < E->ne; e++) { free(E->ecol[e]); free(E->eval[e]); free(E->emk[e]); free(E->emv[e]); }
    free(E->rlen); free(E->rcap); free(E->rcol); free(E->rval); free(E->ralive); free(E->fcnt); free(E->flen); free(E->fcap); free(E->flist); free(E->cgone); free(E->b); free(E->c);
    free(E->eu); free(E->ei); free(E->elen); free(E->emn); free(E->ecol); free(E->eval); free(E->epiv); free(E->ebi); free(E->ecu); free(E->emk); free(E->emv);
    free(E->hk); free(E->hp); free(E->rh); free(E->esr); free(E->esc);
}
/* 5.8 (test, BRISK_CONESPLIT=1): the free variables left after the substitution written as
 * differences of two nonnegative ones (columns a, -a with costs c, -c) and solved by the core
 * without free variables; x_u = x+ - x-, z_u = z+ (= -z-). */
static int core_split(const SocpProb *Q, const SocpOpts *o, SocpRes *R) {
    const int nf = Q->nf, n = Q->n, m = Q->m;
    if (nf == 0 || !getenv("BRISK_CONESPLIT")) return socp_solve_core(Q, o, R);
    const int n2 = n + nf;
    int *Ap = (int *)malloc(sizeof(int) * ((size_t)n2 + 1)); int *Ai = (int *)malloc(sizeof(int) * ((size_t)Q->Ap[n] + (size_t)Q->Ap[nf] + 1));
    double *Ax = (double *)malloc(sizeof(double) * ((size_t)Q->Ap[n] + (size_t)Q->Ap[nf] + 1)), *c2 = (double *)malloc(sizeof(double) * ((size_t)n2 + 1));
    int w = 0; Ap[0] = 0;
    for (int j = 0; j < nf; j++) {           /* x+ */
        for (int p = Q->Ap[j]; p < Q->Ap[j + 1]; p++) { Ai[w] = Q->Ai[p]; Ax[w++] = Q->Ax[p]; }
        c2[j] = Q->c[j]; Ap[j + 1] = w;
    }
    for (int j = 0; j < nf; j++) {           /* x- */
        for (int p = Q->Ap[j]; p < Q->Ap[j + 1]; p++) { Ai[w] = Q->Ai[p]; Ax[w++] = -Q->Ax[p]; }
        c2[nf + j] = -Q->c[j]; Ap[nf + j + 1] = w;
    }
    for (int j = nf; j < n; j++) {
        for (int p = Q->Ap[j]; p < Q->Ap[j + 1]; p++) { Ai[w] = Q->Ai[p]; Ax[w++] = Q->Ax[p]; }
        c2[nf + j] = Q->c[j]; Ap[nf + j + 1] = w;
    }
    SocpProb Q2 = *Q; Q2.n = n2; Q2.nf = 0; Q2.nl = Q->nl + 2 * nf; Q2.Ap = Ap; Q2.Ai = Ai; Q2.Ax = Ax; Q2.c = c2;
    if (o && o->verbose > 0) socp_printf("presolve: %d free variables split into nonnegative pairs (test)\n", nf);
    const int st = socp_solve_core(&Q2, o, R);
    if (R->x && R->z) {
        double *x = (double *)malloc(sizeof(double) * ((size_t)n + 1)), *z = (double *)malloc(sizeof(double) * ((size_t)n + 1));
        for (int j = 0; j < nf; j++) { x[j] = R->x[j] - R->x[nf + j]; z[j] = 0.5 * (R->z[j] - R->z[nf + j]); }
        for (int j = nf; j < n; j++) { x[j] = R->x[nf + j]; z[j] = R->z[nf + j]; }
        free(R->x); free(R->z); R->x = x; R->z = z;
    }
    (void)m;
    free(Ap); free(Ai); free(Ax); free(c2);
    return st;
}
/* (20 October 2026, LP/SOCP) a test, BRISK_CONEFREECONE=1: the free variables left after the substitution written as one
 * second-order cone (t, x_f), t a new variable with no rows and no cost (Andersen 2002, the approach MOSEK reports):
 * the problem is unchanged (t is unbounded above), its dual slack (z_t, z_f) in the dual cone with z_t = 0 at the
 * optimum forces z_f = 0 as for free variables; the embedding then has no free variables (no fixed regularization of
 * the free block, the normal equations open). The cone goes after the others; the columns are permuted back. */
static int core_freecone(const SocpProb *Q, const SocpOpts *o, SocpRes *R) {
    const int nf = Q->nf, n = Q->n, m = Q->m;
    static int on = -1; if (on < 0) { const char *e_ = getenv("BRISK_CONEFREECONE"); on = e_ ? atoi(e_) : 0; }
    if (nf == 0 || !on) return core_split(Q, o, R);
    const int n2 = n + 1, nnz = Q->Ap[n];
    int *Ap = (int *)malloc(sizeof(int) * ((size_t)n2 + 1)), *Ai = (int *)malloc(sizeof(int) * ((size_t)nnz + 1));
    double *Ax = (double *)malloc(sizeof(double) * ((size_t)nnz + 1)), *c2 = (double *)malloc(sizeof(double) * ((size_t)n2 + 1));
    int *q2 = (int *)malloc(sizeof(int) * ((size_t)Q->nq + 1));
    /* new order: the linear and cone columns first (as they are), then t, then the free columns */
    int w = 0; Ap[0] = 0; int j2 = 0;
    for (int j = nf; j < n; j++, j2++) { for (int p = Q->Ap[j]; p < Q->Ap[j + 1]; p++) { Ai[w] = Q->Ai[p]; Ax[w++] = Q->Ax[p]; } c2[j2] = Q->c[j]; Ap[j2 + 1] = w; }
    c2[j2] = 0.0; Ap[j2 + 1] = w; j2++;                         /* t */
    for (int j = 0; j < nf; j++, j2++) { for (int p = Q->Ap[j]; p < Q->Ap[j + 1]; p++) { Ai[w] = Q->Ai[p]; Ax[w++] = Q->Ax[p]; } c2[j2] = Q->c[j]; Ap[j2 + 1] = w; }
    for (int k = 0; k < Q->nq; k++) q2[k] = Q->q[k];
    q2[Q->nq] = nf + 1;
    /* the rotated cones come after the second-order ones in the column order: the free cone must go before them, so
     * with rotated cones present it is placed after the second-order cones and before the rotated ones */
    SocpProb Q2 = *Q; Q2.n = n2; Q2.nf = 0; Q2.nq = Q->nq + 1; Q2.q = q2; Q2.Ap = Ap; Q2.Ai = Ai; Q2.Ax = Ax; Q2.c = c2;
    int *perm = (int *)malloc(sizeof(int) * ((size_t)n2 + 1));   /* perm[j2] = the original column, or -1 for t */
    if (Q->nr > 0) {
        /* reorder so that (t, x_f) sits between the second-order cones and the rotated ones */
        int nrot = 0; for (int k = 0; k < Q->nr; k++) nrot += Q->r[k];
        const int cut = (n - nf) - nrot;                        /* columns (in the new order) before the rotated cones */
        int *Ap3 = (int *)malloc(sizeof(int) * ((size_t)n2 + 1)); int *Ai3 = (int *)malloc(sizeof(int) * ((size_t)nnz + 1));
        double *Ax3 = (double *)malloc(sizeof(double) * ((size_t)nnz + 1)), *c3 = (double *)malloc(sizeof(double) * ((size_t)n2 + 1));
        int w3 = 0, j3 = 0; Ap3[0] = 0;
        #define MOVECOL(src) do { for (int p = Ap[src]; p < Ap[(src) + 1]; p++) { Ai3[w3] = Ai[p]; Ax3[w3++] = Ax[p]; } c3[j3] = c2[src]; Ap3[++j3] = w3; } while (0)
        for (int j = 0; j < cut; j++) { perm[j3] = nf + j; MOVECOL(j); }
        for (int j = n - nf; j < n2; j++) { perm[j3] = j == n - nf ? -1 : j - (n - nf) - 1; MOVECOL(j); }
        for (int j = cut; j < n - nf; j++) { perm[j3] = nf + j; MOVECOL(j); }
        #undef MOVECOL
        free(Ap); free(Ai); free(Ax); free(c2); Ap = Ap3; Ai = Ai3; Ax = Ax3; c2 = c3;
        Q2.Ap = Ap; Q2.Ai = Ai; Q2.Ax = Ax; Q2.c = c2;
    } else {
        for (int j = 0; j < n - nf; j++) perm[j] = nf + j;
        perm[n - nf] = -1;
        for (int j = 0; j < nf; j++) perm[n - nf + 1 + j] = j;
    }
    if (o && o->verbose > 0) socp_printf("presolve: %d free variables as one second-order cone (test)\n", nf);
    const int st = socp_solve_core(&Q2, o, R);
    if (R->x && R->z) {
        double *x = (double *)malloc(sizeof(double) * ((size_t)n + 1)), *z = (double *)malloc(sizeof(double) * ((size_t)n + 1));
        for (int j = 0; j < n2; j++) if (perm[j] >= 0) { x[perm[j]] = R->x[j]; z[perm[j]] = R->z[j]; }
        for (int j = 0; j < nf; j++) z[j] = 0.0;               /* (the dual slack of a free variable) */
        free(R->x); free(R->z); R->x = x; R->z = z;
    }
    (void)m;
    free(Ap); free(Ai); free(Ax); free(c2); free(q2); free(perm);
    return st;
}
static int socp_drop_empty(const SocpProb *P, const SocpOpts *opt, SocpRes *R, int (*inner)(const SocpProb *, const SocpOpts *, SocpRes *));
static int socp_solve_elim(const SocpProb *P, const SocpOpts *opt, SocpRes *R) {
    if (P->nf == 0 || getenv("BRISK_CONENOELIM")) return socp_solve_core(P, opt, R);
    const double t0 = sc_time();
    const int m = P->m, n = P->n, nf = P->nf;
    Elim E_; Elim *E = &E_; memset(E, 0, sizeof *E);
    if (elim_run(E, P) == 0) { elim_free(E); return socp_solve_core(P, opt, R); }
    int *cmap = (int *)malloc(sizeof(int) * ((size_t)n + 1)), *rmap = (int *)malloc(sizeof(int) * ((size_t)m + 1));
    int n2 = 0, m2 = 0, nf2 = 0;
    for (int j = 0; j < n; j++) { cmap[j] = E->cgone[j] ? -1 : n2++; if (j < nf && !E->cgone[j]) nf2++; }
    for (int i = 0; i < m; i++) rmap[i] = E->ralive[i] ? m2++ : -1;
    int *Ap = (int *)calloc((size_t)n2 + 2, sizeof(int));
    for (int i = 0; i < m; i++) if (E->ralive[i]) for (int t = 0; t < E->rlen[i]; t++) Ap[cmap[E->rcol[i][t]] + 1]++;
    for (int j = 0; j < n2; j++) Ap[j + 1] += Ap[j];
    int *Ai = (int *)malloc(sizeof(int) * ((size_t)Ap[n2] + 1)), *nx = (int *)malloc(sizeof(int) * ((size_t)n2 + 1));
    double *Ax = (double *)malloc(sizeof(double) * ((size_t)Ap[n2] + 1));
    memcpy(nx, Ap, sizeof(int) * (size_t)n2);
    for (int i = 0; i < m; i++) if (E->ralive[i]) for (int t = 0; t < E->rlen[i]; t++) { const int j = cmap[E->rcol[i][t]]; Ai[nx[j]] = rmap[i]; Ax[nx[j]++] = E->rval[i][t]; }
    double *b2 = (double *)malloc(sizeof(double) * ((size_t)m2 + 1)), *c2 = (double *)malloc(sizeof(double) * ((size_t)n2 + 1));
    for (int i = 0; i < m; i++) if (rmap[i] >= 0) b2[rmap[i]] = E->b[i];
    for (int j = 0; j < n; j++) if (cmap[j] >= 0) c2[cmap[j]] = E->c[j];
    SocpProb Q = *P;
    { int nlg = 0; for (int j = nf; j < nf + P->nl; j++) if (E->cgone[j]) nlg++; Q.nl = P->nl - nlg; }      /* (nonnegative copies implied by their cones) */
    Q.m = m2; Q.n = n2; Q.Ap = Ap; Q.Ai = Ai; Q.Ax = Ax; Q.b = b2; Q.c = c2; Q.nf = nf2;
    if (opt && opt->verbose > 0) socp_printf("presolve: %d free variables eliminated by substitution (rows %d -> %d, nonzeros %d -> %d; %.2f s)\n", E->ne, m, m2, P->Ap[n], Ap[n2], sc_time() - t0);
    SNOTE("presolve: %d free variables eliminated by substitution (rows %d -> %d)", E->ne, m, m2);
    SocpOpts o2; if (opt) o2 = *opt; else socp_default_opts(&o2);
    const int vb = o2.verbose, att = o2.attempt;
    o2.attempt = att + 10;
    int st;
    if (n2 == 0) {
        /* every variable was substituted out (a system of equations in free variables): the point is
         * the postsolve alone; the core is not called (it refuses n = 0) */
        memset(R, 0, sizeof(*R));
        R->x = (double *)xcalloc(1, sizeof(double)); R->y = (double *)xcalloc((size_t)m2 + 1, sizeof(double)); R->z = (double *)xcalloc(1, sizeof(double));
        st = SOCP_OPTIMAL; R->status = st; R->iters = 0;
    } else st = socp_drop_empty(&Q, &o2, R, core_freecone);      /* (cones tied to free copies are empty only now) */
    if (R->x && R->y && R->z) {
        const int cert = st == SOCP_PINF || st == SOCP_DINF;
        double *x = (double *)calloc((size_t)n + 1, sizeof(double)), *y = (double *)calloc((size_t)m + 1, sizeof(double)), *z = (double *)calloc((size_t)n + 1, sizeof(double));
        for (int j = 0; j < n; j++) if (cmap[j] >= 0) { x[j] = R->x[cmap[j]]; z[j] = R->z[cmap[j]]; }
        for (int i = 0; i < m; i++) if (rmap[i] >= 0) y[i] = R->y[rmap[i]];
        for (int e = E->ne - 1; e >= 0; e--) {
            const int u = E->eu[e], pi = E->ei[e];
            if (st != SOCP_PINF) {
                double sx = cert ? 0.0 : E->ebi[e];
                for (int t = 0; t < E->elen[e]; t++) if (E->ecol[e][t] != u) sx -= E->eval[e][t] * x[E->ecol[e][t]];
                x[u] = sx / E->epiv[e];
            }
            if (st != SOCP_DINF) {
                double sy = cert ? 0.0 : E->ecu[e] / E->epiv[e];
                for (int t = 0; t < E->emn[e]; t++) sy -= E->emv[e][t] * y[E->emk[e][t]];
                y[pi] = sy;
            }
        }
        free(R->x); free(R->y); free(R->z); R->x = x; R->y = y; R->z = z;
        if (!cert) {
            double po = 0, dobj = 0, nb = 0, nc = 0, rp = 0, rd = 0, xz = 0;
            double *ra = (double *)malloc(sizeof(double) * ((size_t)m + 1));
            for (int i = 0; i < m; i++) { ra[i] = -P->b[i]; dobj += P->b[i] * y[i]; nb += P->b[i] * P->b[i]; }
            for (int j = 0; j < n; j++) {
                double s = 0; for (int p = P->Ap[j]; p < P->Ap[j + 1]; p++) { ra[P->Ai[p]] += P->Ax[p] * x[j]; s += P->Ax[p] * y[P->Ai[p]]; }
                const double r = P->c[j] - s - z[j];
                rd += r * r; po += P->c[j] * x[j]; nc += P->c[j] * P->c[j]; xz += x[j] * z[j];
            }
            for (int i = 0; i < m; i++) rp += ra[i] * ra[i];
            free(ra);
            R->pobj = po; R->dobj = dobj;
            R->err[0] = sqrt(rp) / (1.0 + sqrt(nb)); R->err[1] = sqrt(rd) / (1.0 + sqrt(nc));
            R->err[2] = (po - dobj) / (1.0 + fabs(po) + fabs(dobj)); R->err[3] = xz / (1.0 + fabs(po) + fabs(dobj));
            R->maxerr = fmax(fmax(R->err[0], R->err[1]), fmax(fabs(R->err[2]), R->err[3]));
        }
    }
    R->time = sc_time() - t0;
    int st2 = st;
    if (st == SOCP_OPTIMAL && R->maxerr > 10.0 * o2.tol) st2 = SOCP_REDUCED;
    R->status = st2;
    const int deferred = att == 2 || (att == 1 && (st2 == SOCP_REDUCED || st2 == SOCP_NUMERR));
    if (vb > 0 && !deferred) {
        socp_printf("%sstatus: %s", g_stat_prefix, socp_status_line(st2));
        if (st2 != SOCP_PINF && st2 != SOCP_DINF) socp_printf(" (max rel error %.1e)", R->maxerr);
        socp_printf("   iterations: %d   time: %.3fs\n", R->iters, R->time);
        if (st2 != SOCP_PINF && st2 != SOCP_DINF) socp_printf("  primal obj c'x = %+.12e   dual obj b'y = %+.12e\n  errors: pinf %.1e  dinf %.1e  gap %.1e  x'z %.1e\n", R->pobj, R->dobj, R->err[0], R->err[1], R->err[2], R->err[3]);
    }
    free(cmap); free(rmap); free(Ap); free(Ai); free(nx); free(Ax); free(b2); free(c2);
    elim_free(E);
    return st2;
}

/* ---- presolve: split free variables ----
 * A free variable written as the difference of two nonnegative ones (two linear columns a, -a with
 * costs c, -c) makes the central path ill-conditioned: both parts grow without bound while only their
 * difference is determined. Such pairs are found by hashing the columns and are solved as one free
 * variable; the two parts are recovered as the positive and the negative part. */
static unsigned long long col_hash(const SocpProb *P, int j, int *sgn) {
    const int p0 = P->Ap[j], p1 = P->Ap[j + 1];
    unsigned long long h = 1469598103934665603ULL;
    const double s = p1 > p0 ? (P->Ax[p0] < 0 ? -1.0 : 1.0) : (P->c[j] < 0 ? -1.0 : 1.0);
    *sgn = s < 0 ? -1 : 1;
    for (int p = p0; p < p1; p++) {
        double v = s * P->Ax[p]; unsigned long long u; memcpy(&u, &v, 8);
        h = (h ^ (unsigned long long)(unsigned)P->Ai[p]) * 1099511628211ULL; h = (h ^ u) * 1099511628211ULL;
    }
    { double v = s * P->c[j] + 0.0; unsigned long long u; memcpy(&u, &v, 8); h = (h ^ u) * 1099511628211ULL; }
    return h;
}
static int col_opposite(const SocpProb *P, int j, int k) {
    const int p0 = P->Ap[j], q0 = P->Ap[k], len = P->Ap[j + 1] - p0;
    if (len != P->Ap[k + 1] - q0 || P->c[j] != -P->c[k]) return 0;
    for (int t = 0; t < len; t++) if (P->Ai[p0 + t] != P->Ai[q0 + t] || P->Ax[p0 + t] != -P->Ax[q0 + t]) return 0;
    return 1;
}
typedef struct { unsigned long long h; int j; } HashCol;
static int cmp_hashcol(const void *a, const void *b) {
    const HashCol *x = (const HashCol *)a, *y = (const HashCol *)b;
    return x->h < y->h ? -1 : x->h > y->h ? 1 : x->j - y->j;
}
static int socp_solve_pre(const SocpProb *P, const SocpOpts *opt, SocpRes *R) {
    const int nl = P->nl, nf = P->nf, n = P->n;
    if (nl < 2 || getenv("BRISK_CONENOPRE")) return socp_solve_elim(P, opt, R);
    for (int j = 0; j + 1 <= n; j++) if (P->Ap[j + 1] < P->Ap[j]) return socp_solve_elim(P, opt, R);
    HashCol *hc = (HashCol *)malloc(sizeof(HashCol) * (size_t)nl);
    int *mate = (int *)malloc(sizeof(int) * (size_t)nl), *sg = (int *)malloc(sizeof(int) * (size_t)nl);
    for (int k = 0; k < nl; k++) { hc[k].j = k; hc[k].h = col_hash(P, nf + k, &sg[k]); mate[k] = -1; }
    qsort(hc, (size_t)nl, sizeof(HashCol), cmp_hashcol);
    int np = 0;
    for (int a = 0; a < nl; ) {
        int e = a + 1; while (e < nl && hc[e].h == hc[a].h) e++;
        for (int u = a; u < e; u++) {
            const int ju = hc[u].j; if (mate[ju] >= 0) continue;
            for (int v = u + 1; v < e; v++) {
                const int jv = hc[v].j;
                if (mate[jv] < 0 && sg[ju] != sg[jv] && col_opposite(P, nf + ju, nf + jv)) { mate[ju] = jv; mate[jv] = ju; np++; break; }
            }
        }
        a = e;
    }
    free(hc); free(sg);
    if (np == 0) { free(mate); return socp_solve_elim(P, opt, R); }
    /* the reduced problem: [free][merged pairs, free][other linear][cones] */
    const int n2 = n - np;
    int *map = (int *)malloc(sizeof(int) * (size_t)n);          /* column of the reduced problem; -1: the second of a pair */
    { int f = nf, l = nf + np;
      for (int j = 0; j < nf; j++) map[j] = j;
      for (int k = 0; k < nl; k++) map[nf + k] = mate[k] < 0 ? l++ : mate[k] > k ? f++ : -1;
      for (int j = nf + nl; j < n; j++) map[j] = j - np; }
    int *Ap = (int *)calloc((size_t)n2 + 1, sizeof(int)), *Ai = (int *)malloc(sizeof(int) * ((size_t)P->Ap[n] + 1));
    double *Ax = (double *)malloc(sizeof(double) * ((size_t)P->Ap[n] + 1)), *c = (double *)malloc(sizeof(double) * (size_t)n2);
    for (int j = 0; j < n; j++) if (map[j] >= 0) { Ap[map[j] + 1] = P->Ap[j + 1] - P->Ap[j]; c[map[j]] = P->c[j]; }
    for (int j = 0; j < n2; j++) Ap[j + 1] += Ap[j];
    for (int j = 0; j < n; j++) if (map[j] >= 0) {
        const int o = Ap[map[j]], p0 = P->Ap[j], len = P->Ap[j + 1] - p0;
        memcpy(Ai + o, P->Ai + p0, sizeof(int) * (size_t)len); memcpy(Ax + o, P->Ax + p0, sizeof(double) * (size_t)len);
    }
    SocpProb Q = *P;
    Q.n = n2; Q.Ap = Ap; Q.Ai = Ai; Q.Ax = Ax; Q.c = c; Q.nf = nf + np; Q.nl = nl - 2 * np;
    if (opt && opt->verbose > 0) socp_printf("presolve: %d pairs of linear variables are free variables\n", np);
    SNOTE("presolve: %d pairs of linear variables (a, -a) treated as free variables", np);
    const int st = socp_solve_elim(&Q, opt, R);
    if (R->x && R->z) {
        double *x = (double *)calloc((size_t)n, sizeof(double)), *z = (double *)calloc((size_t)n, sizeof(double));
        for (int j = 0; j < n; j++) if (map[j] >= 0) { x[j] = R->x[map[j]]; z[j] = R->z[map[j]]; }
        for (int k = 0; k < nl; k++) if (mate[k] > k) {
            const int j = nf + k, j2 = nf + mate[k];
            const double u = x[j], r = z[j];
            x[j] = u > 0 ? u : 0.0; x[j2] = u < 0 ? -u : 0.0;
            z[j] = r > 0 ? r : 0.0; z[j2] = r < 0 ? -r : 0.0;
        }
        free(R->x); free(R->z); R->x = x; R->z = z;
    }
    free(mate); free(map); free(Ap); free(Ai); free(Ax); free(c);
    return st;
}

/* ---- the solve, with a second attempt ----
 * A solve that stalls short of the tolerance (reduced accuracy, numerical difficulties) is repeated
 * with steps to 0.8 of the way to the boundary: better centred iterates reach the tolerance on badly
 * scaled problems where the default 0.99 does not (at 10-15 % more iterations, which is why it is not
 * the default). The better of the two results is returned. */
static int socp_solve_seq(const SocpProb *P, const SocpOpts *opt, SocpRes *R);
/* 17 October 2026: the SOCP solver is sequential, and runs on one thread: its sparse factorization would take the
 * OpenMP threads (the tree code of sparsechol.c), which made it slower (the SOCP applications 8.4 -> 10.8 s at
 * four threads, robustlp_400 0.82 -> 1.50 s) and its results depend on the thread count */
/* (17 October 2026: ... but the large nodes of that factorization take the threads, by fixed tiles, so the results
 * are the same at any thread count: schol_set_panel_threads; BRISK_CONEPANEL=0 off) */
void schol_set_panel_threads(int nt);
/* ---- presolve: blocks in no row (18 October 2026, LP/SOCP) ----
 * A variable or a whole cone whose columns of A are empty is decoupled from the rest: min c_k'x_k over its cone alone.
 * If c_k is in the dual cone the minimum is x_k = 0 (z_k = c_k) and the block goes; otherwise a ray d in the cone
 * with c_k'd < 0 and A d = 0 proves the dual infeasible at once (a free variable with c_j != 0, a nonnegative one
 * with c_j < 0, a cone (c0, c1) with |c1| > c0: d = (1, -c1/|c1|), a rotated cone through its map to a second-order
 * cone). 100,000 cones of dimension 4 of which about 30 % meet no row: the certificate in 17 iterations before,
 * from the presolve now; and the bounded variant loses those cones. */
static int socp_drop_empty(const SocpProb *P, const SocpOpts *opt, SocpRes *R, int (*inner)(const SocpProb *, const SocpOpts *, SocpRes *)) {
    if (getenv("BRISK_CONENOEMPTY") || P->n < 1) return inner(P, opt, R);
    const int n = P->n, nf = P->nf, nl = P->nl, nb = nf + nl + P->nq + P->nr;
    /* the blocks: free and linear variables one each, then the cones */
    int *bs = (int *)malloc(sizeof(int) * ((size_t)nb + 1)); int *bk = (int *)malloc(sizeof(int) * ((size_t)nb + 1));   /* start, kind (0 free, 1 linear, 2 cone, 3 rotated) */
    { int k = 0, s = 0;
      for (int j = 0; j < nf; j++) { bs[k] = s++; bk[k++] = 0; }
      for (int j = 0; j < nl; j++) { bs[k] = s++; bk[k++] = 1; }
      for (int q = 0; q < P->nq; q++) { bs[k] = s; s += P->q[q]; bk[k++] = 2; }
      for (int q = 0; q < P->nr; q++) { bs[k] = s; s += P->r[q]; bk[k++] = 3; }
      bs[nb] = s;
      if (s != n) { free(bs); free(bk); return inner(P, opt, R); } }
    char *gone = (char *)calloc((size_t)nb + 1, 1); int ngone = 0, ray = -1; double raybest = 0;
    for (int k = 0; k < nb; k++) {
        const int j0 = bs[k], j1 = bs[k + 1];
        if (P->Ap[j1] != P->Ap[j0]) continue;           /* (some column of the block has entries) */
        const double *c = P->c + j0; const int d = j1 - j0;
        double viol;     /* > 0: c_k is outside the dual cone (by that much, relative) */
        if (bk[k] == 0) viol = fabs(c[0]);
        else if (bk[k] == 1) viol = -c[0];
        else {
            double c0 = c[0], s2 = 0; int i1 = 1;
            if (bk[k] == 3) { c0 = (c[0] + c[1]) * sqrt(0.5); const double c1 = (c[0] - c[1]) * sqrt(0.5); s2 = c1 * c1; i1 = 2; }
            for (int i = i1; i < d; i++) s2 += c[i] * c[i];
            viol = sqrt(s2) - c0;
        }
        if (viol > 0) { if (viol > raybest) { raybest = viol; ray = k; } }
        else { gone[k] = 1; ngone++; }
    }
    if (ray >= 0) {
        /* the certificate: d in the cone of block `ray`, normalised to c'd = -1 */
        const int j0 = bs[ray], d = bs[ray + 1] - j0; const double *c = P->c + j0;
        memset(R, 0, sizeof(*R));
        R->x = (double *)xcalloc((size_t)n, sizeof(double)); R->y = (double *)xcalloc((size_t)P->m + 1, sizeof(double)); R->z = (double *)xcalloc((size_t)n, sizeof(double));
        double *x = R->x + j0;
        if (bk[ray] == 0) x[0] = c[0] > 0 ? -1.0 : 1.0;
        else if (bk[ray] == 1) x[0] = 1.0;
        else if (bk[ray] == 2) { double nr = 0; for (int i = 1; i < d; i++) nr += c[i] * c[i]; nr = sqrt(nr); x[0] = 1.0; for (int i = 1; i < d; i++) x[i] = -c[i] / nr; }
        else {
            const double rs = sqrt(0.5), t0 = (c[0] + c[1]) * rs, t1 = (c[0] - c[1]) * rs;
            double nr = t1 * t1; for (int i = 2; i < d; i++) nr += c[i] * c[i]; nr = sqrt(nr);
            const double u0 = 1.0, u1 = -t1 / nr; (void)t0;
            x[0] = (u0 + u1) * rs; x[1] = (u0 - u1) * rs; for (int i = 2; i < d; i++) x[i] = -c[i] / nr;
        }
        double cd = 0; for (int i = 0; i < d; i++) cd += c[i] * x[i];
        if (cd < 0) for (int i = 0; i < d; i++) x[i] /= -cd;
        R->status = SOCP_DINF; R->pobj = -1.0; R->dobj = 0.0; R->iters = 0;
        if (opt && opt->verbose > 0) {          /* (inside the substitution presolve, attempt >= 10, it prints the status) */
            socp_printf("presolve: a %s in no row of A with its cost outside the dual cone: dual infeasible (an improving ray)\n", bk[ray] == 0 ? "free variable" : bk[ray] == 1 ? "nonnegative variable" : "cone");
            if (opt->attempt < 10) socp_printf("status: %s   iterations: 0   time: 0.000s\n", socp_status_str(SOCP_DINF));
        }
        SNOTE("presolve: a block in no row of A has its cost outside the dual cone: dual infeasible, the certificate is that ray");
        free(bs); free(bk); free(gone);
        return SOCP_DINF;
    }
    if (ngone == 0 || ngone == nb) { free(bs); free(bk); free(gone); return inner(P, opt, R); }
    /* the reduced problem without the blocks at x = 0 */
    int *map = (int *)malloc(sizeof(int) * ((size_t)n + 1)); int n2 = 0, nf2 = 0, nl2 = 0, nq2 = 0, nr2 = 0;
    int *q2 = (int *)malloc(sizeof(int) * ((size_t)P->nq + 1)), *r2 = (int *)malloc(sizeof(int) * ((size_t)P->nr + 1));
    for (int k = 0; k < nb; k++) {
        for (int j = bs[k]; j < bs[k + 1]; j++) map[j] = gone[k] ? -1 : n2++;
        if (gone[k]) continue;
        if (bk[k] == 0) nf2++; else if (bk[k] == 1) nl2++;
        else if (bk[k] == 2) q2[nq2++] = bs[k + 1] - bs[k]; else r2[nr2++] = bs[k + 1] - bs[k];
    }
    int *Ap = (int *)malloc(sizeof(int) * ((size_t)n2 + 1)); double *c2 = (double *)malloc(sizeof(double) * ((size_t)n2 + 1));
    for (int j = 0; j < n; j++) if (map[j] >= 0) { Ap[map[j]] = P->Ap[j]; c2[map[j]] = P->c[j]; }
    Ap[n2] = P->Ap[n];          /* (the removed columns are empty: the others keep their positions in Ai, Ax) */
    SocpProb Q = *P; Q.n = n2; Q.nf = nf2; Q.nl = nl2; Q.nq = nq2; Q.q = q2; Q.nr = nr2; Q.r = r2; Q.Ap = Ap; Q.c = c2;
    if (opt && opt->verbose > 0) socp_printf("presolve: %d of %d blocks (variables and cones) in no row of A are at 0 (columns %d -> %d)\n", ngone, nb, n, n2);
    SNOTE("presolve: %d variables and cones in no row of A set to 0 (columns %d -> %d)", ngone, n, n2);
    const int st = inner(&Q, opt, R);
    if (R->x && R->z) {
        double *x = (double *)xcalloc((size_t)n, sizeof(double)), *z = (double *)xcalloc((size_t)n, sizeof(double));
        const int cert = st == SOCP_PINF || st == SOCP_DINF;
        for (int j = 0; j < n; j++) { if (map[j] >= 0) { x[j] = R->x[map[j]]; z[j] = R->z[map[j]]; } else z[j] = cert ? 0.0 : P->c[j]; }
        free(R->x); free(R->z); R->x = x; R->z = z;
    }
    free(map); free(q2); free(r2); free(Ap); free(c2); free(bs); free(bk); free(gone);
    return st;
}
static int socp_solve_seq_empty(const SocpProb *P, const SocpOpts *opt, SocpRes *R) { return socp_drop_empty(P, opt, R, socp_solve_seq); }
/* (19 October 2026, LP/SOCP) the shape of the dual: a block (a nonnegative variable or a cone) all of whose columns are
 * singletons, each in a row of its own, is a slack of those rows; the dual then needs a row only for each free column
 * and each column of the other blocks. Returns that number of rows; sl[block] = 1 for the slack blocks. */
static int dual_shape(const SocpProb *P, unsigned char *slk, int *n_ro) {
    const int m = P->m, nf = P->nf, nl = P->nl, nbk = nl + P->nq + P->nr;
    int *bs = (int *)malloc(sizeof(int) * ((size_t)nbk + 1)), *rowc = (int *)calloc((size_t)m + 1, sizeof(int));
    { int s_ = nf, b_ = 0; for (int j = 0; j < nl; j++) bs[b_++] = s_++; for (int k = 0; k < P->nq; k++) { bs[b_++] = s_; s_ += P->q[k]; } for (int k = 0; k < P->nr; k++) { bs[b_++] = s_; s_ += P->r[k]; } bs[nbk] = s_; }
    for (int b_ = 0; b_ < nbk; b_++) {
        int ok = 1;
        for (int j = bs[b_]; j < bs[b_ + 1] && ok; j++) if (P->Ap[j + 1] - P->Ap[j] != 1 || P->Ax[P->Ap[j]] == 0.0) ok = 0;
        slk[b_] = (unsigned char)ok;
        if (ok) for (int j = bs[b_]; j < bs[b_ + 1]; j++) rowc[P->Ai[P->Ap[j]]]++;
    }
    int mrows = nf, nro = m;
    for (int b_ = 0; b_ < nbk; b_++) {
        if (slk[b_]) for (int j = bs[b_]; j < bs[b_ + 1]; j++) if (rowc[P->Ai[P->Ap[j]]] != 1) { slk[b_] = 0; break; }
        if (slk[b_]) nro -= bs[b_ + 1] - bs[b_]; else mrows += bs[b_ + 1] - bs[b_];
    }
    free(bs); free(rowc);
    *n_ro = nro;
    return mrows;
}
/* (19 October 2026, LP/SOCP) the dual form. min c'x, A x = b, x in K with the slack blocks s (dual_shape) is solved
 * through its dual, written as a problem of the same form: for a slack s of row i (coefficient a_s) the dual's y_i =
 * (c_s - w_s) / a_s with w_s in K* = K; the rows without a slack keep a free y_i; the dual's equations are one for each
 * free column and each column of the other blocks j (with z_j in K):
 *     sum_{i slack} -(a_ij / a_s) w_s + sum_{i free} a_ij y_i (+ z_j) = c_j - sum_{i slack} a_ij c_s / a_s,
 * minimizing sum_{i slack} (b_i / a_s) w_s - sum_{i free} b_i y_i. Its multipliers give x: x_j = -y''_j for the free
 * and other columns, x_s = z''_s (the slack of w_s); an infeasibility certificate of one is a certificate of the other.
 * A cone on an affine expression of a few variables (a CBF file's affine conic constraints, a modelling layer's
 * epigraphs) has many rows and few free columns: the smallest enclosing ball of 20,000 points in 20 dimensions has
 * 420,001 rows and its dual 22. Used when the dual has fewer rows, at most NE_MAXFREE free variables, and its factor,
 * estimated from the union of the rows of each block, is not larger than the primal's by more than 20 %;
 * BRISK_CONEDUAL=0 never, 1 whenever it has fewer rows. */
static int socp_solve_dualform(const SocpProb *P, const SocpOpts *opt, SocpRes *R) {
    const int m = P->m, n = P->n, nf = P->nf, nl = P->nl, nbk = nl + P->nq + P->nr;
    const char *e = getenv("BRISK_CONEDUAL"); const int mode = e ? atoi(e) : 2;
    if (mode == 0 || m < 1 || n - nf < 1) return socp_solve_seq_empty(P, opt, R);
    unsigned char *slk = (unsigned char *)malloc((size_t)nbk + 1); int nro = 0;
    const int md = dual_shape(P, slk, &nro);
    /* (the dual's free variables are the rows without a slack: free variables are the SOCP solver's weak point, and a
     * dual with many of them was much harder: a branch-flow relaxation 22 -> 47 iterations with 60,001, an AC-OPF
     * relaxation 29 -> 70, a powered-descent problem reduced accuracy after 400. At most NE_MAXFREE, as for the normal
     * equations) */
    static int dmf = -1; if (dmf < 0) { const char *e_ = getenv("BRISK_CONEDUALMAXFREE"); dmf = e_ ? atoi(e_) : ne_maxfree(); }   /* (a test: the cap on the dual's free variables) */
    const int use = md >= 1 && md < m && (mode == 1 || (m >= 1000 && nro <= dmf));
    if (!use) { free(slk); return socp_solve_seq_empty(P, opt, R); }
    const double t0 = sc_time();
    const int nz = P->Ap[n];
    /* the role of each column: srow[j] the row of a slack column (else -1); prow[j] the dual's row of a free or other
     * column (else -1); the slack column of each row (else -1) and the dual's free column of each row without one */
    int *srow = (int *)malloc(sizeof(int) * ((size_t)n + 1)), *prow = (int *)malloc(sizeof(int) * ((size_t)n + 1));
    int *rs_ = (int *)malloc(sizeof(int) * ((size_t)m + 1)), *rfree = (int *)malloc(sizeof(int) * ((size_t)m + 1));
    for (int i = 0; i < m; i++) { rs_[i] = -1; rfree[i] = -1; }
    int nrow2 = 0;
    for (int j = 0; j < nf; j++) { srow[j] = -1; prow[j] = nrow2++; }
    {
        int s_ = nf;
        for (int b_ = 0; b_ < nbk; b_++) {
            const int d = b_ < nl ? 1 : b_ < nl + P->nq ? P->q[b_ - nl] : P->r[b_ - nl - P->nq];
            for (int j = s_; j < s_ + d; j++) {
                if (slk[b_]) { srow[j] = P->Ai[P->Ap[j]]; rs_[srow[j]] = j; prow[j] = -1; }
                else { srow[j] = -1; prow[j] = nrow2++; }
            }
            s_ += d;
        }
    }
    int nfr2 = 0;
    for (int i = 0; i < m; i++) if (rs_[i] < 0) rfree[i] = nfr2++;
    const int n2 = nfr2 + (n - nf);
    /* the rows of A (entries of free and other columns only) */
    int *rp = (int *)calloc((size_t)m + 2, sizeof(int)), *rc = (int *)malloc(sizeof(int) * ((size_t)nz + 1)); double *rv = (double *)malloc(sizeof(double) * ((size_t)nz + 1));
    for (int j = 0; j < n; j++) if (prow[j] >= 0) for (int p = P->Ap[j]; p < P->Ap[j + 1]; p++) rp[P->Ai[p] + 2]++;
    for (int i = 0; i < m; i++) rp[i + 2] += rp[i + 1];
    for (int j = 0; j < n; j++) if (prow[j] >= 0) for (int p = P->Ap[j]; p < P->Ap[j + 1]; p++) { const int q = rp[P->Ai[p] + 1]++; rc[q] = j; rv[q] = P->Ax[p]; }
    /* the dual: columns [free y of the rows without a slack | w or z of the blocks, in their order] */
    int *Ap2 = (int *)calloc((size_t)n2 + 2, sizeof(int));
    for (int i = 0; i < m; i++) if (rfree[i] >= 0) Ap2[rfree[i] + 1] = rp[i + 1] - rp[i];
    for (int j = nf; j < n; j++) Ap2[nfr2 + (j - nf) + 1] = srow[j] >= 0 ? rp[srow[j] + 1] - rp[srow[j]] : 1;
    for (int j = 0; j < n2; j++) Ap2[j + 1] += Ap2[j];
    int *Ai2 = (int *)malloc(sizeof(int) * ((size_t)Ap2[n2] + 1)); double *Ax2 = (double *)malloc(sizeof(double) * ((size_t)Ap2[n2] + 1));
    double *b2 = (double *)malloc(sizeof(double) * ((size_t)nrow2 + 1)), *c2 = (double *)malloc(sizeof(double) * ((size_t)n2 + 1));
    for (int j = 0; j < n; j++) if (prow[j] >= 0) b2[prow[j]] = P->c[j];
    for (int i = 0; i < m; i++) {
        if (rfree[i] >= 0) {
            int w = Ap2[rfree[i]];
            for (int q = rp[i]; q < rp[i + 1]; q++) { Ai2[w] = prow[rc[q]]; Ax2[w++] = rv[q]; }
            c2[rfree[i]] = -P->b[i];
        } else {
            const int s = rs_[i]; const double as = P->Ax[P->Ap[s]], cs = P->c[s];
            int w = Ap2[nfr2 + (s - nf)];
            for (int q = rp[i]; q < rp[i + 1]; q++) { Ai2[w] = prow[rc[q]]; Ax2[w++] = -rv[q] / as; b2[prow[rc[q]]] -= rv[q] * cs / as; }
            c2[nfr2 + (s - nf)] = P->b[i] / as;
        }
    }
    for (int j = nf; j < n; j++) if (prow[j] >= 0) { const int w = Ap2[nfr2 + (j - nf)]; Ai2[w] = prow[j]; Ax2[w] = 1.0; c2[nfr2 + (j - nf)] = 0.0; }
    /* each column's rows ascending (the rows of a slack follow the order of its row's columns, which ascends) */
    double est_d = 0, est_p = 0;     /* the factors' sizes, estimated: a block of dimension <= SOCP_DENSE_MAX fills its d columns to the union of their rows */
    /* (27 October 2026, the SOCP developer; BRISK_CONEDUALREN=0: as before) the primal's estimate on the problem its presolve will
     * factor: a free variable u tied to a column v of a block by a row of two, {u, v}, v in no other row, is substituted by moving
     * its column onto v (the renaming of elim_run), so v's block is counted with u's rows and u not at all. The robust affine
     * SOCPs (a CVXPY canonical form: each x_j tied to a cone's entry) were estimated at 1.48e6 entries against the dual's 2.07e6
     * while the substitution made every cone column dense: the primal's normal equations 1,921 dense rows against the dual's
     * 1,153, 3.4 against 2.5 s (MOSEK 3.3), and 19.5 against 14.1 s at twice the size (MOSEK 13.9) */
    static int dren = -1; if (dren < 0) dren = getenv("BRISK_CONEDUALREN") ? atoi(getenv("BRISK_CONEDUALREN")) : 1;
    static int dfill = -1; if (dfill < 0) dfill = getenv("BRISK_CONEDUALFILL") ? atoi(getenv("BRISK_CONEDUALFILL")) : 1;
    int *ren = NULL, *renrow = NULL;     /* ren[v]: the free column moved onto v (-1: none); renrow[v]: the row of two */
    if (dren && nf > 0) {
        int *rlen = (int *)calloc((size_t)m + 1, sizeof(int)), *rfst = (int *)malloc(sizeof(int) * ((size_t)m + 1)), *roth = (int *)malloc(sizeof(int) * ((size_t)m + 1));
        for (int i = 0; i < m; i++) { rfst[i] = roth[i] = -1; }
        for (int j = 0; j < n; j++) for (int p = P->Ap[j]; p < P->Ap[j + 1]; p++) { const int i = P->Ai[p]; if (rlen[i]++ == 0) rfst[i] = j; else roth[i] = j; }
        ren = (int *)malloc(sizeof(int) * ((size_t)n + 1)); renrow = (int *)malloc(sizeof(int) * ((size_t)n + 1));
        unsigned char *used = (unsigned char *)calloc((size_t)n + 1, 1);
        for (int j = 0; j < n; j++) ren[j] = -1;
        for (int u = 0; u < nf; u++) for (int p = P->Ap[u]; p < P->Ap[u + 1]; p++) {
            const int i = P->Ai[p]; if (rlen[i] != 2) continue;
            const int v = rfst[i] == u ? roth[i] : rfst[i];
            if (v < nf || used[v] || P->Ap[v + 1] - P->Ap[v] != 1 || P->b[i] != 0.0) continue;
            ren[v] = u; renrow[v] = i; used[v] = 1; break;
        }
        free(rlen); free(rfst); free(roth); free(used);
    }
    {
        int *mk = (int *)malloc(sizeof(int) * ((size_t)(nrow2 > m ? nrow2 : m) + 1));
        unsigned char *gone = ren ? (unsigned char *)calloc((size_t)nf + 1, 1) : NULL;
        if (ren) for (int j = nf; j < n; j++) if (ren[j] >= 0) gone[ren[j]] = 1;
        for (int pass = 0; pass < 2; pass++) {
            const int *Ap_ = pass ? Ap2 : P->Ap, *Ai_ = pass ? Ai2 : P->Ai; const int off = pass ? nfr2 : nf, mm = pass ? nrow2 : m;
            for (int i = 0; i < mm; i++) mk[i] = -1;
            double est = 0, bfill = 0; int s_ = off;
            for (int b_ = 0; b_ < nbk; b_++) {
                const int d = b_ < nl ? 1 : b_ < nl + P->nq ? P->q[b_ - nl] : P->r[b_ - nl - P->nq];
                int u = 0; long nzb = 0; double fill = 0;
                for (int j = s_; j < s_ + d; j++) {
                    const int jr = !pass && ren && ren[j] >= 0 ? ren[j] : j;      /* (the column it will have) */
                    nzb += Ap_[jr + 1] - Ap_[jr];
                    { const double l_ = Ap_[jr + 1] - Ap_[jr]; fill += fmin(0.5 * l_ * (l_ - 1.0), (double)mm); }     /* (a dense column goes to the low-rank update: about a column of the rows) */
                    for (int p = Ap_[jr]; p < Ap_[jr + 1]; p++) { if (jr != j && Ai_[p] == renrow[j]) continue; if (mk[Ai_[p]] != b_) { mk[Ai_[p]] = b_; u++; } }
                }
                /* (a large cone enters by its diagonal and a rank-two term: each column's elimination joins its rows, l (l - 1) / 2
                 * entries, at most the u (u - 1) / 2 of the union, summed below; BRISK_CONEDUALFILL=0: its entries alone, as before) */
                est += d <= SOCP_DENSE_MAX ? (double)d * u : (double)nzb + 2.0 * u;
                if (d > SOCP_DENSE_MAX && dfill) bfill += fmin(fill, 0.5 * u * (u - 1.0));     /* (each column joins its rows, l (l - 1) / 2, or, dense, goes to the update at about one column of the rows; the block's union at most) */
                s_ += d;
            }
            /* (the large cones' fill lands among the rows: when it reaches the whole row block, the rows left after the
             * substitutions squared, the system is dense of that order and is counted so; below that the low-rank update and the
             * sparse elimination take the cones' columns, as the estimate always assumed) */
            { int nren = 0; if (!pass && ren) for (int j = nf; j < n; j++) nren += ren[j] >= 0;
              const double me_ = (double)(mm - nren), full_ = 0.5 * me_ * (me_ - 1.0); if (bfill >= full_) est += full_; }
            for (int j = 0; j < off; j++) if (pass || !gone || !gone[j]) est += Ap_[j + 1] - Ap_[j];
            if (pass) est_d = est; else est_p = est;
        }
        free(mk); free(gone);
    }
    free(ren); free(renrow);
    if (getenv("BRISK_CONEDBG")) socp_printf("  [the dual form: factor estimated %.3g against the primal's %.3g; nnz %d against %d; rows %d against %d, free %d against %d]\n", est_d, est_p, Ap2[n2], nz, nrow2, m, nfr2, nf);
    /* (the factor's size decides beyond the rows: FIR filter design has 152 rows in the dual against 42,301, but its
     * cones of three each meet most of them, and the dual's factor had 6.4e6 entries against 4.5e6: 6.1 -> 7.4 s) */
    if (getenv("BRISK_CONEDUALSCAN")) { printf("dualscan %s est_d %.4g est_p %.4g rows %d %d\n", est_d > 1.2 * est_p ? "primal" : "dual", est_d, est_p, nrow2, m); fflush(stdout); exit(0); }   /* (experiments: the choice alone) */
    if (mode == 2 && est_d > 1.2 * est_p) {
        free(slk); free(srow); free(prow); free(rs_); free(rfree); free(rp); free(rc); free(rv); free(Ap2); free(Ai2); free(Ax2); free(b2); free(c2);
        return socp_solve_seq_empty(P, opt, R);
    }
    SocpProb Q = *P;
    Q.m = nrow2; Q.n = n2; Q.Ap = Ap2; Q.Ai = Ai2; Q.Ax = Ax2; Q.b = b2; Q.c = c2; Q.nf = nfr2;
    SocpOpts o2; if (opt) o2 = *opt; else socp_default_opts(&o2);
    if (o2.verbose > 0) socp_printf("the dual solved: %d rows instead of %d (%d slack blocks; %d free variables)\n", nrow2, m, (int)(n - nf - (nrow2 - nf)), nfr2);
    SNOTE("the dual solved by the SOCP solver: %d rows instead of %d", nrow2, m);
    const char *pfx = g_stat_prefix; g_stat_prefix = "the dual: ";
    SocpRes R2; memset(&R2, 0, sizeof R2);
    const int st2 = socp_solve_seq_empty(&Q, &o2, &R2);
    g_stat_prefix = pfx;
    int st = st2 == SOCP_PINF ? SOCP_DINF : st2 == SOCP_DINF ? SOCP_PINF : st2;
    memset(R, 0, sizeof *R);
    R->x = (double *)calloc((size_t)n + 1, sizeof(double)); R->y = (double *)calloc((size_t)m + 1, sizeof(double)); R->z = (double *)calloc((size_t)n + 1, sizeof(double));
    R->iters = R2.iters; R->lnz = R2.lnz;
    if (R2.x && R2.y && R2.z) {
        const int cert = st2 == SOCP_PINF || st2 == SOCP_DINF;
        if (st2 != SOCP_DINF) {         /* x from the dual's multipliers (a ray when the dual is infeasible) */
            for (int j = 0; j < n; j++) R->x[j] = prow[j] >= 0 ? -R2.y[prow[j]] : R2.z[nfr2 + (j - nf)];
        }
        if (st2 != SOCP_PINF) {         /* y and z from the dual's point (a certificate when it is a ray) */
            for (int i = 0; i < m; i++) {
                if (rfree[i] >= 0) R->y[i] = R2.x[rfree[i]];
                else { const int s = rs_[i]; R->y[i] = ((cert ? 0.0 : P->c[s]) - R2.x[nfr2 + (s - nf)]) / P->Ax[P->Ap[s]]; }
            }
            for (int j = nf; j < n; j++) R->z[j] = R2.x[nfr2 + (j - nf)];
        }
        if (!cert) {          /* the errors of the problem as given */
            double po = 0, dobj = 0, nb = 0, nc = 0, rpn = 0, rd = 0, xz = 0;
            double *ra = (double *)malloc(sizeof(double) * ((size_t)m + 1));
            for (int i = 0; i < m; i++) { ra[i] = -P->b[i]; dobj += P->b[i] * R->y[i]; nb += P->b[i] * P->b[i]; }
            for (int j = 0; j < n; j++) {
                double s = 0; for (int p = P->Ap[j]; p < P->Ap[j + 1]; p++) { ra[P->Ai[p]] += P->Ax[p] * R->x[j]; s += P->Ax[p] * R->y[P->Ai[p]]; }
                const double r = P->c[j] - s - R->z[j];
                rd += r * r; po += P->c[j] * R->x[j]; nc += P->c[j] * P->c[j]; xz += R->x[j] * R->z[j];
            }
            for (int i = 0; i < m; i++) rpn += ra[i] * ra[i];
            free(ra);
            R->pobj = po; R->dobj = dobj;
            R->err[0] = sqrt(rpn) / (1.0 + sqrt(nb)); R->err[1] = sqrt(rd) / (1.0 + sqrt(nc));
            R->err[2] = (po - dobj) / (1.0 + fabs(po) + fabs(dobj)); R->err[3] = xz / (1.0 + fabs(po) + fabs(dobj));
            R->maxerr = fmax(fmax(R->err[0], R->err[1]), fmax(fabs(R->err[2]), R->err[3]));
            if (st == SOCP_OPTIMAL && R->maxerr > 10.0 * o2.tol) st = SOCP_REDUCED;
        }
    }
    if (getenv("BRISK_CONEDBG") && (st == SOCP_PINF || st == SOCP_DINF)) {     /* (the certificate checked on the problem as given) */
        double *t_ = (double *)calloc((size_t)(m > n ? m : n) + 1, sizeof(double)); double nr = 0, nv = 0, ob = 0;
        if (st == SOCP_DINF) { for (int j = 0; j < n; j++) { nv += R->x[j] * R->x[j]; ob += P->c[j] * R->x[j]; for (int p = P->Ap[j]; p < P->Ap[j + 1]; p++) t_[P->Ai[p]] += P->Ax[p] * R->x[j]; } for (int i = 0; i < m; i++) nr += t_[i] * t_[i]; }
        else { for (int i = 0; i < m; i++) { nv += R->y[i] * R->y[i]; ob += P->b[i] * R->y[i]; } for (int j = 0; j < n; j++) { double v = R->z[j]; for (int p = P->Ap[j]; p < P->Ap[j + 1]; p++) v += P->Ax[p] * R->y[P->Ai[p]]; nr += v * v; } }
        socp_printf("  [the certificate on the problem as given: |%s| %.2e, residual %.2e, %s %.3e]\n", st == SOCP_DINF ? "x" : "y", sqrt(nv), sqrt(nr), st == SOCP_DINF ? "c'x" : "b'y", ob);
        free(t_);
    }
    socp_result_free(&R2);
    R->status = st; R->time = sc_time() - t0;
    if (o2.verbose > 0) {
        const int cert = st == SOCP_PINF || st == SOCP_DINF;
        socp_printf("%sstatus: %s", g_stat_prefix, socp_status_line(st));
        if (!cert) socp_printf(" (max rel error %.1e)", R->maxerr);
        socp_printf("   iterations: %d   time: %.3fs\n", R->iters, R->time);
        if (!cert) socp_printf("  primal obj c'x = %+.12e   dual obj b'y = %+.12e\n  errors: pinf %.1e  dinf %.1e  gap %.1e  x'z %.1e\n", R->pobj, R->dobj, R->err[0], R->err[1], R->err[2], R->err[3]);
    }
    free(slk); free(srow); free(prow); free(rs_); free(rfree); free(rp); free(rc); free(rv); free(Ap2); free(Ai2); free(Ax2); free(b2); free(c2);
    return st;
}
int brisk_threads_busy(int verbose);
/* a solve as given from a warm start (or keeping its centred iterate): when the warm-started iteration breaks down or
 * reaches its limit, the problem is solved from the usual start (the same path), the iterations of both counted
 * (sssd-weak-15-4, a mixed-integer SOCP's node: the warm start NUMERICAL DIFFICULTIES after one iteration, cold OPTIMAL
 * in 24) */
static int socp_solve_warm(const SocpProb *P, const SocpOpts *opt, SocpRes *R) {
    int st = socp_solve_core(P, opt, R);
    if (opt->wx && (st == SOCP_NUMERR || st == SOCP_MAXIT)) {
        const int it = R->iters;
        SocpOpts o2 = *opt; o2.wx = o2.wy = o2.wz = NULL;
        socp_result_free(R);
        st = socp_solve_core(P, &o2, R);
        R->iters += it;
    }
    return st;
}
/* (24 October 2026, the SOCP developer) flush-to-zero and denormals-are-zero, as the SDP solver sets them (solver.c,
 * set_fast_fp): the triangular solves carry a right-hand side that is nonzero in a few places along a long chain of the
 * factor (the dynamics of a control problem over 20,000 steps), where it decays geometrically, correctly, below 1e-308;
 * each operation on such a subnormal number costs about 100 times a normal one (socp_gctrl_20000x12x6, SOCP: 148,000
 * subnormal entries in the solutions of one iteration, its solves 4 times their cost, the iteration 2.0 instead of
 * 0.6 s). Set on every thread of the solver, so that the bits do not depend on which thread computes. */
static void cone_fast_fp(void) {
#if defined(__x86_64__) || defined(__i386__)
    _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
    _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);
#endif
}
/* the caller's mode back on every thread the solve set (the master's mode as found; a host's OpenMP threads keep no
 * flush-to-zero from a solve) */
static void cone_restore_fp(unsigned int csr) {
#if defined(__x86_64__) || defined(__i386__)
    _mm_setcsr(csr);
#ifdef _OPENMP
    if (!omp_in_parallel()) {
        #pragma omp parallel
        _mm_setcsr(csr);
    }
#endif
#else
    (void)csr;
#endif
}
/* (27 October 2026, the SOCP developer) folding by symmetry: the coarsest equitable partition of the rows and columns
 * (the LP method's colour refinement, lpsolve.c), with each cone's first entry (a rotated cone's first two) a class of its
 * own and the other entries of a cone allowed to share a class only with entries of the same cone. Permuting those entries
 * is an automorphism of the cone and permuting the columns of a class with the rows of theirs maps the problem onto
 * itself, so averaging a solution over the classes keeps it feasible and optimal: the problem in the class averages has
 * the same optimal value. It is written isometrically: row class l of m_l rows and column class k of n_k columns give the
 * entry B_kl sqrt(n_k / m_l) (B_kl the sum over the rows of class l of one column of class k), b_l sqrt(m_l) and
 * c_k sqrt(n_k), and x_j = X_k / sqrt(n_k), y_i = Y_l / sqrt(m_l), z_j = Z_k / sqrt(n_k): a cone's class of n_k equal
 * entries is one entry of the same norm, and the residuals, the objectives and every norm of the point are the original
 * problem's. The sums are verified (equal over each class within 1e-12 relative) before the fold is taken. Data fitting
 * with repeated observations (a bootstrap, categorical data): each repetition is a class. BRISK_CONENOFOLD=1: off */
static inline uint64_t fold_mix64(uint64_t x) { x += 0x9e3779b97f4a7c15ull; x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ull; x = (x ^ (x >> 27)) * 0x94d049bb133111ebull; return x ^ (x >> 31); }
static inline uint64_t dbl_bits(double v) { uint64_t u; memcpy(&u, &v, 8); return u; }
/* the number of distinct keys, by an open-addressing table of twice their number */
static long count_distinct64(const uint64_t *k, int n) {
    size_t cap = 16; while (cap < 2 * (size_t)n + 2) cap <<= 1;
    uint64_t *t = (uint64_t *)calloc(cap, sizeof(uint64_t)); char *used = (char *)calloc(cap, 1); long d = 0;
    for (int i = 0; i < n; i++) { size_t h = (size_t)fold_mix64(k[i]) & (cap - 1);
        while (used[h] && t[h] != k[i]) h = (h + 1) & (cap - 1);
        if (!used[h]) { used[h] = 1; t[h] = k[i]; d++; } }
    free(t); free(used); return d;
}
/* keys to classes numbered in the order of their first element (deterministic); returns the number of classes */
static int group64(const uint64_t *k, int n, int *cls, size_t cap, uint64_t *tk, int *tid, char *used) {
    memset(used, 0, cap); int d = 0;
    for (int i = 0; i < n; i++) { size_t h = (size_t)fold_mix64(k[i]) & (cap - 1);
        while (used[h] && tk[h] != k[i]) h = (h + 1) & (cap - 1);
        if (!used[h]) { used[h] = 1; tk[h] = k[i]; tid[h] = d++; }
        cls[i] = tid[h]; }
    return d;
}
/* (27 October 2026) the coarsest equitable partition by colour refinement, each pass from A's columns alone (a row's key the
 * sum of hashes of (value, class of the column) over its entries, with its own class; a column's likewise), the keys grouped
 * by a hash table: no sort and no transpose (the LP method's incremental refinement, lpsolve.c, sorts each pass: 13 ms
 * on a power-flow relaxation of 31,000 rows and columns that does not fold). A collision of two keys can only merge
 * classes; the fold verifies the sums before it is taken. Returns 1 with the stable classes when they number at most
 * 0.8 (m + n), else 0 */
static int socp_fold_partition(const SocpProb *P, const double *u, int *rcl, int *ccl, int *nrp, int *ncp) {
    const int m = P->m, n = P->n, nf = P->nf; const double lim = 0.8 * ((double)m + n);
    size_t cap = 16; while (cap < 2 * (size_t)(m > n ? m : n) + 2) cap <<= 1;
    uint64_t *kr = (uint64_t *)malloc(sizeof(uint64_t) * (size_t)m), *kc = (uint64_t *)malloc(sizeof(uint64_t) * (size_t)n), *tk = (uint64_t *)malloc(sizeof(uint64_t) * cap);
    int *tid = (int *)malloc(sizeof(int) * cap); char *used = (char *)malloc(cap);
    for (int i = 0; i < m; i++) kr[i] = fold_mix64(dbl_bits(P->b[i] == 0.0 ? 0.0 : P->b[i]));
    for (int j = 0; j < n; j++) kc[j] = fold_mix64(dbl_bits(P->c[j] == 0.0 ? 0.0 : P->c[j]) ^ fold_mix64(dbl_bits(u[j]) + (j < nf ? 0x5bd1e995ull : 0)));
    int nr = group64(kr, m, rcl, cap, tk, tid, used), nc = group64(kc, n, ccl, cap, tk, tid, used), ok = 0;
    for (int pass = 0; pass < 200; pass++) {
        if ((double)nr + nc > lim) break;
        for (int i = 0; i < m; i++) kr[i] = fold_mix64((uint64_t)rcl[i] * 0x9e3779b97f4a7c15ull + 1);
        for (int j = 0; j < n; j++) { const uint64_t cj = fold_mix64((uint64_t)ccl[j] + 0x1234567ull);
            for (int p = P->Ap[j]; p < P->Ap[j + 1]; p++) kr[P->Ai[p]] += fold_mix64(dbl_bits(P->Ax[p] == 0.0 ? 0.0 : P->Ax[p]) ^ cj); }
        const int nr2 = group64(kr, m, rcl, cap, tk, tid, used);
        for (int j = 0; j < n; j++) { uint64_t h = fold_mix64((uint64_t)ccl[j] * 0x94d049bb133111ebull + 2);
            for (int p = P->Ap[j]; p < P->Ap[j + 1]; p++) h += fold_mix64(dbl_bits(P->Ax[p] == 0.0 ? 0.0 : P->Ax[p]) ^ fold_mix64((uint64_t)rcl[P->Ai[p]] + 0x7654321ull));
            kc[j] = h; }
        const int nc2 = group64(kc, n, ccl, cap, tk, tid, used);
        const int stable = nr2 == nr && nc2 == nc; nr = nr2; nc = nc2;
        if (stable) { ok = (double)nr + nc <= lim && (nr < m || nc < n); break; }
    }
    free(kr); free(kc); free(tk); free(tid); free(used);
    *nrp = nr; *ncp = nc; return ok;
}
static int socp_fold_try(const SocpProb *P, const SocpOpts *opt, SocpRes *R, int (*inner)(const SocpProb *, const SocpOpts *, SocpRes *), int *done) {
    *done = 0;
    static int off = -1; if (off < 0) { const char *e_ = getenv("BRISK_CONENOFOLD"); off = e_ && atoi(e_) != 0; }
    const int m = P->m, n = P->n, nf = P->nf, nl = P->nl, ncone = P->nq + P->nr;
    if (off || m < 8 || n < 8) return 0;
    const double t0 = sc_time();
    /* the cones' columns: their cone and whether they are a fixed entry (the first, a rotated cone's first two) */
    int *ck = (int *)malloc(sizeof(int) * (size_t)n); double *u = (double *)malloc(sizeof(double) * (size_t)n);
    { int j = 0; for (; j < nf + nl; j++) { ck[j] = -1; u[j] = 1e300; }
      for (int k = 0; k < ncone; k++) { const int d = k < P->nq ? P->q[k] : P->r[k - P->nq], fx = k < P->nq ? 1 : 2;
          for (int t = 0; t < d; t++, j++) { ck[j] = k; u[j] = t < fx ? -1.0 - 3.0 * (double)j : -2.0 - 3.0 * (double)k; } }   /* (distinct marks: a fixed entry alone, a cone's other entries together) */
      if (j != n) { free(ck); free(u); return 0; } }
    const double tpt_ = sc_time();
    int *rcl = (int *)malloc(sizeof(int) * (size_t)m), *ccl = (int *)malloc(sizeof(int) * (size_t)n), nr = m, nc = n;
    const int folds = socp_fold_partition(P, u, rcl, ccl, &nr, &nc);
    free(u);
    if (dbg_on()) socp_printf("  [folding: %d row and %d column classes of %d and %d%s (%.4fs)]\n", nr, nc, m, n, folds ? "" : ": no fold", sc_time() - t0); (void)tpt_;
    const double tv0_ = sc_time();
    if (!folds || (double)nr + nc > 0.8 * ((double)m + n)) { free(ck); free(rcl); free(ccl); return 0; }
    /* the classes: sizes, a representative, the kind; a class within one block (free, linear, or one cone's other entries) */
    int *rsz = (int *)calloc((size_t)nr, sizeof(int)), *csz = (int *)calloc((size_t)nc, sizeof(int)), *crep = (int *)malloc(sizeof(int) * (size_t)nc), *rrep = (int *)malloc(sizeof(int) * (size_t)nr);
    for (int k = 0; k < nc; k++) crep[k] = -1;
    for (int l = 0; l < nr; l++) rrep[l] = -1;
    for (int i = 0; i < m; i++) { if (rrep[rcl[i]] < 0) rrep[rcl[i]] = i; rsz[rcl[i]]++; }
    int ok = 1;
    { int j = nf + nl; for (int k = 0; k < ncone && ok; k++) { const int d = k < P->nq ? P->q[k] : P->r[k - P->nq], fx = k < P->nq ? 1 : 2;     /* (a cone's fixed entries alone in their classes) */
          for (int t = 0; t < fx; t++) { const int jj = j + t; for (int j2 = j; j2 < j + d; j2++) if (j2 != jj && ccl[j2] == ccl[jj]) { ok = 0; break; } }
          j += d; } }
    for (int j = 0; j < n && ok; j++) { const int k = ccl[j]; if (crep[k] < 0) crep[k] = j; else { const int r_ = crep[k];
        if ((j < nf) != (r_ < nf) || (j < nf + nl) != (r_ < nf + nl) || ck[j] != ck[r_]) ok = 0; } csz[k]++; }
    /* the sums verified: for each column, its sums over each row class equal its representative's (B), and for each row,
     * its sums over each column class equal its representative's */
    double *acc = (double *)calloc((size_t)(nr > nc ? nr : nc) + 1, sizeof(double)); int *tch = (int *)malloc(sizeof(int) * ((size_t)(nr > nc ? nr : nc) + 1));
    int *Bp = (int *)calloc((size_t)nc + 1, sizeof(int)), *Bi = NULL; double *Bx = NULL;
    if (ok) {   /* the representatives' column sums first: the quotient's columns */
        long nzq = 0; for (int k = 0; k < nc; k++) { const int j = crep[k]; nzq += P->Ap[j + 1] - P->Ap[j]; }
        Bi = (int *)malloc(sizeof(int) * (size_t)(nzq + 1)); Bx = (double *)malloc(sizeof(double) * (size_t)(nzq + 1)); long q = 0;
        for (int k = 0; k < nc; k++) { const int j = crep[k]; int nt = 0;
            for (int p = P->Ap[j]; p < P->Ap[j + 1]; p++) { const int l = rcl[P->Ai[p]]; if (acc[l] == 0.0) tch[nt++] = l; acc[l] += P->Ax[p]; if (acc[l] == 0.0) acc[l] = 1e-300; }
            for (int a = 0; a < nt; a++) { const int l = tch[a]; if (acc[l] != 1e-300) { Bi[q] = l; Bx[q++] = acc[l]; } acc[l] = 0.0; }
            Bp[k + 1] = (int)q; }
        /* every column against its representative */
        double *rv = (double *)calloc((size_t)nr + 1, sizeof(double));
        for (int j = 0; j < n && ok; j++) { const int k = ccl[j]; if (crep[k] == j) continue;
            for (int q2 = Bp[k]; q2 < Bp[k + 1]; q2++) rv[Bi[q2]] = Bx[q2];
            int nt = 0;
            for (int p = P->Ap[j]; p < P->Ap[j + 1]; p++) { const int l = rcl[P->Ai[p]]; if (acc[l] == 0.0) tch[nt++] = l; acc[l] += P->Ax[p]; if (acc[l] == 0.0) acc[l] = 1e-300; }
            for (int a = 0; a < nt; a++) { const int l = tch[a]; const double v = acc[l] == 1e-300 ? 0.0 : acc[l]; if (fabs(v - rv[l]) > 1e-12 * fmax(fabs(v), fabs(rv[l])) + 1e-300) ok = 0; acc[l] = 0.0; rv[l] = 0.0; }
            for (int q2 = Bp[k]; q2 < Bp[k + 1]; q2++) { if (rv[Bi[q2]] != 0.0) ok = 0; rv[Bi[q2]] = 0.0; }    /* (an entry of the representative missing here) */
        }
        free(rv);
        if (dbg_on()) socp_printf("  [folding: the columns' sums verified %.3fs after the partition]\n", sc_time() - tv0_);
        /* every row against its representative, column class by column class: the sums over the class accumulated by rows
         * from A's columns (no transpose: a dense matrix's transpose costs a cache miss an entry), then each touched row
         * against its class's representative; a class touched in part has rows whose sum is 0 against a nonzero one */
        if (ok) {
            int *cb_ = (int *)calloc((size_t)nc + 1, sizeof(int)), *cj_ = (int *)malloc(sizeof(int) * ((size_t)n + 1));      /* (the columns by class) */
            for (int j = 0; j < n; j++) cb_[ccl[j] + 1]++;
            for (int k = 0; k < nc; k++) cb_[k + 1] += cb_[k];
            { int *fill_ = (int *)malloc(sizeof(int) * ((size_t)nc + 1)); memcpy(fill_, cb_, sizeof(int) * (size_t)nc); for (int j = 0; j < n; j++) cj_[fill_[ccl[j]]++] = j; free(fill_); }
            double *S_ = (double *)calloc((size_t)m + 1, sizeof(double)); char *tr_ = (char *)calloc((size_t)m + 1, 1); int *rt_ = (int *)malloc(sizeof(int) * ((size_t)m + 1));
            int *cnt_ = (int *)calloc((size_t)nr + 1, sizeof(int)), *lt_ = (int *)malloc(sizeof(int) * ((size_t)nr + 1));
            for (int k = 0; k < nc && ok; k++) {
                int nt = 0, nlt = 0;
                for (int a_ = cb_[k]; a_ < cb_[k + 1]; a_++) { const int j = cj_[a_];
                    for (int p = P->Ap[j]; p < P->Ap[j + 1]; p++) { const int i = P->Ai[p]; if (!tr_[i]) { tr_[i] = 1; rt_[nt++] = i; } S_[i] += P->Ax[p]; } }
                for (int t = 0; t < nt && ok; t++) { const int i = rt_[t], l = rcl[i]; const double v = S_[i], w = S_[rrep[l]];
                    if (fabs(v - w) > 1e-12 * fmax(fabs(v), fabs(w)) + 1e-300) ok = 0;
                    if (cnt_[l]++ == 0) lt_[nlt++] = l; }
                for (int t = 0; t < nlt && ok; t++) { const int l = lt_[t]; if (cnt_[l] != rsz[l] && S_[rrep[l]] != 0.0) ok = 0; }       /* (members untouched: a sum of 0) */
                for (int t = 0; t < nlt; t++) cnt_[lt_[t]] = 0;
                for (int t = 0; t < nt; t++) { S_[rt_[t]] = 0.0; tr_[rt_[t]] = 0; }
            }
            for (int i = 0; i < m && ok; i++) if (P->b[i] != P->b[rrep[rcl[i]]]) ok = 0;
            free(cb_); free(cj_); free(S_); free(tr_); free(rt_); free(cnt_); free(lt_);
        }
        for (int j = 0; j < n && ok; j++) if (P->c[j] != P->c[crep[ccl[j]]]) ok = 0;
    }
    free(acc); free(tch);
    if (dbg_on()) socp_printf("  [folding: the sums verified in %.3fs]\n", sc_time() - tv0_);
    if (!ok) { if (dbg_on()) socp_printf("  [folding: the partition of %d row and %d column classes is not equitable: not folded]\n", nr, nc);
        free(ck); free(rcl); free(ccl); free(rsz); free(csz); free(crep); free(rrep); free(Bp); free(Bi); free(Bx); return 0; }
    /* the quotient's columns in the problem's block order: the free classes, the linear ones, then each cone's (its fixed
     * entries first, as in the problem) */
    int *ord = (int *)malloc(sizeof(int) * (size_t)nc), *pos = (int *)malloc(sizeof(int) * (size_t)nc), no = 0;
    { char *seen = (char *)calloc((size_t)nc, 1); for (int j = 0; j < n; j++) { const int k = ccl[j]; if (!seen[k]) { seen[k] = 1; pos[k] = no; ord[no++] = k; } } free(seen); }
    SocpProb Q; memset(&Q, 0, sizeof Q);
    int nfq = 0, nlq = 0; for (int a = 0; a < nc; a++) { const int j = crep[ord[a]]; if (j < nf) nfq++; else if (j < nf + nl) nlq++; }
    int *qq = (int *)malloc(sizeof(int) * ((size_t)ncone + 1)); { for (int k = 0; k < ncone; k++) qq[k] = 0; for (int a = 0; a < nc; a++) { const int j = crep[ord[a]]; if (ck[j] >= 0) qq[ck[j]]++; } }
    int *Qp = (int *)malloc(sizeof(int) * ((size_t)nc + 1)), *Qi = (int *)malloc(sizeof(int) * ((size_t)Bp[nc] + 1)); double *Qx = (double *)malloc(sizeof(double) * ((size_t)Bp[nc] + 1));
    double *bq = (double *)malloc(sizeof(double) * (size_t)nr), *cq = (double *)malloc(sizeof(double) * (size_t)nc);
    /* the quotient's columns with their rows in increasing order: by rows first (the columns in order), then back */
    {   const int nzq = Bp[nc]; int *Tp = (int *)calloc((size_t)nr + 2, sizeof(int)), *Tj = (int *)malloc(sizeof(int) * ((size_t)nzq + 1)); double *Tx = (double *)malloc(sizeof(double) * ((size_t)nzq + 1));
        for (int q2 = 0; q2 < nzq; q2++) Tp[Bi[q2] + 2]++;
        for (int l = 0; l < nr; l++) Tp[l + 2] += Tp[l + 1];
        for (int a = 0; a < nc; a++) { const int k = ord[a]; const double sk = sqrt((double)csz[k]);
            for (int q2 = Bp[k]; q2 < Bp[k + 1]; q2++) { const int l = Bi[q2], t_ = Tp[l + 1]++; Tj[t_] = a; Tx[t_] = Bx[q2] * sk / sqrt((double)rsz[l]); }
            cq[a] = P->c[crep[k]] * sk; }
        for (int a = 0; a <= nc; a++) Qp[a] = 0;
        for (int t_ = 0; t_ < nzq; t_++) Qp[Tj[t_] + 1]++;
        for (int a = 0; a < nc; a++) Qp[a + 1] += Qp[a];
        int *fill_ = (int *)malloc(sizeof(int) * ((size_t)nc + 1)); memcpy(fill_, Qp, sizeof(int) * (size_t)nc);
        for (int l = 0; l < nr; l++) for (int t_ = Tp[l]; t_ < Tp[l + 1]; t_++) { const int w = fill_[Tj[t_]]++; Qi[w] = l; Qx[w] = Tx[t_]; }
        free(fill_); free(Tp); free(Tj); free(Tx); }
    for (int l = 0; l < nr; l++) bq[l] = P->b[rrep[l]] * sqrt((double)rsz[l]);
    Q.m = nr; Q.n = nc; Q.Ap = Qp; Q.Ai = Qi; Q.Ax = Qx; Q.b = bq; Q.c = cq; Q.nf = nfq; Q.nl = nlq;
    Q.nq = P->nq; Q.q = qq; Q.nr = P->nr; Q.r = qq + P->nq;
    if (opt && opt->verbose > 0) socp_printf("folding by symmetry: %d rows -> %d, %d variables -> %d, nonzeros %d -> %d (%.2fs)\n", m, nr, n, nc, P->Ap[n], Qp[nc], sc_time() - t0);
    SNOTE("the problem folded by its symmetry: %d rows -> %d, %d variables -> %d (BRISK_CONENOFOLD=1: not folded)", m, nr, n, nc);
    SocpOpts o2; if (opt) o2 = *opt; else socp_default_opts(&o2);
    if (o2.timelimit > 0) o2.timelimit = fmax(1e-3, o2.timelimit - (sc_time() - t0));
    SocpRes Rq; memset(&Rq, 0, sizeof Rq);
    const int st = inner(&Q, &o2, &Rq);
    *R = Rq; R->x = R->y = R->z = NULL;
    if (Rq.x && Rq.z) { R->x = (double *)malloc(sizeof(double) * (size_t)n); R->z = (double *)malloc(sizeof(double) * (size_t)n);
        for (int j = 0; j < n; j++) { const int k = ccl[j]; const double sk = sqrt((double)csz[k]); R->x[j] = Rq.x[pos[k]] / sk; R->z[j] = Rq.z[pos[k]] / sk; } }
    if (Rq.y) { R->y = (double *)malloc(sizeof(double) * (size_t)m); for (int i = 0; i < m; i++) R->y[i] = Rq.y[rcl[i]] / sqrt((double)rsz[rcl[i]]); }
    free(Rq.x); free(Rq.y); free(Rq.z);
    R->time = sc_time() - t0;
    free(ck); free(rcl); free(ccl); free(rsz); free(csz); free(crep); free(rrep); free(Bp); free(Bi); free(Bx); free(ord); free(pos);
    free(qq); free(Qp); free(Qi); free(Qx); free(bq); free(cq);
    *done = 1;
    return st;
}
static int socp_solve_body(const SocpProb *P, const SocpOpts *opt, SocpRes *R);
int socp_solve(const SocpProb *P, const SocpOpts *opt, SocpRes *R) {
    static int off = -1; if (off < 0) { const char *e_ = getenv("BRISK_CONENOFTZ"); off = e_ && atoi(e_) != 0; }   /* (a test: IEEE subnormals) */
    if (off) return socp_solve_body(P, opt, R);
#if defined(__x86_64__) || defined(__i386__)
    const unsigned int csr0 = _mm_getcsr();
#else
    const unsigned int csr0 = 0;
#endif
    cone_fast_fp();
#ifdef _OPENMP
    if (!omp_in_parallel()) {
        #pragma omp parallel
        cone_fast_fp();
    }
#endif
    blas_one_begin();
    const int st = socp_solve_body(P, opt, R);
    blas_one_end();
    cone_restore_fp(csr0);
    return st;
}
/* (27 October 2026, the SOCP developer; BRISK_CONENODUPROWS=1: off) presolve: rows that are multiples of another row, a_k = lambda a_i
 * with b_k = lambda b_i (and empty rows with b_k = 0), removed; their dual values are 0. Found by a hash of each row's pattern
 * and of its entries divided by its first entry, then compared entry by entry (to a relative 1e-12). A multiple whose right-hand
 * side disagrees is kept (the solver then finds the problem infeasible). Such rows come from models that state a constraint twice
 * (an equality written as two, the same bound from two places); the interior-point method itself is unharmed by them only up to
 * the regularization of a singular system, and the substitution of free variables, which needs a variable's rows to be few,
 * loses its pivots: the independent benchmark's copies with a fifth of the rows repeated, 10x and 8x slower (SOCP) */
static int socp_solve_dualform(const SocpProb *P, const SocpOpts *opt, SocpRes *R);
static int socp_fold_then_dual(const SocpProb *P, const SocpOpts *opt, SocpRes *R) {     /* (the folding by symmetry, then the choice of the dual form) */
    int folded = 0; const int st = socp_fold_try(P, opt, R, socp_solve_dualform, &folded);
    return folded ? st : socp_solve_dualform(P, opt, R);
}
/* (27 October 2026, the SOCP developer) rows that are linear combinations of other rows, found by eliminating the rows one after
 * another (shortest first) against the pivot rows kept so far (Andersen, "Finding all linearly dependent rows in large-scale
 * linear programming", 1995): a row is reduced by the pivot rows whose pivot columns it holds, in the order the pivot rows were
 * made (each pivot row is zero in the earlier pivot columns, so one pass in that order clears them all; a heap of pivot
 * indices); reduced to below 1e-10 of its largest entry it is dependent, and dropped (drop[r] = 2) when its right-hand side,
 * reduced the same way, is below 1e-9 of the sizes that entered it; otherwise it becomes a pivot row, its pivot the entry
 * with the fewest column entries among those at least a tenth of its largest. A row with a column that no other row holds is
 * independent and left out. The work and the stored entries are capped (a few times nnz(A)); past the cap the search stops,
 * the rows found so far stay found. Returns the number of rows dropped. */
static int dep_rows_find(int m, int n, const int *Rp, const int *Ri, const double *Rx, const double *b, const int *clen, unsigned char *drop) {
    int nr = 0; int *ord = (int *)malloc(sizeof(int) * ((size_t)m + 1));
    for (int i = 0; i < m; i++) {
        if (drop[i]) continue;
        int single = 0, len = 0;
        for (int q = Rp[i]; q < Rp[i + 1]; q++) if (Rx[q] != 0.0) { len++; if (clen[Ri[q]] == 1) single = 1; }
        if (!single && len > 0) ord[nr++] = i;
    }
    if (nr < 2) { free(ord); return 0; }
    /* shortest rows first (counting sort by length) */
    { int mx = 0; for (int t = 0; t < nr; t++) { const int l = Rp[ord[t] + 1] - Rp[ord[t]]; if (l > mx) mx = l; }
      int *cnt = (int *)calloc((size_t)mx + 2, sizeof(int)), *o2 = (int *)malloc(sizeof(int) * ((size_t)nr + 1));
      for (int t = 0; t < nr; t++) cnt[Rp[ord[t] + 1] - Rp[ord[t]] + 1]++;
      for (int l = 0; l <= mx; l++) cnt[l + 1] += cnt[l];
      for (int t = 0; t < nr; t++) o2[cnt[Rp[ord[t] + 1] - Rp[ord[t]]]++] = ord[t];
      free(ord); ord = o2; free(cnt); }
    const long nz = Rp[m];
    const double capw = (double)nz + 2e6; const long caps = nz + 1000000;     /* (the work counts the rows' loads, the eliminations and the stored entries: at most the entries of A and 2e6 more, about 5 ms beyond a pass over A; with 1e7 more, a control SOCP of 238,000 entries spent 30 ms finding nothing) */
    double work = 0;
    long scap = nz + 1024, sused = 0; int *pc = (int *)malloc(sizeof(int) * (size_t)scap); double *pv = (double *)malloc(sizeof(double) * (size_t)scap);
    int npv = 0, pcap = 1024; long *ps = (long *)malloc(sizeof(long) * ((size_t)pcap + 1)); int *pcol = (int *)malloc(sizeof(int) * (size_t)pcap); double *pdiag = (double *)malloc(sizeof(double) * (size_t)pcap), *pb = (double *)malloc(sizeof(double) * (size_t)pcap);
    ps[0] = 0;
    int *colpiv = (int *)malloc(sizeof(int) * ((size_t)n + 1)); for (int j = 0; j < n; j++) colpiv[j] = -1;
    double *acc = (double *)calloc((size_t)n + 1, sizeof(double)); int *cmk = (int *)calloc((size_t)n + 1, sizeof(int)), *tl = (int *)malloc(sizeof(int) * ((size_t)n + 1));
    int *pmk = (int *)calloc((size_t)pcap, sizeof(int)), *heap = (int *)malloc(sizeof(int) * (size_t)pcap); int stamp = 0, nd = 0;
    #define HPUSH(k_) do { int h_ = nh++; heap[h_] = (k_); while (h_ > 0 && heap[(h_ - 1) / 2] > heap[h_]) { const int t_ = heap[h_]; heap[h_] = heap[(h_ - 1) / 2]; heap[(h_ - 1) / 2] = t_; h_ = (h_ - 1) / 2; } } while (0)
    for (int t = 0; t < nr && work < capw && sused < caps; t++) {
        const int r = ord[t]; stamp++; int nt = 0, nh = 0; double rmax = 0, bred = b[r], bsz = fabs(b[r]);
        work += (double)(Rp[r + 1] - Rp[r]);
        for (int q = Rp[r]; q < Rp[r + 1]; q++) {
            if (Rx[q] == 0.0) continue;
            const int c = Ri[q];
            if (cmk[c] != stamp) { cmk[c] = stamp; tl[nt++] = c; acc[c] = 0.0; }
            acc[c] += Rx[q]; rmax = fmax(rmax, fabs(Rx[q]));
            const int k = colpiv[c]; if (k >= 0 && pmk[k] != stamp) { pmk[k] = stamp; HPUSH(k); }
        }
        while (nh > 0) {
            const int k = heap[0]; heap[0] = heap[--nh];
            { int h_ = 0; for (;;) { const int a_ = 2 * h_ + 1, b_ = a_ + 1; int s_ = h_; if (a_ < nh && heap[a_] < heap[s_]) s_ = a_; if (b_ < nh && heap[b_] < heap[s_]) s_ = b_; if (s_ == h_) break; const int t_ = heap[h_]; heap[h_] = heap[s_]; heap[s_] = t_; h_ = s_; } }
            const double v = acc[pcol[k]]; if (v == 0.0) continue;
            const double mu = v / pdiag[k];
            for (long e = ps[k]; e < ps[k + 1]; e++) {
                const int c = pc[e];
                if (cmk[c] != stamp) { cmk[c] = stamp; tl[nt++] = c; acc[c] = 0.0; }
                acc[c] -= mu * pv[e];
                const int k2 = colpiv[c]; if (k2 >= 0 && k2 != k && pmk[k2] != stamp) { pmk[k2] = stamp; HPUSH(k2); }
            }
            acc[pcol[k]] = 0.0;
            bred -= mu * pb[k]; bsz += fabs(mu * pb[k]);
            work += (double)(ps[k + 1] - ps[k]);
        }
        double amx = 0; for (int u = 0; u < nt; u++) amx = fmax(amx, fabs(acc[tl[u]]));
        { static int dbr = -2; if (dbr == -2) dbr = getenv("BRISK_DEPDBG") ? atoi(getenv("BRISK_DEPDBG")) : -1;
          if (dbr >= 0 && r >= dbr && amx > 1e-10 * rmax) { fprintf(stderr, "  [dep: row %d of %d entries kept: reduced to %.3g of %.3g, %d entries]\n", r, Rp[r + 1] - Rp[r], amx, rmax, nt); } }
        if (amx <= 1e-10 * rmax) {
            if (fabs(bred) <= 1e-9 * (bsz + 1e-300)) { drop[r] = 2; nd++; }
            else if (getenv("BRISK_DEPDBG")) fprintf(stderr, "  [dep: row %d dependent, right-hand side reduced to %.3g of %.3g]\n", r, bred, bsz);
            continue;
        }
        /* a pivot row: its pivot the entry with the fewest column entries among those >= a tenth of its largest */
        int best = -1;
        for (int u = 0; u < nt; u++) { const int c = tl[u]; if (fabs(acc[c]) >= 0.1 * amx && (best < 0 || clen[c] < clen[best])) best = c; }
        if (npv == pcap) {
            pcap *= 2; ps = (long *)realloc(ps, sizeof(long) * ((size_t)pcap + 1)); pcol = (int *)realloc(pcol, sizeof(int) * (size_t)pcap);
            pdiag = (double *)realloc(pdiag, sizeof(double) * (size_t)pcap); pb = (double *)realloc(pb, sizeof(double) * (size_t)pcap);
            pmk = (int *)realloc(pmk, sizeof(int) * (size_t)pcap); for (int k = npv; k < pcap; k++) pmk[k] = 0; heap = (int *)realloc(heap, sizeof(int) * (size_t)pcap);
        }
        if (sused + nt > scap) { scap = 2 * (sused + nt) + 1024; pc = (int *)realloc(pc, sizeof(int) * (size_t)scap); pv = (double *)realloc(pv, sizeof(double) * (size_t)scap); }
        for (int u = 0; u < nt; u++) { const int c = tl[u]; if (fabs(acc[c]) > 1e-15 * amx) { pc[sused] = c; pv[sused++] = acc[c]; } }
        work += (double)nt;
        pcol[npv] = best; pdiag[npv] = acc[best]; pb[npv] = bred; colpiv[best] = npv; ps[npv + 1] = sused; npv++;
    }
    #undef HPUSH
    if (getenv("BRISK_DEPDBG")) fprintf(stderr, "  [dep: %d pivot rows, %ld entries stored (cap %ld), work %.3g (cap %.3g), %d of %d rows found]\n", npv, sused, caps, work, capw, nd, nr);
    free(ord); free(pc); free(pv); free(ps); free(pcol); free(pdiag); free(pb); free(colpiv); free(acc); free(cmk); free(tl); free(pmk); free(heap);
    return nd;
}
static int socp_solve_nodup(const SocpProb *P, const SocpOpts *opt, SocpRes *R) {
    const int m = P->m, n = P->n;
    static int off = -1; if (off < 0) off = getenv("BRISK_CONENODUPROWS") != NULL;
    if (off || m < 2) return socp_fold_then_dual(P, opt, R);
    const int nz = P->Ap[n];
    int *Rp = (int *)calloc((size_t)m + 2, sizeof(int)), *Ri = (int *)malloc(sizeof(int) * ((size_t)nz + 1));
    double *Rx = (double *)malloc(sizeof(double) * ((size_t)nz + 1));
    for (int p = 0; p < nz; p++) Rp[P->Ai[p] + 2]++;
    for (int i = 0; i < m; i++) Rp[i + 2] += Rp[i + 1];
    for (int j = 0; j < n; j++) for (int p = P->Ap[j]; p < P->Ap[j + 1]; p++) { const int q = Rp[P->Ai[p] + 1]++; Ri[q] = j; Rx[q] = P->Ax[p]; }
    /* (rows in column order; zero entries are skipped in the comparisons through the hash of the pattern of nonzeros) */
    HashCol *hr = (HashCol *)malloc(sizeof(HashCol) * ((size_t)m + 1));
    for (int i = 0; i < m; i++) {
        unsigned long long h = 1469598103934665603ULL; double v0 = 0; int len = 0;
        for (int q = Rp[i]; q < Rp[i + 1]; q++) {
            if (Rx[q] == 0.0) continue;
            if (v0 == 0.0) v0 = Rx[q];
            h = (h ^ (unsigned long long)(unsigned)Ri[q]) * 1099511628211ULL;
            if (len < 3) { const float r_ = (float)(Rx[q] / v0); unsigned int u_; memcpy(&u_, &r_, 4); h = (h ^ (u_ >> 4)) * 1099511628211ULL; }
            len++;
        }
        hr[i].h = h; hr[i].j = i;
    }
    {   /* (grouped by the hash in O(m), not sorted: a table of the distinct hashes, the rows of each group in their order) */
        size_t tsz = 1024; while (tsz < 2 * (size_t)m) tsz <<= 1;
        int *slot = (int *)malloc(sizeof(int) * tsz); for (size_t t_ = 0; t_ < tsz; t_++) slot[t_] = -1;
        int *grp = (int *)malloc(sizeof(int) * ((size_t)m + 1)), *gcnt = (int *)calloc((size_t)m + 2, sizeof(int)); int ng = 0;
        unsigned long long *gh = (unsigned long long *)malloc(sizeof(unsigned long long) * ((size_t)m + 1));
        for (int i = 0; i < m; i++) {
            const unsigned long long h = hr[i].h; size_t t_ = (size_t)((h * 0x9E3779B97F4A7C15ULL) >> 17) & (tsz - 1);
            while (slot[t_] >= 0 && gh[slot[t_]] != h) t_ = (t_ + 1) & (tsz - 1);
            if (slot[t_] < 0) { slot[t_] = ng; gh[ng++] = h; }
            grp[i] = slot[t_]; gcnt[slot[t_] + 1]++;
        }
        for (int g = 0; g < ng; g++) gcnt[g + 1] += gcnt[g];
        HashCol *h2 = (HashCol *)malloc(sizeof(HashCol) * ((size_t)m + 1));
        for (int i = 0; i < m; i++) h2[gcnt[grp[i]]++] = hr[i];
        free(hr); hr = h2; free(slot); free(grp); free(gcnt); free(gh);
    }
    unsigned char *drop = (unsigned char *)calloc((size_t)m + 1, 1); int nd = 0;
    for (int a = 0; a < m; ) {
        int e = a + 1; while (e < m && hr[e].h == hr[a].h) e++;
        for (int u = a; u < e && u < a + 64; u++) {
            const int i = hr[u].j; if (drop[i]) continue;
            for (int v = u + 1; v < e && v < a + 64; v++) {
                const int k = hr[v].j; if (drop[k]) continue;
                int qi = Rp[i], qk = Rp[k], ok = 1; double lam = 0, amx = 0;
                for (;;) {
                    while (qi < Rp[i + 1] && Rx[qi] == 0.0) qi++;
                    while (qk < Rp[k + 1] && Rx[qk] == 0.0) qk++;
                    if (qi == Rp[i + 1] || qk == Rp[k + 1]) { ok = qi == Rp[i + 1] && qk == Rp[k + 1]; break; }
                    if (Ri[qi] != Ri[qk]) { ok = 0; break; }
                    if (lam == 0.0) lam = Rx[qk] / Rx[qi];
                    else if (fabs(Rx[qk] - lam * Rx[qi]) > 1e-12 * fabs(Rx[qk])) { ok = 0; break; }
                    amx = fmax(amx, fabs(Rx[qk]));
                    qi++; qk++;
                }
                if (!ok) continue;
                if (lam == 0.0) { if (P->b[k] == 0.0) { drop[k] = 1; nd++; } continue; }     /* (both empty) */
                if (fabs(P->b[k] - lam * P->b[i]) <= 1e-12 * (fabs(P->b[k]) + amx)) { drop[k] = 1; nd++; }
            }
        }
        a = e;
    }
    int ndep = 0;
    { static int dep = -1; if (dep < 0) dep = getenv("BRISK_CONENODEPROWS") == NULL;
      if (dep) {
          const double t0 = sc_time();
          int *clen = (int *)calloc((size_t)n + 1, sizeof(int));
          for (int j = 0; j < n; j++) for (int p = P->Ap[j]; p < P->Ap[j + 1]; p++) if (P->Ax[p] != 0.0 && !drop[P->Ai[p]]) clen[j]++;
          ndep = dep_rows_find(m, n, Rp, Ri, Rx, P->b, clen, drop);
          free(clen);
          if (dbg_on()) socp_printf("  [linearly dependent rows: %d found (%.3fs)]\n", ndep, sc_time() - t0);
          if (getenv("BRISK_DEPSCANX")) { printf("depscanx m %d n %d nnz %d multiples %d dependent %d time %.4f\n", m, n, nz, nd, ndep, sc_time() - t0); fflush(stdout); exit(0); }   /* (experiments: the search alone) */
      } }
    free(hr); free(Rp); free(Ri); free(Rx);
    if (nd + ndep == 0) { free(drop); return socp_fold_then_dual(P, opt, R); }
    int *rmap = (int *)malloc(sizeof(int) * ((size_t)m + 1)); int m2 = 0;
    for (int i = 0; i < m; i++) rmap[i] = drop[i] ? -1 : m2++;
    int *Ap = (int *)malloc(sizeof(int) * ((size_t)n + 1)), *Ai = (int *)malloc(sizeof(int) * ((size_t)nz + 1));
    double *Ax = (double *)malloc(sizeof(double) * ((size_t)nz + 1)), *b2 = (double *)malloc(sizeof(double) * ((size_t)m2 + 1));
    int w = 0;
    for (int j = 0; j < n; j++) { Ap[j] = w; for (int p = P->Ap[j]; p < P->Ap[j + 1]; p++) if (rmap[P->Ai[p]] >= 0) { Ai[w] = rmap[P->Ai[p]]; Ax[w++] = P->Ax[p]; } }
    Ap[n] = w;
    for (int i = 0; i < m; i++) if (rmap[i] >= 0) b2[rmap[i]] = P->b[i];
    SocpProb Q = *P; Q.m = m2; Q.Ap = Ap; Q.Ai = Ai; Q.Ax = Ax; Q.b = b2;
    if (opt && opt->verbose > 0 && nd > 0) socp_printf("presolve: %d rows that are multiples of another row removed (rows %d -> %d)\n", nd, m, m - nd);
    if (opt && opt->verbose > 0 && ndep > 0) socp_printf("presolve: %d linearly dependent rows removed (rows %d -> %d)\n", ndep, m - nd, m2);
    if (nd > 0) SNOTE("presolve: %d rows that are multiples of another row removed (their dual values 0)", nd);
    if (ndep > 0) SNOTE("presolve: %d linearly dependent rows removed (their dual values 0)", ndep);
    const int st = socp_fold_then_dual(&Q, opt, R);
    if (R->y) {
        double *y = (double *)xcalloc((size_t)m + 1, sizeof(double));
        for (int i = 0; i < m; i++) if (rmap[i] >= 0) y[i] = R->y[rmap[i]];
        free(R->y); R->y = y;
    }
    free(rmap); free(drop); free(Ap); free(Ai); free(Ax); free(b2);
    return st;
}
static int socp_solve_body(const SocpProb *P, const SocpOpts *opt, SocpRes *R) {
    memset(R, 0, sizeof(*R));      /* (every path's result starts empty: cx, cy, cz NULL unless kept) */
#ifdef _OPENMP
    /* (18 October 2026, LP/SOCP: the busy-machine rule of the other engines, solver.c: only the free cores. With one
     * core taken by another process, the truss design problem's products with A, parallel regions of 0.1 ms, took
     * 2.55 s instead of 0.05 s: every region waits for the thread without a core) */
    const int ntb = omp_in_parallel() ? -1 : brisk_threads_busy(0);
    const int nt = omp_get_max_threads();
    static int treepar = -1; if (treepar < 0) { const char *e_ = getenv("BRISK_CONETREEPAR"); treepar = e_ ? atoi(e_) : 0; }   /* (a test: the factorization's tree-parallel path on the threads) */
    if (nt > 1 && !treepar) omp_set_num_threads(1);
    { const char *e = getenv("BRISK_CONEPANEL"); if (!(e && atoi(e) == 0) && !omp_in_parallel()) schol_set_panel_threads(nt); }
    g_cone_nt = nt; g_cone_nt_ldl = nt;
    int st;
    if (opt && (opt->wx || opt->wsave > 0)) st = socp_solve_warm(P, opt, R);
    else st = socp_solve_nodup(P, opt, R);      /* (rows that are multiples of others removed, then the folding by symmetry and the dual form) */
    g_cone_nt = 1; g_cone_nt_ldl = 1;
    schol_set_panel_threads(0);
    if (nt > 1) omp_set_num_threads(nt);
    if (ntb > 0) omp_set_num_threads(ntb);
    return st;
#else
    if (opt && (opt->wx || opt->wsave > 0)) return socp_solve_warm(P, opt, R);
    return socp_solve_nodup(P, opt, R);
#endif
}
static int socp_solve_seq(const SocpProb *P, const SocpOpts *opt, SocpRes *R) {
    const double t0 = sc_time();
    SocpOpts o1; if (opt) o1 = *opt; else socp_default_opts(&o1);
    const int retry = !(o1.step > 0) && !getenv("BRISK_CONENORETRY");
    o1.attempt = retry ? 1 : 0;
    int st = socp_solve_pre(P, &o1, R);
    if (!retry || (st != SOCP_REDUCED && st != SOCP_NUMERR) || *socp_stop || (socp_stop_call && *socp_stop_call)) return st;
    if (o1.noretry && st == SOCP_REDUCED) return st;      /* (the caller recovers a point near the tolerance; a failed attempt is repeated) */
    SocpOpts o2 = o1; o2.step = 0.8; o2.attempt = 2;
    int second = 0;
    if (!(o1.timelimit > 0) || sc_time() - t0 < 0.4 * o1.timelimit) {
        if (o1.timelimit > 0) o2.timelimit = o1.timelimit - (sc_time() - t0);
        SocpRes R2; memset(&R2, 0, sizeof R2);
        const int st2 = socp_solve_pre(P, &o2, &R2);
        const int it = R->iters + R2.iters;
        const int better = st2 == SOCP_OPTIMAL || st2 == SOCP_PINF || st2 == SOCP_DINF || ((st2 == SOCP_REDUCED || st2 == SOCP_NUMERR) && R2.maxerr < R->maxerr);
        if (better) { socp_result_free(R); *R = R2; st = st2; second = 1; } else socp_result_free(&R2);
        SNOTE("second attempt with shorter steps (0.8 of the way to the boundary): %s", second ? "its result is returned" : "the first result stands");
        R->iters = it;
    }
    R->time = sc_time() - t0;
    if (o1.verbose > 0) {
        const int cert = st == SOCP_PINF || st == SOCP_DINF;
        socp_printf("%sstatus: %s", g_stat_prefix, socp_status_line(st));
        if (!cert) socp_printf(" (max rel error %.1e)", R->maxerr);
        socp_printf("   iterations: %d   time: %.3fs%s\n", R->iters, R->time, second ? "   (second attempt)" : "");
        if (!cert) socp_printf("  primal obj c'x = %+.12e   dual obj b'y = %+.12e\n  errors: pinf %.1e  dinf %.1e  gap %.1e  x'z %.1e\n", R->pobj, R->dobj, R->err[0], R->err[1], R->err[2], R->err[3]);
    }
    return st;
}
