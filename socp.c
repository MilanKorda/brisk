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
#include <limits.h>
#include <setjmp.h>
#include <string.h>
#include <math.h>
#if defined(__x86_64__) || defined(__i386__)
#include <xmmintrin.h>
#include <pmmintrin.h>
#endif
#include <time.h>
#include "socp.h"
const char *brisk_threads_label(int nt);     /* (solver.c, 31 October 2026: the reason after "Number of threads: k") */
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
void   schol_set_tinyfloor(double f);
void   schol_set_tinyflags(SChol *S, int on);
int    schol_tiny_rows(const SChol *S, int *rows, int kmax);
int    schol_factor(SChol *S, double shift);
void   schol_solve(const SChol *S, double *b, double *work);
size_t schol_nnz(const SChol *S);
int    schol_set_amd(int mode);
const int *schol_perm(const SChol *S);
void schol_set_perm_only(int on);
int schol_exact_counts(int m, const int *deg, int *const *nbr, const int *P, double *flops_out, double *nnz_out);
int schol_amd_estimate(int m, const int *deg, int *const *nbr, double *flops_out, double *nnz_out);
void schol_set_dynpiv(SChol *S, double p);
void   schol_set_amd_dense(double d);
void   schol_set_solve_seq(int on);
void   schol_set_solve_nt(int nt);
void   schol_set_panel_threads(int nt);
int    brisk_team_begin(int nt);
int    brisk_team_adapt(int cur);
void   brisk_team_end(void);
void   schol_set_perm(const int *perm);
void   schol_set_flopcap(double cap);
void   schol_set_oom_jmp(void *j); void *schol_oom_jmp_get(void); void schol_oom(void);
int    schol_amd_estimate_cap(int m, const int *deg, int *const *nbr, const volatile double *cap, double *flops_out, double *nnz_out);
void   schol_zero(SChol *S);
void   schol_set_inplace(SChol *S, int on);
int    schol_solve_part(SChol *S, double *b, double *work, int nt);
/* (23 October 2026, the SOCP developer) the supernodal solve by independent subtrees (sparsechol.c schol_solve_part: the same
 * bits at any thread count); BRISK_CONESOLVEPART=1 */
/* (23 October 2026, the SOCP developer) the triangular solves of the supernodal factor by independent subtrees
 * (schol_solve_part), by default: the same bits at any thread count, one thread inside an outer parallel region; on the
 * 47 SOCP files where the partition applies (the top of the tree at most half of the solve's work) 0.94 of the total time
 * at two threads, 1.007 at one. BRISK_CONESOLVEPART=0: the sequential solve */
/* (10 November 2026, the SOCP developer, a test: BRISK_CONESOLVETP=1) the solves with the factor by the task-parallel solve instead of
 * the sequential supernode loops */
static int cone_seqsolve(void) { static int v = -1; if (v < 0) { const char *e_ = getenv("BRISK_CONESOLVETP"); v = e_ && atoi(e_) > 0 ? 0 : 1; } return v; }
static int cone_part(void) { static int v = -1; if (v < 0) { const char *e_ = getenv("BRISK_CONESOLVEPART"); v = e_ ? atoi(e_) : 1; } return v; }
double schol_flops(const SChol *S);
int    schol_ntiny(const SChol *S);

/* printing and interruption are the caller's: socp_printf (default: printf), *socp_stop (default: never) */
int (*socp_printf)(const char *fmt, ...) = printf;
void brisk_itab_row(int it, double pobj, double dobj, double pinf, double dinf, double gap, double mu, double step);   /* (main.c) */
void brisk_itab_end(void);
void (*socp_note)(const char *fmt, ...) = NULL;    /* 5.8: a line for the caller's summary at the end of the run (brisk_note in main.c) */
#define SNOTE(...) do { if (socp_note) socp_note(__VA_ARGS__); } while (0)
void (*socp_note_diag)(const char *fmt, ...) = NULL;   /* (31 October 2026, ISSUES 725) a note of the solver's internal steps: the Summary of -v only (brisk_note_diag) */
#define SNOTED(...) do { if (socp_note_diag) socp_note_diag(__VA_ARGS__); else if (socp_note) socp_note(__VA_ARGS__); } while (0)
/* (31 October 2026, the SOCP developer; ISSUES 725, the owner: "only the things that can be reasonably understood by somebody
 * not having an insight into the internal workings of the solver stay at the default option") the default log of the SOCP
 * route is the user's: while g_log_brief is set (main.c), a line printed to stdout appears only inside log_user; the user's
 * lines of the SOCP solver are the presolve steps ("Presolve:"), the thread count, the method in one line, one row per
 * iteration (brisk_log_row: objectives, errors, gap, mu, step, time) and the messages that end a solve early; the rest is
 * the diagnosis (-v) */
extern int g_log_brief, g_log_user;
void brisk_log_head(int ipm);
void brisk_log_row(int it, double po, double dob, double pinf, double dinf, double gap, double mu, double step, double t);
#define SUSER(...) do { g_log_user++; socp_printf(__VA_ARGS__); g_log_user--; } while (0)
static _Thread_local int g_user_pre = 0, g_user_head = 0, g_user_meth = 0;   /* (g_user_meth: the threads and the method printed once a log; g_user_head: the table's head once a solve) */
/* the rows of the default log show the problem as read: the steps that change the objective (the free variables'
 * substitution: both objectives shifted by a constant; the dual form: the primal and dual objectives exchanged and negated
 * about a constant) compose the map outer = a * inner + k, the two objectives exchanged when swap is set */
static _Thread_local double g_row_a = 1.0, g_row_k = 0.0; static _Thread_local int g_row_swap = 0;
int socp_threads_used = 0;    /* (the threads the last solve's factorization used: the Summary's line) */
void socp_log_reset(void) { g_user_pre = 0; g_user_head = 0; g_user_meth = 0; g_row_a = 1.0; g_row_k = 0.0; g_row_swap = 0; socp_threads_used = 0; }
void socp_log_method_shown(void) { g_user_meth = 1; }   /* (a caller that printed its own method line: the QCQP route) */
void socp_log_rowmap(double a, double k) { g_row_a = a; g_row_k = k; g_row_swap = 0; }
void socp_log_rowmap_swap(double a, double k) { g_row_a = a; g_row_k = k; g_row_swap = 1; }   /* (the caller's problem is the dual of the one solved) */   /* (the caller's map: a file maximized, a constant term) */
static void socp_user_row(int it, double po, double dob, double ep, double ed, double eg, double mu, double step, double t) {
    if (g_row_swap) { const double a = po; po = dob; dob = a; const double b = ep; ep = ed; ed = b; }
    brisk_log_row(it, g_row_a * po + g_row_k, g_row_a * dob + g_row_k, ep, ed, eg, mu, step, t);
}   /* (the presolve lines and the table's head printed in this run of the log; socp_log_reset) */
void socp_log_reset(void);
/* a presolve step of the default log ("Presolve: <first>", the others below it) */
#include <stdarg.h>
static void socp_user_presolve(const char *fmt, ...) {
    if (!g_log_brief) return;
    char t[400]; va_list ap; va_start(ap, fmt); vsnprintf(t, sizeof t, fmt, ap); va_end(ap);
    SUSER("%s%s\n", g_user_pre++ ? "          " : "Presolve: ", t);
}
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
static _Thread_local long g_part_try = 0, g_part_ok = 0;     /* (debug: the partitioned solves tried and done) */
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
    double dyn_delta, pivfloor; int flaux0, flaux1; int ndyn; int pivdbg, pivnf, pivn, pivnaux; double pivmin[4]; long pivcnt[4];
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
    if (!F->Li || !F->Lx) { fprintf(stderr, "brisk: out of memory (the column factorization, %.1f GB)\n", 12.0 * (double)F->lnz / 1e9); schol_oom(); }
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
        if (getenv("BRISK_CONEDDTOPDBG")) {     /* (3 November 2026, debug: the top rows' work by the kind of column that updates them) */
            int *cnt = (int *)calloc((size_t)n, sizeof(int)), *stk = (int *)malloc(sizeof(int) * (size_t)n); unsigned char *mk = (unsigned char *)calloc((size_t)n, 1);
            double wp = 0, wt = 0, wpart = 0; long npairs_p = 0;
            for (int k = 0; k < n; k++) {
                int ns = 0;
                for (int p = F->Ap[k]; p < F->Ap[k + 1]; p++) { int j = F->Ai[p]; if (j >= k) continue; while (j != -1 && j < k && !mk[j]) { mk[j] = 1; stk[ns++] = j; j = F->etree[j]; } }
                for (int s = 0; s < ns; s++) { const int c = stk[s]; mk[c] = 0;
                    if (F->ptpos[k] >= 0) { if (F->ptpos[c] < 0) { wp += cnt[c]; npairs_p++; } else wt += cnt[c]; } else wpart += cnt[c];
                    cnt[c]++; }
            }
            socp_printf("  [the top rows' updates: %.3g from the parts' columns (%ld column visits), %.3g from the top's columns; the parts' rows %.3g]\n", wp, npairs_p, wt, wpart);
            free(cnt); free(stk); free(mk);
        }
        if (getenv("BRISK_CONETOPDBG")) {     /* (12 November 2026, debug: the parallelism within the top: its subtree's longest chain of work
                                               * against its total, the row work counted as in the shares above (each column's entries)) */
            double *cw = (double *)calloc((size_t)n, sizeof(double)), tw = 0, cp = 0; int nroot = 0, nleaf = 0; unsigned char *hasc = (unsigned char *)calloc((size_t)n, 1);
            for (int j = 0; j < n; j++) if (F->ptpos[j] >= 0) { const double w = (double)F->Lnz[j] * F->Lnz[j] + F->Lnz[j] + 1; cw[j] += w; tw += w;
                const int pj = F->etree[j]; if (pj >= 0 && F->ptpos[pj] >= 0) { if (cw[j] > cw[pj]) cw[pj] = cw[j]; hasc[pj] = 1; } else { nroot++; if (cw[j] > cp) cp = cw[j]; } }
            for (int j = 0; j < n; j++) if (F->ptpos[j] >= 0 && !hasc[j]) nleaf++;
            socp_printf("  [the top's own subtree: %d nodes, %d roots, %d leaves; its longest chain %.1f %% of its work]\n", ntop, nroot, nleaf, 100.0 * cp / fmax(tw, 1.0));
            free(cw); free(hasc);
        }
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
    double fw_all = 0, fw_top = 0, fw_tt = 0, sv_all = 0, sv_par = 0;
    for (int j = 0; j < F->n; j++) {
        int p = F->Lp[j];
        if (F->ptpos[j] < 0) while (p < F->Lp[j + 1] && F->ptpos[F->Li[p]] < 0) p++;
        F->ptsplit[j] = p;
        for (int q = F->Lp[j]; q < F->Lp[j + 1]; q++) { const double w = (double)(q - F->Lp[j]) + 1.0; fw_all += w; if (F->ptpos[F->Li[q]] >= 0) { fw_top += w; if (F->ptpos[j] >= 0) fw_tt += w; } }
        sv_all += F->Lp[j + 1] - F->Lp[j]; sv_par += p - F->Lp[j];
    }
    F->pt_fac = fw_all > 0 && fw_top <= 0.85 * fw_all;     /* (the factorization's parts cost nothing extra: joint_FC_5, 72.6 % in the top, 31.6 -> 25.0 s) */ F->pt_sol = sv_all > 0 && sv_par >= 0.5 * sv_all;
    if (getenv("BRISK_CONEDBG")) socp_printf("  [the column factorization's parts: %.1f %% of its work in the top rows (%.1f %% from top columns)%s, %.1f %% of the forward solve in the parts%s]\n", 100.0 * fw_top / fmax(fw_all, 1.0), 100.0 * fw_tt / fmax(fw_all, 1.0), F->pt_fac ? "" : " (the factorization sequential)", 100.0 * sv_par / fmax(sv_all, 1.0), F->pt_sol ? "" : " (the solves sequential)");
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
#define LDL_PIVFLOOR(F_, k_) ((F_)->pivfloor > 0 && !((F_)->perm[k_] >= (F_)->flaux0 && (F_)->perm[k_] < (F_)->flaux1) ? (F_)->pivfloor : 0.0)
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
    if (!(F->sign[k] * dk > LDL_PIVFLOOR(F, k))) {   /* (BRISK_CONEPIVDBG=1: each replaced pivot printed, 22 October 2026; the floor: BRISK_CONEQDBIG, below) */
        if (getenv("BRISK_CONEPIVDBG")) { const int o = F->perm[k]; double d0 = 0; for (int p = F->Ap[k]; p < F->Ap[k + 1]; p++) if (F->Ai[p] == k) d0 = F->Ax[p];
            socp_printf("        [pivot %d (original %d, %s, sign %+d): %.3e, its diagonal %.3e]\n", k, o, o < F->pivnf ? "free" : o < F->pivn ? "cone" : o < F->pivn + F->pivnaux ? "aux" : (F->sign[k] < 0 ? "linear" : "row"), F->sign[k], dk, d0); }
        { static double bigp = -1; if (bigp < 0) { const char *e_ = getenv("BRISK_CONEBIGPIV"); bigp = e_ ? atof(e_) : 0.0; }   /* (a test, 3 November 2026: a huge replacement decouples the unknown instead of amplifying the rows after it) */
          dk = F->sign[k] * (bigp > 0 ? bigp : F->dyn_delta); } ndyn++; }       /* a pivot of the wrong sign: replaced (the caller turns to double-double) */
    F->D[k] = dk; F->Dinv[k] = 1.0 / dk;
    if (F->pivdbg) {     /* (the debug log's pivot statistics; under a lock, so that they no longer make the factorization sequential, 12 November 2026) */
        const int o = F->perm[k]; const int cl = o < F->pivnf ? 0 : o < F->pivn ? 1 : o < F->pivn + F->pivnaux ? 2 : 3;
        #pragma omp critical(ldl_pivdbg)
        { if (fabs(dk) < F->pivmin[cl]) F->pivmin[cl] = fabs(dk); if (fabs(dk) < 1e-6) F->pivcnt[cl]++; } }
    return ndyn;
}
static int ldl_top_d(Ldl *F, int *next, int nt);
static int ldl_numeric(Ldl *F) {
    const int n = F->n;
    ldl_alloc(F);
    int *yIdx = F->iw, *elim = F->iw + n, *next = F->iw + 2 * n;
    unsigned char *mark = F->bw;
    double *y = F->fw;
    F->ndyn = 0;
    for (int i = 0; i < n; i++) { mark[i] = 0; y[i] = 0.0; next[i] = F->Lp[i]; }
    if (!F->pt_built) ldl_part_build(F);
    const int nt = F->npart > 0 && F->ptsplit && F->pt_fac ? ldl_part_nt(F) : 1;     /* (the first factorization sequential: the shares need its pattern) */
    static int tpard = -1; if (tpard < 0) { const char *e_ = getenv("BRISK_CONETOPPAR"); tpard = e_ ? atoi(e_) : 0; }
    const int tp_ = tpard && F->npart > 0 && F->ptsplit && F->pt_fac;     /* (at every number of threads: the same bits) */
    if (nt > 1 || tp_) {
        int nd[LDL_NPART] = { 0 };
        #pragma omp parallel for schedule(dynamic, 1) num_threads(nt) if (nt > 1)
        for (int q = 0; q < F->npart; q++) {
            const int tid = omp_get_thread_num();
            int *yI = tid ? F->wsi[tid - 1] : yIdx, *el = tid ? F->wsi[tid - 1] + n : elim; unsigned char *mk = tid ? F->wsb[tid - 1] : mark; double *yy = tid ? F->wsf[tid - 1] : y;
            for (int a = F->pnp[q]; a < F->pnp[q + 1]; a++) nd[q] += ldl_row(F, F->pnl[a], yI, el, next, mk, yy);
        }
        for (int q = 0; q < F->npart; q++) F->ndyn += nd[q];
        int rt_ = -1; if (tp_) rt_ = ldl_top_d(F, next, nt);
        if (rt_ >= 0) F->ndyn += rt_;
        else for (int a = 0; a < F->ntop; a++) F->ndyn += ldl_row(F, F->ptop[a], yIdx, elim, next, mark, y);
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
/* (3 November 2026, the SOCP developer) the smallest pivot a factorization in double-double accepts: every diagonal block of
 * the factored matrix but the large cones' auxiliary unknowns carries a regularization of at least delta (the primal blocks
 * -H - delta, the rows +delta), and eliminating an unknown only pushes the remaining blocks further from zero (a primal
 * pivot adds a positive term to the rows, a row's pivot a negative one to the primal block), so every pivot of any order
 * has magnitude at least delta in exact arithmetic. A pivot below half of it is rounding, as one of the wrong sign is:
 * joint_FC_5 (SOCP), the small cones in their eigenbasis, double-double: a pivot of -1e-320 passed the sign test, its
 * inverse overflowed, and NaN spread through 2,794 later pivots. The auxiliary unknowns (diagonal +-1) are not bounded so */
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
    if (!(F->sign[k] * dk.h > LDL_PIVFLOOR(F, k))) {
        if (getenv("BRISK_CONEPIVDBG")) { const int o = F->perm[k]; double d0 = 0; for (int p = F->Ap[k]; p < F->Ap[k + 1]; p++) if (F->Ai[p] == k) d0 = F->Ax[p];
            socp_printf("        [dd pivot %d (original %d, %s, sign %+d): %.3e, its diagonal %.3e]\n", k, o, o < F->pivnf ? "free" : o < F->pivn ? "cone" : o < F->pivn + F->pivnaux ? "aux" : (F->sign[k] < 0 ? "linear" : "row"), F->sign[k], dk.h, d0); }
        dk.h = F->sign[k] * F->dyn_delta; dk.l = 0.0; ndyn++; }
    const sdd di = dd_inv(dk);
    F->D[k] = dk.h; F->Dl[k] = dk.l; F->Dinv[k] = di.h; F->Dinvl[k] = di.l;
    { static int nd_ = -1; if (nd_ < 0) nd_ = getenv("BRISK_CONENANDBG") != NULL;
      if (nd_ && !(isfinite(di.h) && isfinite(dk.h))) { static int cnt_ = 0; if (cnt_++ < 3) socp_printf("        [dd pivot %d: %.3e + %.3e, sign %d, floor %.1e, inverse %.3e]\n", k, dk.h, dk.l, F->sign[k], LDL_PIVFLOOR(F, k), di.h); } }
    return ndyn;
}
/* (3 November 2026, the SOCP developer) the top rows of the double-double factorization in three stages, the first two on
 * the threads. With A = [A_PP A_PT; A_TP A_TT] (P the parts' rows, done; T the top's), row k of the top is the sparse
 * solve with L_PP for its entries in the parts' columns (independent from row to row: stage 1), the sums
 * S(r, k) = sum_c L(r, c) D_c L(k, c) over the parts' columns for the top rows r < k (independent from row to row once
 * every top row's entries in the parts' columns are in place: stage 2, each row's sums by one thread in the order of its
 * columns, so the same bits on any number of threads), and the elimination of row k against the top's own columns, in
 * order (stage 3). db-joint-soerensen (SOCP): 3,772 top rows held 75 % of the factorization's updates, about half of them
 * from the parts' columns. Returns the replaced pivots. */
static int ldl_top_dd(Ldl *F, int *next, int nt) {
    const int n = F->n, T = F->ntop;
    const double tt0 = dbg_on() ? sc_time() : 0.0; double tt1 = 0, tt2 = 0;
    int *pcnt = (int *)calloc((size_t)T + 1, sizeof(int)), *tcnt = (int *)calloc((size_t)T + 1, sizeof(int));
    double *dkh = (double *)calloc((size_t)T + 1, sizeof(double)), *dkl = (double *)calloc((size_t)T + 1, sizeof(double));
    int *npc = (int *)malloc(sizeof(int) * (size_t)n);       /* (the parts' rows of each column: its entries before the top's) */
    if (!pcnt || !tcnt || !dkh || !dkl || !npc) { free(pcnt); free(tcnt); free(dkh); free(dkl); free(npc); return -1; }
    for (int j = 0; j < n; j++) npc[j] = next[j] - F->Lp[j];
    /* the reach of row k in the order of the sequential code (each entry's path in the elimination tree, then reversed) */
    #define TOP_REACH(k_, yI_, el_, mk_, nny_) do { nny_ = 0; \
        for (int p_ = F->Ap[k_]; p_ < F->Ap[k_ + 1]; p_++) { const int i_ = F->Ai[p_]; if (i_ == k_ || mk_[i_]) continue; \
            int ne_ = 0, j_ = i_; mk_[j_] = 1; el_[ne_++] = j_; j_ = F->etree[j_]; \
            while (j_ != -1 && j_ < k_ && !mk_[j_]) { mk_[j_] = 1; el_[ne_++] = j_; j_ = F->etree[j_]; } \
            while (ne_) yI_[nny_++] = el_[--ne_]; } } while (0)
    /* stage 0: the counts */
    #pragma omp parallel num_threads(nt)
    {
        const int tid = omp_get_thread_num();
        int *yI = tid ? F->wsi[tid - 1] : F->iw, *el = tid ? F->wsi[tid - 1] + n : F->iw + n; unsigned char *mk = tid ? F->wsb[tid - 1] : F->bw;
        #pragma omp for schedule(dynamic, 16)
        for (int a = 0; a < T; a++) {
            const int k = F->ptop[a]; int nny; TOP_REACH(k, yI, el, mk, nny);
            int cp = 0, ct = 0; for (int t = 0; t < nny; t++) { const int c = yI[t]; mk[c] = 0; if (F->ptpos[c] < 0) cp++; else ct++; }
            pcnt[a + 1] = cp; tcnt[a + 1] = ct;
        }
    }
    for (int a = 0; a < T; a++) { pcnt[a + 1] += pcnt[a]; tcnt[a + 1] += tcnt[a]; }
    const long NP = pcnt[T], NT = tcnt[T];
    int *pc = (int *)malloc(sizeof(int) * ((size_t)NP + 1)), *tc = (int *)malloc(sizeof(int) * ((size_t)NT + 1));
    double *pyh = (double *)malloc(sizeof(double) * ((size_t)NP + 1)), *pyl = (double *)malloc(sizeof(double) * ((size_t)NP + 1));
    double *plh = (double *)malloc(sizeof(double) * ((size_t)NP + 1)), *pll = (double *)malloc(sizeof(double) * ((size_t)NP + 1));
    int **sr = (int **)calloc((size_t)T + 1, sizeof(int *)); double **sh = (double **)calloc((size_t)T + 1, sizeof(double *)), **sl = (double **)calloc((size_t)T + 1, sizeof(double *)); int *sn = (int *)calloc((size_t)T + 1, sizeof(int));
    int ok = pc && tc && pyh && pyl && plh && pll && sr && sh && sl && sn;
    if (ok) {
    /* stage 1: each top row's solve against the parts' columns (in the sequential code's order of those columns) */
    #pragma omp parallel num_threads(nt)
    {
        const int tid = omp_get_thread_num();
        int *yI = tid ? F->wsi[tid - 1] : F->iw, *el = tid ? F->wsi[tid - 1] + n : F->iw + n; unsigned char *mk = tid ? F->wsb[tid - 1] : F->bw;
        double *y = tid ? F->wsf[tid - 1] : F->fw, *yl = tid ? F->wsfl[tid - 1] : F->fwl;
        #pragma omp for schedule(dynamic, 4)
        for (int a = 0; a < T; a++) {
            const int k = F->ptop[a]; int nny; TOP_REACH(k, yI, el, mk, nny);
            for (int p = F->Ap[k]; p < F->Ap[k + 1]; p++) { const int i = F->Ai[p]; if (i != k && F->ptpos[i] < 0) { y[i] = F->Ax[p]; yl[i] = 0.0; } }
            sdd dk = { 0.0, 0.0 }; long q = pcnt[a], qt = tcnt[a];
            for (int t = nny - 1; t >= 0; t--) {
                const int c = yI[t]; mk[c] = 0;
                if (F->ptpos[c] >= 0) { tc[qt++] = c; continue; }
                const sdd yc = { y[c], yl[c] };
                const int pe = F->Lp[c] + npc[c];
                for (int p = F->Lp[c]; p < pe; p++) {
                    const int r = F->Li[p];
                    const sdd lv = { F->Lx[p], F->Lxl[p] }, yr = { y[r], yl[r] };
                    const sdd v = dd_sub(yr, dd_mul(lv, yc)); y[r] = v.h; yl[r] = v.l;
                }
                const sdd di = { F->Dinv[c], F->Dinvl[c] };
                const sdd l = dd_mul(yc, di);
                dk = dd_sub(dk, dd_mul(yc, l));
                pc[q] = c; pyh[q] = yc.h; pyl[q] = yc.l; plh[q] = l.h; pll[q] = l.l; q++;
                y[c] = 0.0; yl[c] = 0.0;
            }
            dkh[a] = dk.h; dkl[a] = dk.l;
        }
    }
    if (dbg_on()) tt1 = sc_time();
    /* the top rows' entries in the parts' columns, in increasing row order (each column's next positions) */
    for (int a = 0; a < T; a++) { const int k = F->ptop[a];
        for (long q = pcnt[a]; q < pcnt[a + 1]; q++) { const int c = pc[q]; const int pe = next[c]++; F->Li[pe] = k; F->Lx[pe] = plh[q]; F->Lxl[pe] = pll[q]; } }
    /* stage 2: the sums over the parts' columns for each top row */
    #pragma omp parallel num_threads(nt)
    {
        const int tid = omp_get_thread_num();
        int *yI = tid ? F->wsi[tid - 1] : F->iw; unsigned char *mk = tid ? F->wsb[tid - 1] : F->bw;
        double *y = tid ? F->wsf[tid - 1] : F->fw, *yl = tid ? F->wsfl[tid - 1] : F->fwl;
        #pragma omp for schedule(dynamic, 4)
        for (int a = 0; a < T; a++) {
            const int k = F->ptop[a]; int nr = 0;
            for (long q = pcnt[a]; q < pcnt[a + 1]; q++) {
                const int c = pc[q]; const sdd yc = { pyh[q], pyl[q] };
                for (int p = F->Lp[c] + npc[c]; p < next[c]; p++) {
                    const int r = F->Li[p]; if (r >= k) break;
                    if (!mk[r]) { mk[r] = 1; yI[nr++] = r; }
                    const sdd lv = { F->Lx[p], F->Lxl[p] }, yr = { y[r], yl[r] };
                    const sdd v = dd_sub(yr, dd_mul(lv, yc)); y[r] = v.h; yl[r] = v.l;
                }
            }
            sr[a] = (int *)malloc(sizeof(int) * ((size_t)nr + 1)); sh[a] = (double *)malloc(sizeof(double) * ((size_t)nr + 1)); sl[a] = (double *)malloc(sizeof(double) * ((size_t)nr + 1)); sn[a] = nr;
            for (int t = 0; t < nr; t++) { const int r = yI[t]; if (sr[a] && sh[a] && sl[a]) { sr[a][t] = r; sh[a][t] = y[r]; sl[a][t] = yl[r]; } y[r] = 0.0; yl[r] = 0.0; mk[r] = 0; }
        }
    }
    for (int a = 0; a < T; a++) if (!sr[a] || !sh[a] || !sl[a]) ok = 0;
    if (!ok) for (int j = 0; j < n; j++) next[j] = F->Lp[j] + npc[j];     /* (out of memory: the top's entries undone, the sequential rows instead) */
    }
    int ndyn = 0;
    if (dbg_on()) tt2 = sc_time();
    if (ok) {
    /* stage 3: each top row against the top's columns, in order */
    double *y = F->fw, *yl = F->fwl;
    for (int a = 0; a < T; a++) {
        const int k = F->ptop[a];
        sdd dk = { 0.0, 0.0 };
        for (int p = F->Ap[k]; p < F->Ap[k + 1]; p++) { const int i = F->Ai[p]; if (i == k) dk.h = F->Ax[p]; else if (F->ptpos[i] >= 0) { y[i] = F->Ax[p]; yl[i] = 0.0; } }
        { const sdd dp = { dkh[a], dkl[a] }; const sdd mdp = { -dp.h, -dp.l }; dk = dd_sub(dk, mdp); }
        for (int t = 0; t < sn[a]; t++) { const int r = sr[a][t]; const sdd yr = { y[r], yl[r] }, ms = { -sh[a][t], -sl[a][t] }; const sdd v = dd_sub(yr, ms); y[r] = v.h; yl[r] = v.l; }
        for (long q = tcnt[a]; q < tcnt[a + 1]; q++) {
            const int c = tc[q];
            const sdd yc = { y[c], yl[c] };
            const int pe = next[c];
            for (int p = F->Lp[c]; p < pe; p++) {
                const int r = F->Li[p];
                const sdd lv = { F->Lx[p], F->Lxl[p] }, yr = { y[r], yl[r] };
                const sdd v = dd_sub(yr, dd_mul(lv, yc)); y[r] = v.h; yl[r] = v.l;
            }
            const sdd di = { F->Dinv[c], F->Dinvl[c] };
            const sdd l = dd_mul(yc, di);
            dk = dd_sub(dk, dd_mul(yc, l));
            F->Li[pe] = k; F->Lx[pe] = l.h; F->Lxl[pe] = l.l; next[c]++;
            y[c] = 0.0; yl[c] = 0.0;
        }
        if (!(F->sign[k] * dk.h > LDL_PIVFLOOR(F, k))) { dk.h = F->sign[k] * F->dyn_delta; dk.l = 0.0; ndyn++; }
        const sdd di = dd_inv(dk);
        F->D[k] = dk.h; F->Dl[k] = dk.l; F->Dinv[k] = di.h; F->Dinvl[k] = di.l;
    }
    }
    #undef TOP_REACH
    if (dbg_on()) socp_printf("      [the top's stages: the solves against the parts %.3fs, the sums %.3fs, the top's own rows %.3fs; entries %ld from the parts' columns, %ld top columns visited]\n", tt1 - tt0, tt2 - tt1, sc_time() - tt2, NP, NT);
    for (int a = 0; a < T; a++) { if (sr) free(sr[a]); if (sh) free(sh[a]); if (sl) free(sl[a]); }
    free(sr); free(sh); free(sl); free(sn); free(pc); free(tc); free(pyh); free(pyl); free(plh); free(pll);
    free(pcnt); free(tcnt); free(dkh); free(dkl); free(npc);
    return ok ? ndyn : -1;
}
/* (12 November 2026, the SOCP developer, a test: BRISK_CONETOPPAR=1) the top rows of the double factorization in the three stages of
 * ldl_top_dd, the first two on the threads: the solves of the top rows against the parts' columns, then their sums over those
 * columns, then each top row against the top's own columns in order. db-joint-soerensen (SOCP), its full system: 65.7 % of the
 * factorization's work in the top rows, about half of it from the parts' columns, and the top on one thread (1.27 s a
 * factorization on one thread, 1.15 s on four). The order of the top rows' sums is not the sequential code's, so it is used at
 * every number of threads, one included (the same bits whatever the threads). Returns the replaced pivots, -1 when out of memory
 * (next[] untouched then). */
static int ldl_top_d(Ldl *F, int *next, int nt) {
    const int n = F->n, T = F->ntop;
    const double tt0 = dbg_on() ? sc_time() : 0.0; double tt1 = 0, tt2 = 0;
    int *pcnt = (int *)calloc((size_t)T + 1, sizeof(int)), *tcnt = (int *)calloc((size_t)T + 1, sizeof(int));
    double *dkp = (double *)calloc((size_t)T + 1, sizeof(double));
    int *npc = (int *)malloc(sizeof(int) * (size_t)n);
    if (!pcnt || !tcnt || !dkp || !npc) { free(pcnt); free(tcnt); free(dkp); free(npc); return -1; }
    for (int j = 0; j < n; j++) npc[j] = next[j] - F->Lp[j];
    #define TOP_REACH_D(k_, yI_, el_, mk_, nny_) do { nny_ = 0; \
        for (int p_ = F->Ap[k_]; p_ < F->Ap[k_ + 1]; p_++) { const int i_ = F->Ai[p_]; if (i_ == k_ || mk_[i_]) continue; \
            int ne_ = 0, j_ = i_; mk_[j_] = 1; el_[ne_++] = j_; j_ = F->etree[j_]; \
            while (j_ != -1 && j_ < k_ && !mk_[j_]) { mk_[j_] = 1; el_[ne_++] = j_; j_ = F->etree[j_]; } \
            while (ne_) yI_[nny_++] = el_[--ne_]; } } while (0)
    #pragma omp parallel num_threads(nt) if (nt > 1)
    {
        const int tid = omp_get_thread_num();
        int *yI = tid ? F->wsi[tid - 1] : F->iw, *el = tid ? F->wsi[tid - 1] + n : F->iw + n; unsigned char *mk = tid ? F->wsb[tid - 1] : F->bw;
        #pragma omp for schedule(dynamic, 16)
        for (int a = 0; a < T; a++) {
            const int k = F->ptop[a]; int nny; TOP_REACH_D(k, yI, el, mk, nny);
            int cp = 0, ct = 0; for (int t = 0; t < nny; t++) { const int c = yI[t]; mk[c] = 0; if (F->ptpos[c] < 0) cp++; else ct++; }
            pcnt[a + 1] = cp; tcnt[a + 1] = ct;
        }
    }
    for (int a = 0; a < T; a++) { pcnt[a + 1] += pcnt[a]; tcnt[a + 1] += tcnt[a]; }
    const long NP = pcnt[T], NT = tcnt[T];
    int *pc = (int *)malloc(sizeof(int) * ((size_t)NP + 1)), *tc = (int *)malloc(sizeof(int) * ((size_t)NT + 1));
    double *py = (double *)malloc(sizeof(double) * ((size_t)NP + 1)), *pl = (double *)malloc(sizeof(double) * ((size_t)NP + 1));
    int **sr = (int **)calloc((size_t)T + 1, sizeof(int *)); double **sh = (double **)calloc((size_t)T + 1, sizeof(double *)); int *sn = (int *)calloc((size_t)T + 1, sizeof(int));
    int ok = pc && tc && py && pl && sr && sh && sn;
    if (ok) {
    /* stage 1: each top row's solve against the parts' columns */
    #pragma omp parallel num_threads(nt) if (nt > 1)
    {
        const int tid = omp_get_thread_num();
        int *yI = tid ? F->wsi[tid - 1] : F->iw, *el = tid ? F->wsi[tid - 1] + n : F->iw + n; unsigned char *mk = tid ? F->wsb[tid - 1] : F->bw;
        double *y = tid ? F->wsf[tid - 1] : F->fw;
        #pragma omp for schedule(dynamic, 4)
        for (int a = 0; a < T; a++) {
            const int k = F->ptop[a]; int nny; TOP_REACH_D(k, yI, el, mk, nny);
            for (int p = F->Ap[k]; p < F->Ap[k + 1]; p++) { const int i = F->Ai[p]; if (i != k && F->ptpos[i] < 0) y[i] = F->Ax[p]; }
            double dk = 0.0; long q = pcnt[a], qt = tcnt[a];
            for (int t = nny - 1; t >= 0; t--) {
                const int c = yI[t]; mk[c] = 0;
                if (F->ptpos[c] >= 0) { tc[qt++] = c; continue; }
                const double yc = y[c];
                const int pe = F->Lp[c] + npc[c];
                for (int p = F->Lp[c]; p < pe; p++) y[F->Li[p]] -= F->Lx[p] * yc;
                const double l = yc * F->Dinv[c];
                dk -= yc * l;
                pc[q] = c; py[q] = yc; pl[q] = l; q++;
                y[c] = 0.0;
            }
            dkp[a] = dk;
        }
    }
    if (dbg_on()) tt1 = sc_time();
    for (int a = 0; a < T; a++) { const int k = F->ptop[a];
        for (long q = pcnt[a]; q < pcnt[a + 1]; q++) { const int c = pc[q]; const int pe = next[c]++; F->Li[pe] = k; F->Lx[pe] = pl[q]; } }
    /* stage 2: the sums over the parts' columns for each top row */
    #pragma omp parallel num_threads(nt) if (nt > 1)
    {
        const int tid = omp_get_thread_num();
        int *yI = tid ? F->wsi[tid - 1] : F->iw; unsigned char *mk = tid ? F->wsb[tid - 1] : F->bw;
        double *y = tid ? F->wsf[tid - 1] : F->fw;
        #pragma omp for schedule(dynamic, 4)
        for (int a = 0; a < T; a++) {
            const int k = F->ptop[a]; int nr = 0;
            for (long q = pcnt[a]; q < pcnt[a + 1]; q++) {
                const int c = pc[q]; const double yc = py[q];
                for (int p = F->Lp[c] + npc[c]; p < next[c]; p++) {
                    const int r = F->Li[p]; if (r >= k) break;
                    if (!mk[r]) { mk[r] = 1; yI[nr++] = r; }
                    y[r] -= F->Lx[p] * yc;
                }
            }
            sr[a] = (int *)malloc(sizeof(int) * ((size_t)nr + 1)); sh[a] = (double *)malloc(sizeof(double) * ((size_t)nr + 1)); sn[a] = nr;
            for (int t = 0; t < nr; t++) { const int r = yI[t]; if (sr[a] && sh[a]) { sr[a][t] = r; sh[a][t] = y[r]; } y[r] = 0.0; mk[r] = 0; }
        }
    }
    for (int a = 0; a < T; a++) if (!sr[a] || !sh[a]) ok = 0;
    if (!ok) for (int j = 0; j < n; j++) next[j] = F->Lp[j] + npc[j];
    }
    int ndyn = 0;
    if (dbg_on()) tt2 = sc_time();
    if (ok) {
    /* stage 3: each top row against the top's columns, in order */
    double *y = F->fw;
    for (int a = 0; a < T; a++) {
        const int k = F->ptop[a];
        double dk = 0.0;
        for (int p = F->Ap[k]; p < F->Ap[k + 1]; p++) { const int i = F->Ai[p]; if (i == k) dk = F->Ax[p]; else if (F->ptpos[i] >= 0) y[i] = F->Ax[p]; }
        dk += dkp[a];
        for (int t = 0; t < sn[a]; t++) y[sr[a][t]] += sh[a][t];
        for (long q = tcnt[a]; q < tcnt[a + 1]; q++) {
            const int c = tc[q];
            const double yc = y[c];
            const int pe = next[c];
            for (int p = F->Lp[c]; p < pe; p++) y[F->Li[p]] -= F->Lx[p] * yc;
            const double l = yc * F->Dinv[c];
            dk -= yc * l;
            F->Li[pe] = k; F->Lx[pe] = l; next[c]++;
            y[c] = 0.0;
        }
        if (!(F->sign[k] * dk > LDL_PIVFLOOR(F, k))) {
            static double bigp = -1; if (bigp < 0) { const char *e_ = getenv("BRISK_CONEBIGPIV"); bigp = e_ ? atof(e_) : 0.0; }
            dk = F->sign[k] * (bigp > 0 ? bigp : F->dyn_delta); ndyn++; }
        F->D[k] = dk; F->Dinv[k] = 1.0 / dk;
        if (F->pivdbg) { const int o = F->perm[k]; const int cl = o < F->pivnf ? 0 : o < F->pivn ? 1 : o < F->pivn + F->pivnaux ? 2 : 3; if (fabs(dk) < F->pivmin[cl]) F->pivmin[cl] = fabs(dk); if (fabs(dk) < 1e-6) F->pivcnt[cl]++; }
    }
    }
    #undef TOP_REACH_D
    if (dbg_on()) socp_printf("      [the double factorization's top in three stages: the solves against the parts %.3fs, the sums %.3fs, the top's own rows %.3fs]\n", tt1 - tt0, tt2 - tt1, sc_time() - tt2);
    for (int a = 0; a < T; a++) { if (sr) free(sr[a]); if (sh) free(sh[a]); }
    free(sr); free(sh); free(sn); free(pc); free(tc); free(py); free(pl);
    free(pcnt); free(tcnt); free(dkp); free(npc);
    return ok ? ndyn : -1;
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
    static int tpar = -1; if (tpar < 0) { const char *e_ = getenv("BRISK_CONEDDTOPPAR"); tpar = e_ ? atoi(e_) : 1; }   /* (14 November 2026, the SOCP developer: 1, the default; the 40 SOCPs whose solves end in double-double 0.945, the same iterations on all, db-joint-soerensen 119.9 -> 112.1 s; left off on 4 November for the pivots double-double then replaced in joint_HO_02 and 24, which the small cones' eigenbasis now removes. 0: the top on one thread) */
    /* (the three-stage top changes the order of the top rows' sums: it is used at every number of threads, one included,
     * so that the factor has the same bits whatever the threads) */
    if (nt > 1 || (tpar && F->npart > 0 && F->ptsplit && F->pt_fac)) {
        int nd[LDL_NPART] = { 0 };
        #pragma omp parallel for schedule(dynamic, 1) num_threads(nt) if (nt > 1)
        for (int q = 0; q < F->npart; q++) {
            const int tid = omp_get_thread_num();
            int *yI = tid ? F->wsi[tid - 1] : yIdx, *el = tid ? F->wsi[tid - 1] + n : elim; unsigned char *mk = tid ? F->wsb[tid - 1] : mark;
            double *yy = tid ? F->wsf[tid - 1] : y, *yyl = tid ? F->wsfl[tid - 1] : yl;
            for (int a = F->pnp[q]; a < F->pnp[q + 1]; a++) nd[q] += ldl_row_dd(F, F->pnl[a], yI, el, next, mk, yy, yyl);
        }
        for (int q = 0; q < F->npart; q++) F->ndyn += nd[q];
        const double t0_ = dbg_on() ? sc_time() : 0.0;
        int rt = -1; if (tpar) rt = ldl_top_dd(F, next, nt);
        if (rt >= 0) F->ndyn += rt;
        else { if (tpar) { /* (memory: the sequential rows; next[] is untouched until stage 1 succeeds) */ }
            for (int a = 0; a < F->ntop; a++) F->ndyn += ldl_row_dd(F, F->ptop[a], yIdx, elim, next, mark, y, yl); }
        if (dbg_on()) socp_printf("      [double-double factorization: %d parts on %d threads, then %d of %d rows %s, %.3fs]\n", F->npart, nt, F->ntop, n, rt >= 0 ? "in three stages, two on the threads" : "on one thread", sc_time() - t0_);
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
    double *axc_part;                    /* (28 October 2026) A x by fixed chunks of columns: AXC_CH partial sums of m rows (NULL: off) */
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
    double sreg, sregx, sregf, sreg0, bump; int ref_pred, bump_on, nbump, bump_hold, bump_try_it, cur_it;
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
    int ne_onec, ne_onec_ok; double ne_onec_eta2, ne_scale;   /* (2 November 2026: one dense cone's constant part factored once, ne_factor_onec) */
    SChol *ne; size_t *ne_off, *ne_doff; double *ne_G; int *ne_r, **ne_rows; int ne_ok, ne_fallback, ne_tiny, ne_left, ne_tinykept, free_last, dz_refine, dd_stall, dd_refused; double t_begin; double *wdz, *wdx, *wdy, *wdy2; long dz_count, zproj_used; int dd_force; double *ne_W, *ne_T, *ne_xf, *ne_Af; long ndyn_total; long ne_nfact, ne_lnz; double *ne_C; int nr_rot; int **bpos; double *bth; int *Rq; long nboost; int anyrot; double ne_dl; int dl_exact;
    int ne_nd, ne_ndc, ne_md, *ne_dk, *ne_drow, *ne_dpos; size_t *ne_doffd; char ne_dense_msg[96]; int ne_nwl, *ne_wl, ne_wrank; int kkt_defer, kkt_defer_lb, kkt_defer_md; long nrot_tr; int rot_def, rot_mode, nsmallc, rot_switch, rot_back, rot_want, qd_raised, meas_sup, qd_ever, rsp, ne_iron, ne_back; SChol *ne_park; double it_rs2, it_rt2, *rspb; int rspL; double *lrb, *lrnb; unsigned char *lrsel; double *scu_w, *scu_eta, *scu_t, *scu_l; unsigned char *scu_redo; int scu_ok; long scu_nredo;   /* (the eigenbasis of the small cones: 0 never, 1 from the start, 2 when the solves need it; the residual's share in the small cones this iteration) */
    /* (23 October 2026) rows that fix one entry of a small cone (a row with a single entry, in that cone's column), eliminated
     * from the normal equations: ne_fixrow[kc] the row (-1: none), ne_fixpos[kc] the entry's place in the cone, ne_fixa[kc]
     * its coefficient, ne_fixt[kc] the row's right-hand side during a solve; ne_elim[i] 1 for such a row */
    int ne_nfix, *ne_fixrow, *ne_fixpos; double *ne_fixa, *ne_fixt; char *ne_elim;
    /* (23 October 2026) dense linear columns kept as unknowns of the normal equations: [S A_d; A_d' -H_d] (dy; dx_d), the
     * unknowns m .. m + ne_nkd - 1; ne_kdcol the columns, ne_kdoff their diagonals' places, ne_kaoff the places of A_d's
     * entries (by A's index) */
    int ne_nkd, *ne_kdcol; size_t *ne_kdoff, *ne_kaoff; int ne_mm;   /* (ne_mm: the system's unknowns, m + ne_nkd) */
    /* (13 November 2026, BRISK_CONEPARTNE) the partial normal equations: cones kept as unknowns; ne_kcsel the selection (per cone, kept
     * across a rebuild), ne_kdcone / ne_kdpos the cone and entry of each kept unknown (-1: a linear column), ne_kcoff the places of
     * each kept cone's off-diagonal entries (its lower triangle row by row), ne_nkc the cones kept, ne_part the rebuild done */
    unsigned char *ne_kcsel; int *ne_kdcone, *ne_kdpos; size_t *ne_kcoff; int ne_nkc, ne_part; double ne_part_mu;   /* (ne_part_mu: mu at the last selection) */
    /* (BRISK_CONEPARTNEEIG, the default with the partial normal equations) a kept cone's unknowns in the eigenbasis of its scaling: ne_kus /
     * ne_kun the start and count of its rows (the union of its columns' rows) in ne_kurow per kept unknown (-1: a linear column), ne_kuo the
     * start of each kept unknown's places in ne_kuoff, ne_kq the basis (NE_MAXCONE entries per kept unknown: its eigenvector), ne_keig on */
    int *ne_kus, *ne_kun, *ne_kurow; long *ne_kuo; size_t *ne_kuoff; double *ne_kq; int ne_keig;
    int ne_ndt, *ne_dtcol, *ne_dtk; char *ne_xden;   /* (25 October 2026) the dense tail columns of the sparse-form cones, in the Woodbury update */
    int ne_full, ne_dfull; size_t ne_dbase;   /* (1 November 2026) a block over all m rows: the pattern complete; the factor one dense supernode (positions base + j m + i) */
    int ne_nsp, *ne_spk, ne_nex, *ne_excol, *ne_exk, **ne_exrows, *ne_exr, ne_nu, ne_nu0, ne_wcap, ne_ntf, ne_depmode, ne_tfrow[NE_TINYMAX], ne_ndr, ne_dprow[NE_TINYMAX], ne_tfdep[NE_TINYMAX], ne_depused; double ne_tfd[NE_TINYMAX], ne_dpw[NE_TINYMAX], *ne_dg; double *ne_wU, *ne_wW, *ne_wC; int *ne_wpiv; unsigned char *fcone;   /* (large cones by their sparse low-rank form: the tail columns as pairs, [v a_0] by a Woodbury update) */     /* (the dense linear columns of the normal equations: their blocks, rows, offsets) */
    int ne_nspl, *ne_splk; double *ne_splg, *ne_spld;   /* (BRISK_CONENESPLIT: the dense-route large cones whose largest eigenvalue's term goes to the Woodbury update; its column g and its coefficient) */
    SChol *sc; size_t *soff; long knz; double *swork;   /* the supernodal factorization, when it is used */
    int sc_lazy, *lz_ti, *lz_tj, *lz_pos, *lz_perm; long lz_nz; double lz_amalg;   /* (2 November 2026) its analysis put off (kkt_sc_build) */
    int refine_max, verbose;
    long refine_total, solves;
} Socp;
static double start_mu_rest(const Socp *S, const unsigned char *skip, double dflt);
static int start_fixmu(void);

/* y += alpha A x ; y += alpha A' x.
 * 18 October 2026 (LP/SOCP): on the threads the SOCP solver keeps for its large blocks (g_cone_nt), when A has 1e5
 * entries or more: A'x by columns (each entry its own sum), A x as a gather by rows from a row copy whose entries
 * are in ascending column order, each y_i starting from its value: the additions into y_i in the order of the
 * scatter by columns, so the same bits at any thread count (BRISK_CONEAPAR=0: the sequential products). */
static _Thread_local int g_cone_nt = 1;     /* (per thread: the calling thread's solve; ISSUES 291) */
static _Thread_local int g_team_nt0 = 0;    /* (the count of the start, for the busy-machine rule during the solve; 0: none) */
/* (28 October 2026) the busy-machine rule during the solve: called at each iteration, before each factorization and each
 * solve (a few times an iteration: an SOCP may take only a handful of iterations, and the rule needs two windows of at
 * least 50 ms over its threshold before it lowers the count; a call closer than 50 ms to the last one decides nothing) */
static void cone_threads_adapt(int verbose) {
    if (g_team_nt0 <= 1 || omp_in_parallel()) return;
    const int w_ = brisk_team_adapt(g_cone_nt);
    if (w_ == g_cone_nt) return;
    g_cone_nt = g_cone_nt_ldl = w_; schol_set_panel_threads(w_);
    if (verbose > 1) socp_printf("      [threads: %d of %d]\n", w_, g_team_nt0);
}
static int cone_part_nt(void) { return omp_in_parallel() ? 1 : g_cone_nt; }
/* (3 November 2026, the SOCP developer, a test: BRISK_CONENEPART=1) the normal equations' solves by independent subtrees on the
 * threads (schol_solve_part, the same bits at any thread count), as the full system's; db-joint-soerensen (SOCP): 372
 * applications of the normal equations' factor took 25 of the first 58 s, on one thread. Only for a factor of at least 4
 * million entries (32 MB, beyond the cache): a solve is bound by memory bandwidth, and a factor that fits in the cache gains
 * nothing from the threads and pays the partitioning (socp_manysmall, SOCP: 3.6 -> 9.9 s with it) */
/* (BRISK_CONENERATE) the column code's slower rate assumed only where it is: a full system of at least 60 flops per entry of
 * its factor (the backend rule's own lower threshold, kkt_analyse: the supernodal code from 60 to 110 flops per entry); at
 * lower density both codes are bound by memory traffic (socp_gctrl_20000x12x6, SOCP: 14.5 flops per entry, the normal
 * equations taken for it, 7.4 -> 12 s; db-joint-soerensen, SOCP: 99) */
static _Thread_local double g_meas_lnz_kept;     /* (the presolve's count of the full system's entries of L, when the analysis is put off on it) */
static int nerate_dense(const Socp *S) {
    const double lz = S->meas_sup >= 0 ? g_meas_lnz_kept : (double)S->F.lnz;
    return lz > 0 && S->cflops >= 60.0 * lz;
}
/* (4 November 2026, the SOCP developer, at the owner's decision: "Make it on with the low tolerance") the package that stops at
 * the limit of double precision (the substitution chosen by cost, the routes compared by predicted time, the normal equations'
 * solves by subtrees, the full system's analysis put off, the full system's double factor checked against the quasi-definite
 * bound, and the precision-limit stop) is the default at a tolerance of 1e-6 or looser (-acc low), off at the default and
 * high accuracy: there it makes the joint_HO structural designs (SOCP) end at reduced accuracy where the default reaches
 * OPTIMAL in double-double (ISSUES 820). Each switch's environment variable, when set, still decides. g_cone_lowtol is set by
 * socp_solve for the solve (one solve at a time in a process, as the other solver-wide settings) */
static int g_cone_lowtol = 0;
static int cone_pkg(const char *name, int *cache) {
    if (*cache == -2) { const char *e = getenv(name); *cache = e ? atoi(e) : -1; }
    return *cache >= 0 ? *cache : g_cone_lowtol;
}
/* (5 November 2026, the owner's decision for release 1.6.5: "on for the release") four of the package's settings are the default
 * at every tolerance: the substitution chosen by cost (BRISK_CONEELIMCOST), the routes compared by predicted time
 * (BRISK_CONENERATE), the normal equations' solves split across the threads (BRISK_CONENEPART) and the full system's analysis
 * put off (BRISK_CONEMEASDEFER); the quasi-definite check and the precision-limit stop stay those of -acc low (cone_pkg). Each
 * variable, when set, still decides (0 turns the setting off). Measured with b9d2c52f on the 214 SOCPs of the development set
 * (with the return to the normal equations below): 6537 s against the earlier defaults' 6753 s; db-joint-soerensen (SOCP)
 * 113.6 s instead of the time limit */
static int cone_on(const char *name, int *cache) {
    if (*cache == -2) { const char *e = getenv(name); *cache = e ? atoi(e) : -1; }
    return *cache >= 0 ? *cache : 1;
}
/* (5 November 2026, the same decision) the return to the normal equations (BRISK_CONENEBACK) and GMRES's stagnation stop after it
 * (BRISK_CONEGMBACK) are the default at the default and high accuracy; at -acc low they stay off: there the precision-limit stop
 * answers the same broken factor, and -acc low keeps the results it was validated with */
static int cone_back(const char *name, int *cache) {
    if (*cache == -2) { const char *e = getenv(name); *cache = e ? atoi(e) : -1; }
    return *cache >= 0 ? *cache : !g_cone_lowtol;
}
static int ne_part(void) { static int c_ = -2; return cone_on("BRISK_CONENEPART", &c_); }   /* (the partitioned solve's threads) */   /* (the partitioned solve's threads) */
static int cone_apar(const Socp *S) {
    static int on = -1; if (on < 0) { const char *e = getenv("BRISK_CONEAPAR"); on = !(e && atoi(e) == 0); }
    return on && g_cone_nt > 1 && S->Rp && !omp_in_parallel();
}
/* (28 October 2026, the SOCP developer; ISSUES 805) y += al A x by columns in AXC_CH chunks of about equal entries, each
 * chunk's sum into its own vector of m rows, the chunks then added row by row in their order. By rows (the row copy), each
 * row gathers its entries of x from all over x: with many more columns than rows, a cache miss an entry (600,000 cones of
 * dimension 3 over 300 rows, 7.2 million entries: 25 ms a product against 5 ms for A'y by columns). By columns x is read in
 * order and the m partial sums stay in the cache. The chunks are fixed by A alone, so the bits are the same at any thread
 * count; the rule that takes this path (axc_setup) looks at A's shape only. */
#define AXC_CH 64
static void axc_mul(const Socp *S, const int *Ap, const int *Ai, const double *Ax, double al, const double *x, double *y) {
    const int n = S->n, m = S->m; double *part = S->axc_part; const long nz = Ap[n];
    const int nt = g_cone_nt > 1 && !omp_in_parallel() ? g_cone_nt : 1;
    #pragma omp parallel for schedule(static) num_threads(nt) if (nt > 1)
    for (int c = 0; c < AXC_CH; c++) {
        int lo = 0, hi = n;                    /* the chunk's columns: from the first column with Ap >= c nz / AXC_CH */
        { const long t = nz * c / AXC_CH; int a = 0, b = n; while (a < b) { const int h = (a + b) / 2; if (Ap[h] < t) a = h + 1; else b = h; } lo = a; }
        { const long t = nz * (c + 1) / AXC_CH; int a = 0, b = n; while (a < b) { const int h = (a + b) / 2; if (Ap[h] < t) a = h + 1; else b = h; } hi = c + 1 == AXC_CH ? n : a; }
        double *pc = part + (size_t)c * m;
        for (int i = 0; i < m; i++) pc[i] = 0.0;
        for (int j = lo; j < hi; j++) { const double xj = al * x[j]; if (xj != 0.0) for (int p = Ap[j]; p < Ap[j + 1]; p++) pc[Ai[p]] += Ax[p] * xj; }
    }
    for (int i = 0; i < m; i++) { double v = 0.0; for (int c = 0; c < AXC_CH; c++) v += part[(size_t)c * m + i]; y[i] += v; }
}
/* the rule, by A's shape alone: at least 1e5 entries, m partial sums small enough for a core's cache (m <= 32,768), and the
 * chunks' own work (each product zeroes and adds AXC_CH m partial sums) at most an eighth of the product's entries. With a
 * quarter, socp_beamform_200x3000 (SOCP, 8,143 rows, 2.2 million entries: the partial sums half the product's work) lost 9 %
 * (median of three, 3.93 -> 4.28 s at four threads); BRISK_CONEAXC=0 off */
static void axc_setup(Socp *S) {
    static int on = -1; if (on < 0) { const char *e = getenv("BRISK_CONEAXC"); on = !(e && atoi(e) == 0); }
    const long nz = S->Ap[S->n];
    if (on && nz >= 100000 && S->m > 0 && S->m <= 32768 && 2.0 * AXC_CH * (double)S->m <= 0.125 * (double)nz)
        S->axc_part = (double *)malloc(sizeof(double) * (size_t)AXC_CH * (size_t)S->m);
    if (dbg_on() && S->axc_part) socp_printf("  [A x by %d chunks of columns: %d rows, %ld entries]\n", AXC_CH, S->m, nz);
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
/* (2 November 2026, the SOCP developer) the rows or columns of A on the threads by ENT_CH chunks of about equal entries, assigned
 * statically (contiguous chunks a thread: a thread meets the same rows in every product and finds them in its cache; handed
 * out in turn, small problems of the held-out set ran 12-20 % slower with the same bits): split by count, a matrix whose long rows sit together gave one thread nearly all the work (CBLIB's beam7, SOCP:
 * 107,000 rows of one entry next to 106,000 of about 140, and 467 columns of about 31,000 entries; its products with A as slow at
 * four threads as at one). Each row's or column's sum is formed by one thread in the order of its entries: the same bits at any
 * thread count. ent_lo(p, n, c): the first index of chunk c (p the pointers, n the count) */
#define ENT_CH 64
static inline int ent_lo(const int *p, int n, int c) {
    if (c <= 0) return 0; if (c >= ENT_CH) return n;
    const long t = (long)p[n] * c / ENT_CH; int a = 0, b = n;
    while (a < b) { const int h = (a + b) / 2; if (p[h] < t) a = h + 1; else b = h; }
    return a;
}
/* ent_part(p, n, t, T): the first index of part t of T parts of about equal entries (a thread's range: the setup's passes) */
static inline int ent_part(const int *p, int n, int t, int T) {
    if (t <= 0) return 0; if (t >= T) return n;
    const long tt = (long)p[n] * t / T; int a = 0, b = n;
    while (a < b) { const int h = (a + b) / 2; if (p[h] < tt) a = h + 1; else b = h; }
    return a;
}
/* A x and A'y of the rest of A, the dense block apart (S->nd > 0) */
static void B_mul(const Socp *S, double al, const double *x, double *y) {
    if (S->axc_part) { axc_mul(S, S->Bp, S->Bi, S->Bx, al, x, y); return; }
    if (cone_apar(S)) {
        #pragma omp parallel for schedule(static) num_threads(g_cone_nt)
        for (int ch = 0; ch < ENT_CH; ch++) { const int i1 = ent_lo(S->Rp, S->m, ch + 1);
        for (int i = ent_lo(S->Rp, S->m, ch); i < i1; i++) {
            if (S->dmap[i] >= 0) continue;
            double v = y[i];
            for (int q = S->Rp[i]; q < S->Rp[i + 1]; q++) { const double xj = al * x[S->Ri[q]]; if (xj != 0.0) v += S->Rx[q] * xj; }
            y[i] = v;
        } }
        return;
    }
    for (int j = 0; j < S->n; j++) { const double xj = al * x[j]; if (xj != 0.0) for (int p = S->Bp[j]; p < S->Bp[j + 1]; p++) y[S->Bi[p]] += S->Bx[p] * xj; }
}
static void Bt_mul(const Socp *S, double al, const double *x, double *y) {
    #pragma omp parallel for schedule(static) num_threads(g_cone_nt) if (cone_apar(S))
    for (int ch = 0; ch < ENT_CH; ch++) { const int j1 = ent_lo(S->Bp, S->n, ch + 1);
    for (int j = ent_lo(S->Bp, S->n, ch); j < j1; j++) { double s = 0.0; for (int p = S->Bp[j]; p < S->Bp[j + 1]; p++) s += S->Bx[p] * x[S->Bi[p]]; y[j] += al * s; } }
}
static void A_mul(const Socp *S, double al, const double *x, double *y) {
    if (S->nd > 0) { B_mul(S, al, x, y); dn_mul(S, al, x, y, NULL, NULL); return; }
    if (S->axc_part) { axc_mul(S, S->Ap, S->Ai, S->Ax, al, x, y); return; }
    if (cone_apar(S)) {
        #pragma omp parallel for schedule(static) num_threads(g_cone_nt)
        for (int ch = 0; ch < ENT_CH; ch++) { const int i1 = ent_lo(S->Rp, S->m, ch + 1);
        for (int i = ent_lo(S->Rp, S->m, ch); i < i1; i++) {
            double v = y[i];
            for (int q = S->Rp[i]; q < S->Rp[i + 1]; q++) { const double xj = al * x[S->Ri[q]]; if (xj != 0.0) v += S->Rx[q] * xj; }
            y[i] = v;
        } }
        return;
    }
    for (int j = 0; j < S->n; j++) { const double xj = al * x[j]; if (xj != 0.0) for (int p = S->Ap[j]; p < S->Ap[j + 1]; p++) y[S->Ai[p]] += S->Ax[p] * xj; }
}
static void At_mul(const Socp *S, double al, const double *x, double *y) {
    if (S->nd > 0) { Bt_mul(S, al, x, y); dn_mul(S, al, NULL, NULL, x, y); return; }
    #pragma omp parallel for schedule(static) num_threads(g_cone_nt) if (cone_apar(S))
    for (int ch = 0; ch < ENT_CH; ch++) { const int j1 = ent_lo(S->Ap, S->n, ch + 1);
    for (int j = ent_lo(S->Ap, S->n, ch); j < j1; j++) { double s = 0.0; for (int p = S->Ap[j]; p < S->Ap[j + 1]; p++) s += S->Ax[p] * x[S->Ai[p]]; y[j] += al * s; } }
}
/* (27 October 2026, the SOCP developer) the loops over the cones on the threads: each cone writes only its own entries, and a
 * minimum is the same in any order, so the results are the same bits at any thread count (the SDP developer and
 * integrator's finding on 400,000 small cones). Worth a parallel region (a few microseconds) from about 32,768 entries on. */
static int cone_vpar(const Socp *S) {
    static int on = -1; if (on < 0) { const char *e = getenv("BRISK_CONEVPAR"); on = !(e && atoi(e) == 0); }
    return on && g_cone_nt > 1 && S->n >= 32768 && !omp_in_parallel();
}
static double dotv(int n, const double *a, const double *b) { double s = 0; for (int i = 0; i < n; i++) s += a[i] * b[i]; return s; }
/* (29 October 2026, the SOCP developer) the refinement's long vectors on the threads: a dot product of 2^18 entries or more in
 * 64 fixed chunks summed in their order (the chunks do not depend on the team: the same bits at any thread count, also on
 * one thread), and y += a x entry by entry (the same bits as the loop). A structural-design SOCP's full system of 2.4 million
 * unknowns: the Gram-Schmidt passes of the refinement ran on one thread */
#define DOTC_MIN 262144
static double dotc(const Socp *S, int n, const double *a, const double *b) {
    if (n < DOTC_MIN) return dotv(n, a, b);
    double part[64];
    #pragma omp parallel for schedule(static) num_threads(g_cone_nt) if (cone_vpar(S))
    for (int c = 0; c < 64; c++) {
        const long lo = (long)n * c / 64, hi = (long)n * (c + 1) / 64; double t = 0;
        for (long i = lo; i < hi; i++) t += a[i] * b[i];
        part[c] = t;
    }
    double t = 0; for (int c = 0; c < 64; c++) t += part[c];
    return t;
}
static void axpyc(const Socp *S, int n, double a, const double *x, double *y) {
    #pragma omp parallel for schedule(static) num_threads(g_cone_nt) if (n >= DOTC_MIN && cone_vpar(S))
    for (int i = 0; i < n; i++) y[i] += a * x[i];
}
static double nrm2(int n, const double *a) { return sqrt(dotv(n, a, a)); }
static double sqrt_dotc_(const Socp *S, int n, const double *a) { return n < DOTC_MIN ? nrm2(n, a) : sqrt(dotc(S, n, a, a)); }
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
static inline int lrot_cand(const Socp *S, int k);
static void H_apply(const Socp *S, const double *in, double *out) {
    for (int i = 0; i < S->nf; i++) out[i] = 0.0;
    const int par = cone_vpar(S);
    #pragma omp parallel for schedule(static) num_threads(g_cone_nt) if (par)
    for (int i = 0; i < S->nl; i++) out[S->nf + i] = S->h[i] * in[S->nf + i];
    #pragma omp parallel for schedule(static) num_threads(g_cone_nt) if (par)
    for (int k = 0; k < S->nq; k++) {
        if (S->rspL && S->lrsel && S->lrsel[k] && lrot_cand(S, k)) continue;   /* (held in the eigenbasis: kkt_mul writes its block itself) */
        const int d = S->q[k], s = S->qs[k]; const double *w = S->wbar + s; const double e2 = S->eta[k] * S->eta[k];
        double wx = 0.0;
        for (int i = 0; i < d; i++) wx += w[i] * in[s + i];
        out[s] = e2 * (2.0 * wx * w[0] - in[s]);
        for (int i = 1; i < d; i++) out[s + i] = e2 * (2.0 * wx * w[i] + in[s + i]);
    }
}

/* the values of the factored matrix from the current scaling */
static int kkt_analyse(Socp *S);
static void kkt_sc_build(Socp *S);
static void kkt_fill(Socp *S) {
    Ldl *F = &S->F;
    if (S->ne && !S->ne_fallback) return;            /* (the normal equations are assembled by ne_factor) */
    if (S->kkt_defer) {                               /* (the full system needed for the first time: analysed now) */
        const double ts = sc_time(); const int rc = kkt_analyse(S); S->t_setup += sc_time() - ts;
        if (dbg_on()) socp_printf("  [the full system analysed when first needed: %.3fs%s]\n", sc_time() - ts, rc ? ", failed" : "");
    }
    if (S->sc_lazy) { const double ts = sc_time(); kkt_sc_build(S); S->t_setup += sc_time() - ts;
        if (dbg_on()) socp_printf("  [the full system's supernodal analysis made when first needed: %.3fs]\n", sc_time() - ts); }
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
    static double rgr = -1, rgf = 1e-14, rgc = 0; if (rgr < 0) { const char *e_ = getenv("BRISK_CONEREGREL"); rgr = 0.0; if (e_) sscanf(e_, "%lf,%lf,%lf", &rgr, &rgf, &rgc); }
    const int rgon = rgr > 0 && !S->use_dd && S->bump == 1.0;
    /* (31 October 2026, the SOCP developer; ISSUES 810: the third value 1 caps each entry's regularization at the fixed one, so
     * that it is only lowered, where the entry of H is small; 2 also measures a cone by the smallest eigenvalue of its block,
     * eta^2 / (w0 + |w1|)^2, instead of its diagonal entries, which can be orders larger: the regularization a small fraction
     * of the block's smallest eigenvalue, so the factor stays close to the matrix in every direction; 3: each cone by its
     * scale eta^2 alone (H = eta^2 times a matrix of determinant 1), the linear block at the fixed value) */
#define RGREL(h_) (rgc > 0 ? fmin(sx, fmax(rgr * (h_), rgf)) : fmax(rgr * (h_), rgf))
    for (int i = 0; i < S->nl; i++) F->Ax[S->kdiag[S->nf + i]] = -S->h[i] - (rgon && rgc < 3 ? RGREL(S->h[i]) : sx);
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
                    F->Ax[pos[t++]] = i == j ? -e2 * lam - (rgon ? RGREL(rgc >= 3 ? e2 : e2 * lam) : sxs) : 0.0;
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
            double lmin_ = 0.0;      /* (the smallest eigenvalue of H / eta^2: 1 / (w0 + |w1|)^2) */
            if (rgon && rgc >= 2) { double nq_ = 0.0; for (int i = 1; i < d; i++) nq_ += w[i] * w[i]; nq_ = w[0] + sqrt(nq_); lmin_ = 1.0 / (nq_ * nq_); }
            for (int j = 0; j < d; j++)
                for (int i = 0; i <= j; i++) {
                    double v = 2.0 * w[i] * w[j];
                    if (i == j) v += i == 0 ? -1.0 : 1.0;
                    F->Ax[pos[t++]] = -e2 * v - (i == j ? (rgon ? RGREL(rgc >= 3 ? e2 : rgc >= 2 ? e2 * lmin_ : e2 * fabs(v)) : sxs) : 0.0);
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
            for (int i = 0; i < d; i++) F->Ax[S->kdiag[s + i]] = -e2 - (rgon ? RGREL(rgc == 2 ? e2 * lm : e2) : sx);
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
    /* (3 November 2026, the SOCP developer, a test: BRISK_CONEFREEREL=kappa[,floor]) each free variable's regularization in
     * the factored matrix relative to its curvature through the rows, delta_j = min(delta_f, max(kappa s_j, floor)),
     * s_j = sum_i a_ij^2 / M_ii with M_ii the rows' diagonal of the normal equations without the free columns. The
     * refinement targets the matrix without it, and its preconditioned eigenvalue on that variable is about
     * s_j / (s_j + delta_j): with a fixed delta_f = 1e-6 it falls as mu does where the cones' z vanish (M_ii grows like
     * 1 / mu, s_j falls like mu), which is where db-joint-soerensen's free columns' dual residual stalls. Relative to s_j the
     * eigenvalue stays near 1 / (1 + kappa), and eliminating the variable adds a_ij^2 / delta_j = M_ii / kappa to its rows'
     * pivots, within 1 / kappa of their own size (quasi-definite growth about 1 / sqrt(kappa)). */
    { static double frk = -1, frf = 1e-12; if (frk < 0) { const char *e_ = getenv("BRISK_CONEFREEREL"); frk = 0.0; if (e_) sscanf(e_, "%lf,%lf", &frk, &frf); }
      if (frk > 0 && !S->use_dd && S->nf > 0) {
        const int m = S->m;
        if (!S->fpr_si) S->fpr_si = (double *)calloc((size_t)m + 1, sizeof(double));
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
        long nlow = 0; double dmin = S->sregf;
        for (int j = 0; j < S->nf; j++) {
            double sj = 0.0; for (int p = S->Ap[j]; p < S->Ap[j + 1]; p++) sj += S->Ax[p] * S->Ax[p] / si[S->Ai[p]];
            const double dj = fmin(S->sregf, fmax(frk * sj, frf));
            F->Ax[S->kdiag[j]] = -dj;
            if (dj < S->sregf) nlow++;
            if (dj < dmin) dmin = dj;
        }
        if (dbg_on()) socp_printf("      [the free variables' regularization relative to their curvature: %ld of %d below %.1e, the smallest %.1e]\n", nlow, S->nf, S->sregf, dmin);
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
static double g_ctime[6];     /* (debug: the centrality correctors' time: the correction term, the solves, the step, the copies; tried and rejected, accepted) */
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
    free(S->ne_kdcone); free(S->ne_kdpos); free(S->ne_kcoff); S->ne_kdcone = S->ne_kdpos = NULL; S->ne_kcoff = NULL; S->ne_nkc = 0;
    free(S->ne_kus); free(S->ne_kun); free(S->ne_kurow); free(S->ne_kuo); free(S->ne_kuoff); free(S->ne_kq);
    S->ne_kus = S->ne_kun = S->ne_kurow = NULL; S->ne_kuo = NULL; S->ne_kuoff = NULL; S->ne_kq = NULL; S->ne_keig = 0;
    if (S->ne) schol_free(S->ne);
    S->ne = NULL;
    free(S->ne_off); free(S->ne_doff); free(S->ne_G); free(S->ne_r); free(S->ne_C); S->ne_C = NULL;
    if (S->ne_rows) { free(S->ne_rows[0]); free(S->ne_rows); }
    S->ne_off = S->ne_doff = NULL; S->ne_G = NULL; S->ne_r = NULL; S->ne_rows = NULL;
    free(S->ne_W); S->ne_W = NULL; free(S->ne_T); S->ne_T = NULL; free(S->ne_xf); S->ne_xf = NULL; free(S->ne_Af); S->ne_Af = NULL;
    free(S->ne_wl); S->ne_wl = NULL; S->ne_nwl = 0; S->ne_wrank = 0;
    free(S->ne_dk); free(S->ne_drow); free(S->ne_dpos); free(S->ne_doffd); S->ne_dk = S->ne_drow = S->ne_dpos = NULL; S->ne_doffd = NULL; S->ne_nd = S->ne_ndc = S->ne_md = 0;
    free(S->ne_spk); free(S->ne_excol); free(S->ne_exk); free(S->ne_exr); if (S->ne_exrows) { free(S->ne_exrows[0]); free(S->ne_exrows); }
    free(S->ne_splk); free(S->ne_splg); free(S->ne_spld); S->ne_splk = NULL; S->ne_splg = NULL; S->ne_spld = NULL; S->ne_nspl = 0;
    free(S->ne_wU); free(S->ne_wW); free(S->ne_wC); free(S->ne_wpiv); free(S->ne_dg); S->ne_dg = NULL; S->ne_ntf = 0;
    S->ne_spk = S->ne_excol = S->ne_exk = S->ne_exr = NULL; S->ne_exrows = NULL; S->ne_wU = S->ne_wW = S->ne_wC = NULL; S->ne_wpiv = NULL; S->ne_nsp = S->ne_nex = S->ne_nu = 0; S->ne_wcap = 0; S->ne_onec = -1; S->ne_onec_ok = 0; S->ne_scale = 1.0;
    free(S->ne_dtcol); free(S->ne_dtk); S->ne_dtcol = S->ne_dtk = NULL; S->ne_ndt = 0; free(S->ne_xden); S->ne_xden = NULL;
}
/* (20 October 2026, LP/SOCP) C += alpha W W' (lower triangle, C r x r column-major, W r x k), by row blocks of 256 on
 * the threads: each block's diagonal tile by dsyrk, the rest of its rows by dgemm (disjoint parts of C: the same bits at
 * any thread count) */
static void syrk_blocks(int rr, int kk, double alpha, const double *W, double beta, double *C) {
    static int bs = -1; if (bs < 0) { const char *e_ = getenv("BRISK_CONESYRKBLK"); bs = e_ && atoi(e_) >= 16 ? atoi(e_) : 256; }   /* (a test: the rows of a block) */
    const int nbk = (rr + bs - 1) / bs;
    #pragma omp parallel for schedule(dynamic, 1) num_threads(g_cone_nt) if (g_cone_nt > 1 && !omp_in_parallel())
    for (int bk = nbk - 1; bk >= 0; bk--) {
        const int i0 = bk * bs, ib = rr - i0 < bs ? rr - i0 : bs;
        SC_BL(dsyrk_)("L", "N", &ib, &kk, &alpha, W + i0, &rr, &beta, C + (size_t)i0 * rr + i0, &rr);
        if (i0 > 0) SC_BL(dgemm_)("N", "T", &ib, &i0, &kk, &alpha, W + i0, &rr, W, &rr, &beta, C + i0, &rr);
    }
}
static int cmp_dbl(const void *a, const void *b) { const double x = *(const double *)a, y = *(const double *)b; return x < y ? -1 : x > y; }
static int cmp_ll(const void *a, const void *b) { const long long x = *(const long long *)a, y = *(const long long *)b; return x < y ? -1 : x > y; }
static int cmp_int(const void *a, const void *b) { const int x = *(const int *)a, y = *(const int *)b; return x < y ? -1 : x > y; }
/* (23 October 2026) the rows that fix one entry of a small cone: a row with a single entry, in a column of a cone of at
 * most NE_MAXCONE entries; one such row a cone (a second stays a row). Such a row touches no other block, so the normal
 * equations can do without it: its unknown is eliminated by its own equation, and the cone enters the rows that remain
 * through the Schur complement of its fixed entry, H_k^-1 - h h' / h_t (h = H_k^-1 e_t). CBLIB's 2013_firL1 and
 * 2013_firL2L1eps (SOCPs): 19,902 cones (t, u, v) with t = 1 by such a row, next to 1,000 dense rows; MOSEK's presolve
 * removes the rows likewise (20,902 constraints -> 1,000). With elim NULL: only counted. BRISK_CONENOFIX=1: none. */
static int find_fixed_d(const Socp *S, char *elim, int *frow, int *fpos, double *fa, int maxd);
static int ne_find_fixed(const Socp *S, char *elim, int *frow, int *fpos, double *fa) { return find_fixed_d(S, elim, frow, fpos, fa, NE_MAXCONE); }
/* (8 November 2026, the SOCP developer) the cones of at most maxd entries with an entry fixed by a row of its own: the normal
 * equations eliminate those rows (maxd NE_MAXCONE, the small cones they treat entry by entry); the start places a cone whose
 * first entry is so fixed at the dilation of its centre, which holds for a cone of any size (start_fix_big) */
static int find_fixed_d(const Socp *S, char *elim, int *frow, int *fpos, double *fa, int maxd) {
    if (S->nf > 0 || getenv("BRISK_CONENOFIX")) return 0;
    const int m = S->m;
    int *cnt = (int *)calloc((size_t)m + 1, sizeof(int)); if (!cnt) return 0;
    for (int p = 0; p < S->Ap[S->n]; p++) cnt[S->Ai[p]]++;
    char *own = elim ? elim : (char *)calloc((size_t)m + 1, 1);
    int nfix = 0;
    for (int kc = 0; kc < S->nq; kc++) {
        if (frow) frow[kc] = -1;
        const int d = S->q[kc]; if (d > maxd || d < 2) continue;
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
    /* (29 October 2026, the SOCP developer; ISSUES 804) fewer dense columns than their rows: the full system's dense part is
     * then at least the dense columns' block, w + nd^3 / 3; the analysis is put off only when the normal equations at their
     * largest (all m(m+1)/2 pairs: the work (pairs^2 / m) of ne_setup's test) pass that test against this work, so that they
     * are certain to be taken. CBLIB's 2013_firL2a (SOCP): 6,839 dense columns over 10,001 rows, the full system analysed for
     * 36 s (6.67e11 flops, the estimate 6.56e11) and never used, the normal equations taken (3.3e11). BRISK_CONEDEFERFEW=0: off */
    static int few = -1; if (few < 0) { const char *e_ = getenv("BRISK_CONEDEFERFEW"); few = !(e_ && atoi(e_) == 0); }
    const double wf = w + (double)nd * nd * nd / 3.0, pmax = 0.5 * (double)m * ((double)m + 1.0), wne = pmax * pmax / fmax((double)m, 1.0);
    const double r = nd > 1 && nd >= mda ? w + (double)mda * mda * mda / 3.0 : few && nd > 1 && wne <= 1.5 * wf ? wf : 0;
    if (dbg_on()) socp_printf("  [dense columns for the deferral: %d dense columns over %d rows (%d of linear columns and large cones), the full system's work %.3g, the normal equations' at most %.3g: %s]\n", nd, mda, md, nd >= mda ? w + (double)mda * mda * mda / 3.0 : wf, wne, r > 0 ? "put off" : "not put off");
    return r;
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
    /* (13 November 2026, the SOCP developer, a test: BRISK_CONEPARTNE) the partial normal equations: the selected small cones (ne_kcsel,
     * chosen where the normal equations lose their accuracy) leave the normal matrix and keep their d unknowns, with their block
     * -(H_k + delta) and their columns of A: [S A_K; A_K' -H_K] (dy; dx_K) = (ry + A_s H_s^-1 rx_s; rx_K), quasi-definite. The cones
     * whose weights lie between those of the inactive cones (about 1/mu) and the active ones (about mu), the ones without strict
     * complementarity, no longer add to the rows' sums, where past a spread of 1e16 the small parts were rounded away
     * (db-joint-soerensen, SOCP). Not with the free variables, the Woodbury update, the dense block or a fixed entry of the cone */
    S->ne_nkc = 0;
    if (S->ne_kcsel && S->nf == 0 && S->ne_nwl == 0 && S->ne_md == 0 && S->ne_nsp == 0) {
        int nk_ = 0;
        for (int kc = 0; kc < S->nq; kc++) if (S->ne_kcsel[kc] && S->q[kc] <= NE_MAXCONE && !(S->ne_nfix > 0 && S->ne_fixrow[kc] >= 0)) nk_ += S->q[kc];
        if (nk_ > 0) {
            const int nk0 = S->ne_nkd, nkt = nk0 + nk_;
            int *kdc = (int *)realloc(S->ne_kdcol, sizeof(int) * (size_t)nkt);
            S->ne_kdcone = (int *)malloc(sizeof(int) * (size_t)nkt); S->ne_kdpos = (int *)malloc(sizeof(int) * (size_t)nkt);
            if (kdc && S->ne_kdcone && S->ne_kdpos) {
                S->ne_kdcol = kdc;
                for (int a = 0; a < nk0; a++) { S->ne_kdcone[a] = -1; S->ne_kdpos[a] = 0; }
                int a = nk0;
                for (int kc = 0; kc < S->nq; kc++) if (S->ne_kcsel[kc] && S->q[kc] <= NE_MAXCONE && !(S->ne_nfix > 0 && S->ne_fixrow[kc] >= 0)) {
                    const int k = S->nl + kc;
                    for (int t = 0; t < S->q[kc]; t++) { S->ne_kdcol[a] = S->qs[kc] + t; S->ne_kdcone[a] = kc; S->ne_kdpos[a] = t; a++; }
                    pairs -= 0.5 * (double)r[k] * (r[k] + 1); tot -= r[k]; r[k] = 0; S->ne_nkc++;
                }
                S->ne_nkd = nkt;
                /* (BRISK_CONEPARTNEEIG=0: the cone's own entries, its block H_k dense) each kept unknown is an eigenvector's coordinate of the
                 * cone's scaling and couples to the union of the cone's columns' rows; its block is diagonal, -(lambda + delta) */
                static int peig = -1; if (peig < 0) { const char *e_ = getenv("BRISK_CONEPARTNEEIG"); peig = e_ ? atoi(e_) : 1; }
                if (peig) {
                    long tot_ = 0; for (int b = nk0; b < nkt; b++) if (S->ne_kdpos[b] == 0) { const int kc = S->ne_kdcone[b]; for (int t = 0; t < S->q[kc]; t++) tot_ += S->Ap[S->qs[kc] + t + 1] - S->Ap[S->qs[kc] + t]; }
                    S->ne_kus = (int *)malloc(sizeof(int) * (size_t)nkt); S->ne_kun = (int *)malloc(sizeof(int) * (size_t)nkt);
                    S->ne_kurow = (int *)malloc(sizeof(int) * ((size_t)tot_ + 1)); S->ne_kq = (double *)calloc((size_t)nkt * NE_MAXCONE, sizeof(double));
                    if (S->ne_kus && S->ne_kun && S->ne_kurow && S->ne_kq) {
                        long o_ = 0;
                        for (int b = 0; b < nkt; b++) {
                            if (b < nk0) { S->ne_kus[b] = 0; S->ne_kun[b] = -1; continue; }
                            if (S->ne_kdpos[b] > 0) { S->ne_kus[b] = S->ne_kus[b - 1]; S->ne_kun[b] = S->ne_kun[b - 1]; continue; }
                            const int kc = S->ne_kdcone[b]; int *ru = S->ne_kurow + o_; int nu = 0;
                            for (int t = 0; t < S->q[kc]; t++) for (int pp = S->Ap[S->qs[kc] + t]; pp < S->Ap[S->qs[kc] + t + 1]; pp++) ru[nu++] = S->Ai[pp];
                            qsort(ru, (size_t)nu, sizeof(int), cmp_int);
                            int nu2 = 0; for (int e = 0; e < nu; e++) if (e == 0 || ru[e] != ru[e - 1]) ru[nu2++] = ru[e];
                            S->ne_kus[b] = (int)o_; S->ne_kun[b] = nu2; o_ += nu2;
                        }
                        S->ne_keig = 1;
                    } else { free(S->ne_kus); free(S->ne_kun); free(S->ne_kurow); free(S->ne_kq); S->ne_kus = S->ne_kun = S->ne_kurow = NULL; S->ne_kq = NULL; }
                }
                const size_t l_ = strlen(S->ne_dense_msg); snprintf(S->ne_dense_msg + l_, sizeof S->ne_dense_msg - l_, "; %d cones kept as unknowns (%d unknowns)", S->ne_nkc, nk_);
            } else { S->ne_kdcol = kdc; free(S->ne_kdcone); free(S->ne_kdpos); S->ne_kdcone = S->ne_kdpos = NULL; }
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
      static int nerate0_c = -2; const int nerate0 = cone_on("BRISK_CONENERATE", &nerate0_c);
      const double lnz_eq = nerate0 && !(S->sc || S->sc_lazy || S->meas_sup == 1) && nerate_dense(S) ? (double)S->F.lnz * (6e9 / 1.2e9) : (double)S->F.lnz;   /* (BRISK_CONENERATE: the column code's slower rate, below) */
      /* (7 November 2026, the SOCP developer; the owner's SOCP benchmark of 6 November; the default, BRISK_CONENEWORK=0: off) the bound P^2 / m
       * holds for P the factor's entries, at most m (m + 1) / 2; the pairs as counted add each block's triangle, the same triangle
       * once per cone over the same rows, and can exceed it: two cones over all 8,596 rows of a trust-region SOCP (P04_01) counted
       * m^2 pairs, a "lower bound" of m^3 = 6.4e11 against the full system's 4.0e11 work, and the normal equations of about
       * m^3 / 3 = 2.1e11 were refused */
      static int nework = -1; if (nework < 0) { const char *e_ = getenv("BRISK_CONENEWORK"); nework = !(e_ && atoi(e_) == 0); }
      const double pe_ = nework ? fmin(pairs, 0.5 * (double)m * ((double)m + 1.0)) : pairs;
      if (!getenv("BRISK_CONENE") && !S->kkt_defer && pairs > 0.75 * lnz_eq && (bynnz || !dense_ || pe_ * pe_ / fmax((double)m, 1.0) > 1.5 * S->cflops + 1e6)) { free(r); free(mark); ne_free(S); return 0; } }
    /* (1 November 2026, the SOCP developer; the owner's SOCP benchmark, the largest wide cones) a block over all m rows makes the
     * pattern complete: the factor is one dense supernode of m (m + 1) / 2 entries in the given order, and a large cone's block
     * takes its places in it by formula (base + j m + i, its rows sorted), with no table. So the caps count what the route
     * stores: the factor's entries and the table of the other blocks; the pairs as counted add each block's triangle, the
     * same triangle once per cone over the same rows. A047 (SOCP: two cones over all 20,002 rows) counted 4.0e8 pairs against
     * the cap of 4e8 for a factor of 2.0e8 entries; refused, the full system was ordered (261 s) for the same factor work
     * (2.67e12 flops). BRISK_CONENEFULL=0: off */
    S->ne_full = 0; S->ne_dfull = 0; S->ne_dbase = 0;
    double tpairs = pairs;           /* (the pairs of the table) */
    { static int fon = -1; if (fon < 0) { const char *e_ = getenv("BRISK_CONENEFULL"); fon = !(e_ && atoi(e_) == 0); }
      if (fon && m > 1 && !elim && S->ne_nkd == 0) {
          /* (a block's rows are a clique of the pattern: when the largest holds at least 90 % of the complete triangle, the exact
           * pattern's factor does at least (r/m)^3 > 0.85 of the complete one's work, so the pattern is taken as complete, its
           * few missing pairs as explicit zeros; A047: two cones over 20,001 of the 20,002 rows) */
          int rmax = 0; for (int k = 0; k < nb; k++) if (r[k] > rmax) rmax = r[k];
          if (0.5 * (double)rmax * (rmax + 1.0) >= 0.9 * 0.5 * (double)m * (m + 1.0)) S->ne_full = 1;
          /* (7 November 2026, the SOCP developer; with BRISK_CONENEWORK, the default) a pattern of at least half the triangle, from
           * many blocks: its factor does at most the complete one's m^3 / 3, which decides at once against the full system's
           * work, known here: above it, refused without the sparse analysis (a network-capacity SOCP, P12_00: 79 % of the
           * triangle, 10 s of analysis to find a factor above the limit; a group-sparse SOCP, P07_01: 12.6 s for 5.1e11 flops
           * against the full system's 4.35e11, the same factorization time); below, taken as complete (the dense factor costs no
           * more than its bound). Not 1.5 times, the allowance of the sparse estimate for the dense kernel's rate: at these
           * sizes both factorizations run at the same rate (P07_01), and a redundant-coupling SOCP (A04_02) took 10 iterations
           * of 3.82e11 against 8 of the full system's 3.31e11 */
          else { static int nwh = -1; if (nwh < 0) { const char *e_ = getenv("BRISK_CONENEWORK"); nwh = !(e_ && atoi(e_) == 0); }
              const double T_ = 0.5 * (double)m * (m + 1.0);
              if (nwh && !S->kkt_defer && S->cflops > 0 && !getenv("BRISK_CONENE") && fmin(pairs, T_) >= 0.5 * T_) {
                  if ((double)m * m * (double)m / 3.0 <= S->cflops) S->ne_full = 1;
              } }
          /* (2 November 2026) the table also holds the sparse cones' tail columns' pairs (exp_, ne_exrows): without them
           * the table was allocated short and their offsets written past its end (CBLIB 2013_wbNRL, SOCP: heap corruption) */
          if (S->ne_full) { tpairs = exp_; for (int k = 0; k < nb; k++) if (!(k >= S->nl && S->q[k - S->nl] > NE_MAXCONE)) tpairs += 0.5 * (double)r[k] * (r[k] + 1); }
      } }
    /* (7 November 2026; with BRISK_CONENEWORK, the default) the refusal of a pattern of at least half the triangle by the complete factor's work,
     * over the rows left after the fixed rows' elimination (which the complete route does not take: the sparse analysis then) */
    { static int nwr = -1; if (nwr < 0) { const char *e_ = getenv("BRISK_CONENEWORK"); nwr = !(e_ && atoi(e_) == 0); }
      const double mr_ = (double)(m - (elim ? S->ne_nfix : 0)), T_ = 0.5 * mr_ * (mr_ + 1.0);
      if (nwr && !S->ne_full && !S->kkt_defer && S->cflops > 0 && !getenv("BRISK_CONENE") && mr_ > 1 && fmin(pairs, T_) >= 0.5 * T_ && mr_ * mr_ * mr_ / 3.0 > S->cflops) {
          if (dbg_on()) socp_printf("  [normal equations: %.0f%% of the triangle, the complete factor's %.3g flops above the full system's %.3g: refused without analysis]\n", 100.0 * fmin(pairs, T_) / T_, mr_ * mr_ * mr_ / 3.0, S->cflops);
          free(r); free(mark); ne_free(S); return 0;
      } }
    const double cpairs = S->ne_full ? tpairs + 0.5 * (double)m * (m + 1) : pairs;   /* (what the route stores: the table, and the dense factor) */
    if (S->ne_full && dbg_on()) socp_printf("  [normal equations: a block over nearly all of the %d rows, the pattern taken as complete: %.0f entries of the factor and %.0f of the table]\n", m, 0.5 * (double)m * (m + 1), tpairs);
    /* (7 November 2026, the SOCP developer; with BRISK_CONENEWORK, the default) the limit of 40 entries of A for each entry of the normal
     * equations stands for the work when the full system's work is not known; when it is (analysed, not put off), the work
     * decides: the work bound above has passed, and the complete pattern's factor does m^3 / 3 (P04_01, a trust-region SOCP:
     * 3.7e7 entries against 40 times 168,438, refused for 2.1e11 flops against the full system's 4.0e11). The memory cap
     * stays */
    static int nework2 = -1; if (nework2 < 0) { const char *e_ = getenv("BRISK_CONENEWORK"); nework2 = !(e_ && atoi(e_) == 0); }
    const double newk_ = S->ne_full ? (double)m * m * (double)m / 3.0 : 0.0;
    const int bywork_ = nework2 && !S->kkt_defer && S->cflops > 0 && !getenv("BRISK_CONENE") && (S->ne_full ? newk_ <= S->cflops : 1);
    if ((!bywork_ && (S->ne_full ? 0.5 * (double)m * (m + 1) : pairs) > 40.0 * (double)S->Ap[n] + 1e5) || cpairs > 2e8 * brisk_mem_scale()) { free(r); free(mark); ne_free(S); return 0; }        /* dense columns */
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
    const double tne0_ = sc_time();
    /* (1 November 2026) the pattern taken as complete: the dense analysis, no pairs and no lists (schol_analyze_dense; the
     * factor is the one the general analysis gives a complete pattern in the given order). BRISK_CONENEDENSEAN=0: off */
    SChol *schol_analyze_dense(int m, int inplace);
    static int dan_on = -1; if (dan_on < 0) { const char *e_ = getenv("BRISK_CONENEDENSEAN"); dan_on = !(e_ && atoi(e_) == 0); }
    SChol *sc = (dan_on && S->ne_full && S->ne_nkd == 0) ? schol_analyze_dense(m, !getenv("BRISK_CONENOINPLACE")) : NULL;
    const int dense_an_ = sc != NULL;
    if (dense_an_ && dbg_on()) socp_printf("  [normal equations: the dense analysis %.3fs (%d rows)]\n", sc_time() - tne0_, m);
    const int mm = m + S->ne_nkd;          /* (the rows, then the dense linear columns kept as unknowns) */
    if (!dense_an_) {
    const size_t np = pairs > (double)tot ? (size_t)(pairs - (double)tot) : 0;   /* off-diagonal pairs with repetitions */
    /* (29 October 2026, the SOCP developer) the distinct pairs by a bit table of the lower triangle when the rows are few and
     * the pairs many (a dense normal matrix: H_A_scaling_3072, SOCP, 9.4 million pairs over 3,075 rows: the sort 0.55 s),
     * otherwise by sorting; both give the pairs in the same order (by row, then column), so the neighbour lists, the ordering
     * and the factor are the same */
    const int use_bm = !S->ne_full && !getenv("BRISK_CONENEPAIRSORT") && m <= 16384 && (double)np >= (double)m * m / 64.0;
    long long *key = NULL; uint64_t *bm = NULL; long q = 0, u = 0;
#define NE_BSET(i_, j_) do { const size_t b__ = (size_t)(i_) * (size_t)m + (size_t)(j_); bm[b__ >> 6] |= (uint64_t)1 << (b__ & 63); } while (0)
    if (use_bm) bm = (uint64_t *)calloc(((size_t)m * m >> 6) + 1, sizeof(uint64_t));
    if (S->ne_full) {
        u = (long)m * (m - 1) / 2; q = (long)np;      /* (complete: every pair, no enumeration; the lists below in the pairs' order) */
    } else if (bm) {
        for (int k = 0; k < nb; k++) for (int a = 1; a < r[k]; a++) for (int b_ = 0; b_ < a; b_++) { NE_BSET(rows[k][a], rows[k][b_]); q++; }
        for (int e_ = 0; e_ < S->ne_nex; e_++) { const int *rw = S->ne_exrows[e_]; for (int a = 1; a < S->ne_exr[e_]; a++) for (int b_ = 0; b_ < a; b_++) { NE_BSET(rw[a], rw[b_]); q++; } }
        for (int a = 1; a < S->ne_md; a++) for (int b_ = 0; b_ < a; b_++) { const int i = S->ne_drow[a], j = S->ne_drow[b_]; NE_BSET(i > j ? i : j, i > j ? j : i); q++; }
        for (size_t w = 0; w < ((size_t)m * m >> 6) + 1; w++) u += __builtin_popcountll(bm[w]);
    } else {
        key = (long long *)malloc(sizeof(long long) * (np + 1));
        for (int k = 0; k < nb; k++) for (int a = 1; a < r[k]; a++) for (int b_ = 0; b_ < a; b_++) key[q++] = ((long long)rows[k][a] << 32) | (unsigned)rows[k][b_];
        for (int e_ = 0; e_ < S->ne_nex; e_++) { const int *rw = S->ne_exrows[e_]; for (int a = 1; a < S->ne_exr[e_]; a++) for (int b_ = 0; b_ < a; b_++) key[q++] = ((long long)rw[a] << 32) | (unsigned)rw[b_]; }
        for (int a = 1; a < S->ne_md; a++) for (int b_ = 0; b_ < a; b_++) {
            const int i = S->ne_drow[a], j = S->ne_drow[b_];
            key[q++] = ((long long)(i > j ? i : j) << 32) | (unsigned)(i > j ? j : i);
        }
        qsort(key, (size_t)q, sizeof(long long), cmp_ll);
        for (long e = 0; e < q; e++) if (e == 0 || key[e] != key[e - 1]) key[u++] = key[e];
    }
    /* the pairs in order: by the table's bits row by row, or the sorted keys */
#define NE_FORPAIRS(BODY) do { if (S->ne_full) { for (int i = 1; i < m; i++) for (int j = 0; j < i; j++) { BODY; } } else if (bm) { for (int i = 1; i < m; i++) { const size_t r0 = (size_t)i * m; for (int j = 0; j < i; j++) if (bm[(r0 + j) >> 6] >> ((r0 + j) & 63) & 1) { BODY; } } } \
                               else { for (long e = 0; e < u; e++) { const int i = (int)(key[e] >> 32), j = (int)(key[e] & 0xffffffffLL); BODY; } } } while (0)
    int *deg = (int *)calloc((size_t)mm + 1, sizeof(int)), **nbr = (int **)malloc(sizeof(int *) * ((size_t)mm + 1));
    NE_FORPAIRS(deg[i]++; deg[j]++);
    long kdn = 0;
    for (int a = 0; a < S->ne_nkd; a++) {
        if (S->ne_keig && S->ne_kun[a] >= 0) { for (int e = 0; e < S->ne_kun[a]; e++) { deg[S->ne_kurow[S->ne_kus[a] + e]]++; deg[m + a]++; kdn++; } continue; }
        const int j = S->ne_kdcol[a]; for (int p = S->Ap[j]; p < S->Ap[j + 1]; p++) { deg[S->Ai[p]]++; deg[m + a]++; kdn++; } }
    /* (BRISK_CONEPARTNE) a kept cone's unknowns are coupled by its block H_k: every pair of them (none in its eigenbasis) */
    long kcn = 0;
    for (int a = 0; S->ne_kdcone && !S->ne_keig && a < S->ne_nkd; a++) { const int kc = S->ne_kdcone[a]; if (kc < 0) continue;
        for (int b_ = a - S->ne_kdpos[a]; b_ < a; b_++) { deg[m + a]++; deg[m + b_]++; kcn++; } }
    int *npool = (int *)malloc(sizeof(int) * (size_t)(2 * u + 2 * kdn + 2 * kcn + 1));
    { long o = 0; for (int i = 0; i < mm; i++) { nbr[i] = npool + o; o += deg[i]; deg[i] = 0; } }
    NE_FORPAIRS(nbr[i][deg[i]++] = j; nbr[j][deg[j]++] = i);
    for (int a = 0; a < S->ne_nkd; a++) { const int j = S->ne_kdcol[a], d_ = m + a;
        if (S->ne_keig && S->ne_kun[a] >= 0) { for (int e = 0; e < S->ne_kun[a]; e++) { const int i = S->ne_kurow[S->ne_kus[a] + e]; nbr[d_][deg[d_]++] = i; nbr[i][deg[i]++] = d_; } continue; }
        for (int p = S->Ap[j]; p < S->Ap[j + 1]; p++) { const int i = S->Ai[p]; nbr[d_][deg[d_]++] = i; nbr[i][deg[i]++] = d_; } }
    for (int a = 0; S->ne_kdcone && !S->ne_keig && a < S->ne_nkd; a++) { const int kc = S->ne_kdcone[a]; if (kc < 0) continue;
        for (int b_ = a - S->ne_kdpos[a]; b_ < a; b_++) { nbr[m + a][deg[m + a]++] = m + b_; nbr[m + b_][deg[m + b_]++] = m + a; } }
    free(key); free(bm);
#undef NE_FORPAIRS
#undef NE_BSET
    long npair_ = 0; for (int i = 0; i < mm; i++) npair_ += deg[i];
    const int om = schol_set_amd(2); const double oam = schol_set_amalg(cone_amalg(mm > 0 ? 0.5 * (double)npair_ / mm : 0.0));
    const double tne1_ = sc_time();
    /* (29 October 2026, the SOCP developer) a complete pattern: every order gives the same factor, so the given order is
     * used instead of the minimum-degree ordering (CBLIB's 2013_firL2a, SOCP: 10,001 rows, 5.0e7 pairs, the ordering about
     * 6 s of the analysis' 7.7 s). BRISK_CONENEFULLAMD=1: ordered as before */
    int *idp_ = NULL;
    { static int fa = -1; if (fa < 0) fa = getenv("BRISK_CONENEFULLAMD") != NULL;
      if (!fa && mm > 1 && npair_ == (long)mm * (mm - 1)) { idp_ = (int *)malloc(sizeof(int) * (size_t)mm); if (idp_) { for (int i = 0; i < mm; i++) idp_[i] = i; schol_set_perm(idp_); } } }
    /* (31 October 2026, the SOCP developer; ISSUES 809, the owner's SOCP benchmark: robust inequalities) the analysis stops
     * once its counted work exceeds the bound at which the test below refuses the normal equations (1.5 times the full
     * system's work): P028 (SOCP, 54,036 rows, 6.5 million pairs) analysed them for 27.3 s of a 35.8 s solve and 4.8 GB to
     * refuse them. Not with the minimum-degree argument (their work is the full system's then) or BRISK_CONENE (forced) */
    const int fcap_ = !getenv("BRISK_CONENE") && !S->kkt_defer_md && !getenv("BRISK_CONENENOFCAP");
    static int nerate1_c = -2; const int nerate1 = cone_on("BRISK_CONENERATE", &nerate1_c);
    const double cfq = nerate1 && !(S->sc || S->sc_lazy || S->meas_sup == 1) && nerate_dense(S) ? S->cflops * (6e9 / 1.2e9) : S->cflops;   /* (BRISK_CONENERATE: the full system's flops at the supernodal rate, as below) */
    /* first the minimum-degree ordering alone, stopped once its own running count of the factor's work passes twice that
     * bound (the count of the pivots so far, slight upper bounds of their work by approximate degrees: twice the bound
     * leaves the decision to the exact count below whenever it is near); stopped, the normal equations are refused
     * without the symbolic factorization (P031: the analysis 21.4 s, of which the counting up to the cap most) */
    double afl_ne = -1;      /* (AMD's estimated flops, for the choice of METIS's order below; -1: not counted) */
    int amd_over_ne = 0;
    if (fcap_ && !idp_) {      /* (a complete pattern is analysed in its given order: no ordering to run twice) */
        const volatile double acap = 2.0 * (1.5 * cfq + 1e6); double afl = 0, anz = 0;
        const int rce_ = schol_amd_estimate_cap(mm, deg, nbr, &acap, &afl, &anz);
        if (rce_ == 0) afl_ne = afl;
#ifdef HAVE_METIS
        if (rce_ == 2) { static int nm0 = -1; if (nm0 < 0) { const char *e_ = getenv("BRISK_CONENEMETIS"); nm0 = e_ ? atoi(e_) : 2; } if (nm0 && !S->ne_kdcone && mm >= 1000) amd_over_ne = 1; }
#endif
        if (rce_ == 2 && !amd_over_ne) {
            if (dbg_on()) socp_printf("  [normal equations: the ordering's count of their work passed %.3g (twice the refusal bound) after %.3fs: refused]\n", (double)acap, sc_time() - tne1_);
            schol_set_amd(om); schol_set_amalg(oam); if (idp_) { schol_set_perm(NULL); free(idp_); }
            free(deg); free(nbr); free(npool); free(mark); free(r); free(pool); free(rows); ne_free(S); return 0;
        }
    }
    if (fcap_) schol_set_flopcap(1.5 * cfq + 1e6);
    /* (4 November 2026, the SOCP developer, a test: BRISK_CONENEND=1) nested dissection as a candidate order of the normal
     * equations (the factorization library's trial, kept when its factor costs fewer flops; the LP method's rule): a
     * finite-element structure such as db-joint-soerensen, SOCP (4.97e9 flops by minimum degree), is the case it is made for */
    static int nend = -1; if (nend < 0) { const char *e_ = getenv("BRISK_CONENEND"); nend = e_ ? atoi(e_) : 0; }
    extern void schol_set_nd(int on);
    if (nend) schol_set_nd(1);
    /* (13 November 2026, the SOCP developer; with the partial normal equations, BRISK_CONEPARTNEORD=0: AMD on the whole pattern) the kept
     * unknowns after their rows: the rows ordered by AMD alone, each kept unknown placed right after the last of its rows. By AMD on the
     * whole pattern 26,388 of 26,409 kept unknowns of db-joint-soerensen (SOCP) came before some of their rows; eliminated first, a kept
     * unknown adds its cone's H^-1 back into the rows, which rebuilds the normal equations and their loss of accuracy */
    int *kperm_ = NULL;
    { static int pord = -1; if (pord < 0) { const char *e_ = getenv("BRISK_CONEPARTNEORD"); pord = e_ ? atoi(e_) : 1; }
      if (pord && S->ne_nkc > 0 && S->ne_kdcone) {
          int *d2 = (int *)calloc((size_t)m + 1, sizeof(int)), **n2 = (int **)malloc(sizeof(int *) * ((size_t)m + 1));
          long t2 = 0; for (int i = 0; i < m; i++) for (int e = 0; e < deg[i]; e++) if (nbr[i][e] < m) t2++;
          int *p2 = (int *)malloc(sizeof(int) * ((size_t)t2 + 1));
          if (d2 && n2 && p2) {
              long o_ = 0; for (int i = 0; i < m; i++) { n2[i] = p2 + o_; for (int e = 0; e < deg[i]; e++) if (nbr[i][e] < m) p2[o_++] = nbr[i][e]; d2[i] = (int)(o_ - (n2[i] - p2)); }
              SChol *sr = schol_analyze_adj(m, d2, n2, (size_t)1 << 31);
              if (sr) {
                  const int *pr = schol_perm(sr); int *pos = (int *)malloc(sizeof(int) * ((size_t)m + 1)), *head = (int *)malloc(sizeof(int) * ((size_t)m + 2)), *nxt = (int *)malloc(sizeof(int) * ((size_t)S->ne_nkd + 1));
                  kperm_ = (int *)malloc(sizeof(int) * ((size_t)mm + 1));
                  if (pos && head && nxt && kperm_) {
                      for (int t = 0; t < m; t++) pos[pr[t]] = t;
                      for (int t = 0; t <= m; t++) head[t] = -1;
                      for (int a = S->ne_nkd - 1; a >= 0; a--) {
                          int last = -1;
                          for (int e = 0; e < deg[m + a]; e++) { const int i = nbr[m + a][e]; if (i < m && pos[i] > last) last = pos[i]; }
                          const int b = last < 0 ? m : last; nxt[a] = head[b]; head[b] = a;
                      }
                      int o2 = 0;
                      for (int t = 0; t <= m; t++) { if (t < m) kperm_[o2++] = pr[t]; for (int a = head[t]; a >= 0; a = nxt[a]) kperm_[o2++] = m + a; }
                      if (o2 == mm) { schol_set_perm(kperm_); schol_set_perm_only(1); } else { free(kperm_); kperm_ = NULL; }
                  } else { free(kperm_); kperm_ = NULL; }
                  free(pos); free(head); free(nxt); schol_free(sr);
              }
          }
          free(d2); free(n2); free(p2);
      } }
#ifdef HAVE_METIS
    /* (13 November 2026, the SOCP developer) METIS's nested dissection (the bundled metis/, parallel, the same order at any number of
     * threads) as a second candidate order, chosen before any analysis: AMD's flops come from the estimate above, METIS's from the exact
     * counts of its order (schol_exact_counts), and only the order kept is analysed. Tried where it can pay: its time, about 6.4e-7 s an
     * entry of the graph on one thread (2.6 times faster on four), at most a quarter of AMD's factorizations over a solve (30 at 2.5e9
     * flops a second a thread, and those at least 0.2 s), and not where AMD's factor fills more than a quarter of the triangle (no
     * separator worth ordering last, nd.c's rule for the LP method); kept when its factor costs less than 0.9 of AMD's, or when AMD's
     * passed the cap (the normal equations otherwise refused) and METIS's is within it. db-joint-soerensen (SOCP): 4.97e9 -> 7.34e8
     * flops, an iteration 1.4 -> 0.9 s. BRISK_CONENEMETIS=0: AMD alone; 1: always tried */
    int *mperm_ = NULL;
    { static int nmet = -1; if (nmet < 0) { const char *e_ = getenv("BRISK_CONENEMETIS"); nmet = e_ ? atoi(e_) : 2; }
      if (nmet && !kperm_ && !idp_ && mm >= 1000) {
          if (afl_ne < 0 && !amd_over_ne) { double a1 = 0, a2 = 0; if (schol_amd_estimate(mm, deg, nbr, &a1, &a2) == 0) afl_ne = a1; }
          long tot_ = 0; for (int i = 0; i < mm; i++) tot_ += deg[i];
          const int nt_ = g_cone_nt > 0 ? g_cone_nt : 1;
          const double capf_ = fcap_ ? 1.5 * cfq + 1e6 : 1e300;
          const double fa_ = amd_over_ne ? capf_ : afl_ne;
          const double tmet_ = 6.4e-7 * (double)tot_ / fmin((double)nt_, 2.6), tfac_ = 30.0 * fa_ / (2.5e9 * nt_);
          /* (AMD's factor nearly dense: its flops above a quarter of the complete factor's m^3 / 3; socp_rlsinf_20000x5000xnear, SOCP: 3.79e10
           * of 4.2e10, METIS's order 3.88e10 and its trial 6.56 -> 7.27 s) */
          const int dense_l = !amd_over_ne && afl_ne > 0 && afl_ne > 0.25 * (double)mm * (double)mm * (double)mm / 3.0;
          if (fa_ > 0 && !dense_l && (nmet == 1 || amd_over_ne || (tfac_ >= 0.2 && tmet_ <= 0.25 * tfac_))) {
              extern int METIS_NodeND(int *nvtxs, int *xadj, int *adjncy, int *vwgt, int *options, int *perm, int *iperm);
              extern void BRISK_METIS_SetThreads(int nt);
              int *xa = (int *)malloc(sizeof(int) * ((size_t)mm + 1)), *ad = (int *)malloc(sizeof(int) * ((size_t)tot_ + 1)), *ip_ = (int *)malloc(sizeof(int) * ((size_t)mm + 1));
              mperm_ = (int *)malloc(sizeof(int) * ((size_t)mm + 1));
              if (xa && ad && ip_ && mperm_) {
                  long o_ = 0; xa[0] = 0;
                  for (int i = 0; i < mm; i++) { for (int e = 0; e < deg[i]; e++) if (nbr[i][e] != i) ad[o_++] = nbr[i][e]; xa[i + 1] = (int)o_; }
                  int nv = mm; const double tm_ = sc_time();
                  BRISK_METIS_SetThreads(nt_);
                  double f2 = 1e300, n2 = 0;
                  if (METIS_NodeND(&nv, xa, ad, NULL, NULL, mperm_, ip_) != 1 || schol_exact_counts(mm, deg, nbr, mperm_, &f2, &n2) != 0) f2 = 1e300;
                  const int take = amd_over_ne ? f2 <= capf_ : f2 < 0.9 * fa_;
                  if (dbg_on()) socp_printf("  [normal equations: METIS's order %.3fs (estimated %.2fs against %.2fs of AMD's factorizations), %.3g flops against AMD's %.3g%s: %s]\n", sc_time() - tm_, tmet_, tfac_, f2, fa_, amd_over_ne ? " (AMD's over the cap)" : "", take ? "taken" : "AMD's kept");
                  if (take) { schol_set_perm(mperm_); schol_set_perm_only(1); } else { free(mperm_); mperm_ = NULL; }
              } else { free(mperm_); mperm_ = NULL; }
              free(xa); free(ad); free(ip_);
          }
      }
      if (amd_over_ne && !mperm_) {      /* (AMD's order over the cap and METIS's no better: refused, as before) */
          if (dbg_on()) socp_printf("  [normal equations: the ordering's count of their work passed twice the refusal bound: refused]\n");
          if (fcap_) schol_set_flopcap(0);
          if (nend) schol_set_nd(0);
          schol_set_amd(om); schol_set_amalg(oam);
          free(deg); free(nbr); free(npool); free(mark); free(r); free(pool); free(rows); ne_free(S); return 0;
      } }
#endif
    sc = schol_analyze_adj(mm, deg, nbr, (size_t)1 << 31);
    if (kperm_) { schol_set_perm(NULL); schol_set_perm_only(0); free(kperm_); if (dbg_on()) socp_printf("  [normal equations: the kept unknowns ordered after their rows]\n"); }
#ifdef HAVE_METIS
    if (mperm_) { schol_set_perm(NULL); schol_set_perm_only(0); free(mperm_); }
#endif
    if (nend) schol_set_nd(0);
    if (fcap_) schol_set_flopcap(0);
    if (dbg_on() && fcap_ && !sc) socp_printf("  [normal equations: the analysis stopped above 1.5 times the full system's work (%.3g flops)]\n", S->cflops);
    if (idp_) { schol_set_perm(NULL); free(idp_); }
    schol_set_amd(om); schol_set_amalg(oam);
    if (dbg_on()) socp_printf("  [normal equations: the pattern %.3fs (%ld distinct pairs of %ld), its analysis %.3fs]\n", tne1_ - tne0_, u, q, sc_time() - tne1_);
    free(deg); free(nbr); free(npool);
    }                                      /* (the general analysis) */
    free(mark);
    if (!sc) { free(r); free(pool); free(rows); ne_free(S); return 0; }
    /* worth it against the factorization of the full system? (work of the two, the same unit; the dense columns' panels
     * counted as the factorization's flops are, both BLAS-3) */
    /* (20 October 2026, LP/SOCP) the free columns' and the Woodbury columns' solves, and the Schur complement's products */
    const double fw = (wood_on() ? ((double)S->nf + S->ne_wrank) * 2.0 * (double)schol_nnz(sc) + (double)S->nf * (double)S->Ap[S->nf] + (double)S->ne_wrank * S->ne_wrank * (double)m : 0.0)
                      + (double)S->ne_ndt * 2.0 * (double)schol_nnz(sc) + (double)S->ne_ndt * S->ne_ndt * (double)m;   /* (the dense tail columns' solves and U'W) */
    if (S->kkt_defer_md) S->cflops = schol_flops(sc);      /* (the full system's work at least the normal equations' own: see kkt_defer_md) */
    /* (3 November 2026, the SOCP developer, a test: BRISK_CONENERATE=1) the two routes compared by their time in the solver's
     * model (kkt_model), each factorization at its own rate: the normal equations are factored by the supernodal code
     * (6e9 flops a second), the full system by it or by the column code (1.2e9), so equal flops are not equal times.
     * db-joint-soerensen (SOCP), its free variables substituted: the normal equations 4.97e9 flops against the full system's
     * 1.85e9, refused, while the full system goes to the column code; there an iteration took 3.5-4 s against about 2 s on the
     * normal equations (taken when the small cones' eigenbasis made the full system's flops 5.51e9) */
    static int nerate_c = -2; const int nerate = cone_on("BRISK_CONENERATE", &nerate_c);
    const double cf_eq = nerate && !(S->sc || S->sc_lazy || S->meas_sup == 1) && nerate_dense(S) ? S->cflops * (6e9 / 1.2e9) : S->cflops;     /* (the full system's flops at the supernodal rate) */
    /* (8 November 2026, the SOCP developer, a test: BRISK_CONENEALLOW=a, BRISK_CONENEPASS=k) the allowance a (1.5: the full system was
     * often factored column by column, at a fifth of the supernodal rate) and the update's columns' solves at k flops per factor entry
     * and pass (2 as before; a pass is bound by the memory: on a host factoring at about 2e11 flops a second and passing at about 1 ns an
     * entry, a pass costs as much as about 200 flops an entry), the large cones' sparse-form columns (v and a_0) counted with them.
     * Distributionally robust (P06_01, SOCP): the normal equations 5.12e10 flops against the full system's 5.12e10, taken on the tie
     * with two passes more a factorization and left for the full system at the end; the dual form, whose full system is taken, 7.7
     * against 8.9 s */
    { static double nea = -1, nep = -1; if (nea < 0) { const char *e_ = getenv("BRISK_CONENEALLOW"); nea = e_ ? atof(e_) : 1.5; e_ = getenv("BRISK_CONENEPASS"); nep = e_ ? atof(e_) : 0.0; }
      double fw2 = fw;
      if (nep > 0) { int nsc_ = 0; for (int t_ = 0; t_ < S->ne_nsp; t_++) { const int kc = S->ne_spk[t_]; nsc_ += 1 + (S->Ap[S->qs[kc] + 1] > S->Ap[S->qs[kc]]); }
          fw2 = fw + ((double)(S->ne_wrank + S->ne_ndt) * (nep - 2.0) + (double)nsc_ * nep) * (double)schol_nnz(sc); }     /* (the free columns' solves as before: a small factor in the cache passes far faster than a large one; array design, P01_00, 1,480 free columns, 4.5 -> 11.7 s when they were weighted too) */
      if (dbg_on() && nep > 0) socp_printf("  [normal equations: %.3g flops and %.3g for the update's passes against the full system's %.3g (allowance %.2f)]\n", schol_flops(sc) + dflops, fw2, cf_eq, nea);
      /* (the allowance a only where the normal equations carry the large cones' sparse-form update, whose passes and end are their own
       * costs; elsewhere 1.5: array design (P01_00, SOCP), 1,480 free columns and no large cone, the normal equations 1.0-1.5 times the
       * full system's flops and 4.5 against 11.7 s, the full system's regularized free block costing 577 passes over its factor) */
      int nsc2_ = 0; for (int t_ = 0; t_ < S->ne_nsp; t_++) nsc2_++;
      const double allow_ = nep > 0 && nsc2_ == 0 ? 1.5 : nea;
      if (dbg_on() && S->ne_nkc > 0) socp_printf("  [the partial normal equations' factor: nnz(L) %.3g, %.3g flops against the bound %.3g]\n", (double)schol_nnz(sc), schol_flops(sc) + dflops + fw2, allow_ * cf_eq + 1e6);
      if (!getenv("BRISK_CONENE") && !(S->ne_nkc > 0 && getenv("BRISK_CONENENOFCAP")) && schol_flops(sc) + dflops + fw2 > allow_ * cf_eq + 1e6) { schol_free(sc); free(r); free(pool); free(rows); ne_free(S); return 0; } }
    S->ne = sc; S->ne_r = r; S->ne_rows = rows; S->ne_lnz = (long)schol_nnz(sc); S->ne_mm = mm;
    /* (8 November 2026, the SOCP developer; ISSUES 852; the default 5, BRISK_CONENESMALL=0: off) the normal equations' factorization
     * with the full system's update of a node: whole rows of tiles instead of the SDP solver's tiles of 256 columns, and the sources
     * of at most 4 columns added straight into their target instead of a library product and a scatter each, in every column tile
     * (robust dose, SOCP: the root's update from about 28,000 small nodes 0.57 s of a 1.6 s factorization, at 1.8 threads of 4).
     * The factorization an iteration 15-20 % cheaper: P11_06 1.69 -> 1.40 s, distributionally robust 0.87, robust inventory 0.92
     * in time over three cases each; the development mezo set 0.987, no status changed. The grouped products of the full system
     * (bit 2) are left out: they sum many sources before subtracting them, and on the normal equations cost correction steps
     * near the end (P11_06: 171 -> 250 passes over the factor) */
    { extern void schol_set_cone_small(SChol *S, int on); static int ns_ = -1; if (ns_ < 0) { const char *e_ = getenv("BRISK_CONENESMALL"); ns_ = e_ ? atoi(e_) : 5; }
      if (ns_ && sc) schol_set_cone_small(sc, ns_); }     /* (bits: 1 the small sources straight into the target, 2 the grouped products, 4 whole rows of tiles) */
    if (dbg_on()) socp_printf("  [normal equations' factor: %d unknowns, nnz(L) %.3g, %.3g flops (the full system's %.3g)]\n", mm, (double)schol_nnz(sc), schol_flops(sc), S->cflops);
    /* (20 October 2026, LP/SOCP) the free columns as a dense m x nf matrix when they are dense enough, for the Schur
     * complement's product T = A_f' W by dgemm (the FIR filters: 151 columns of 14,100 rows, 1.6e8 products a factorization) */
    free(S->ne_Af); S->ne_Af = NULL;
    if (S->nf > NE_MAXFREE && (double)S->Ap[S->nf] >= 0.25 * (double)m * S->nf && !getenv("BRISK_CONENEAFLOOP")) {
        S->ne_Af = (double *)calloc((size_t)m * S->nf, sizeof(double));
        if (S->ne_Af) for (int f = 0; f < S->nf; f++) for (int p = S->Ap[f]; p < S->Ap[f + 1]; p++) S->ne_Af[(size_t)f * m + S->Ai[p]] += S->Ax[p];
    }
    const double tne2_ = sc_time();
    /* (1 November 2026) a complete pattern: the factor one dense supernode in the given order, checked on the diagonal and the
     * first column (base + j m + i); a large cone's block over all rows then takes its places by that formula (ne_factor) */
    if (S->ne_full) {
        const size_t b0 = schol_offset(sc, 0, 0); int ok_ = b0 != (size_t)-1;
        for (int i = 0; i < m && ok_; i++) ok_ = schol_offset(sc, (size_t)i, (size_t)i) == b0 + (size_t)i * m + i && schol_offset(sc, (size_t)i, 0) == b0 + (size_t)i;
        if (ok_) ok_ = schol_offset(sc, (size_t)m - 1, (size_t)m / 2) == b0 + (size_t)(m / 2) * m + (size_t)m - 1;
        if (ok_) { S->ne_dfull = 1; S->ne_dbase = b0;
            /* (factored over the assembled values, which ne_factor zeroes and writes again before every factorization, as the
             * full system is: one array of the dense factor's size instead of two; A047, SOCP: 3.2 GB each) */
            if (!dense_an_ && !getenv("BRISK_CONENOINPLACE")) schol_set_inplace(sc, 1); }
        else { if (dbg_on()) socp_printf("  [normal equations: the complete pattern's factor not one dense block in the given order: refused]\n");
            ne_free(S); return 0; }        /* (S->ne, ne_r and ne_rows hold sc, r and the rows: ne_free frees them) */
    } else if (!getenv("BRISK_CONENOINPLACE")) {
        /* (2 November 2026, the SOCP developer) any normal equations factored in place: ne_factor zeroes and assembles M again
         * before every factorization, and nothing but the solves reads M afterwards, so the copy of M into the factor's own
         * array is not needed (the same arithmetic). The owner's SOCP benchmark, flow P152 (SOCP, 38,520 rows, a factor of
         * 1.06e8 entries): one array of 850 MB instead of two, and no copy of it at each factorization (the first 9.4 s on
         * this host, the faults of the new array on four threads at once; then 0.27 s each) */
        schol_set_inplace(sc, 1);
    }
#define NE_BYFORMULA(k_) (S->ne_dfull && (k_) >= S->nl && S->q[(k_) - S->nl] > NE_MAXCONE)
    /* (2 November 2026) the table's size counted by the loops that fill it below, not from the running tallies of the
     * pairs (which count blocks the table does not hold and, before this date, missed the tail columns' pairs on the
     * complete pattern: CBLIB 2013_wbNRL, SOCP, a write past the table's end) */
    size_t ntab_ = 0;
    for (int k = 0; k < nb; k++) if (!NE_BYFORMULA(k)) ntab_ += (size_t)r[k] * ((size_t)r[k] + 1) / 2;
    for (int e_ = 0; e_ < S->ne_nex; e_++) ntab_ += (size_t)S->ne_exr[e_] * ((size_t)S->ne_exr[e_] + 1) / 2;
    S->ne_off = (size_t *)malloc(sizeof(size_t) * (ntab_ + 1)); S->ne_doff = (size_t *)malloc(sizeof(size_t) * (size_t)m);
    size_t o = 0; int bad = 0; long gsz = 0;
    for (int k = 0; k < nb; k++) {
        if (NE_BYFORMULA(k)) continue;       /* (no table: base + j m + i) */
        for (int a = 0; a < r[k]; a++) for (int b_ = 0; b_ <= a; b_++) { const size_t f = schol_offset(sc, (size_t)rows[k][a], (size_t)rows[k][b_]); if (f == (size_t)-1) bad = 1; S->ne_off[o++] = f; }
        if (k >= S->nl && S->q[k - S->nl] <= NE_MAXCONE) gsz += (long)r[k] * S->q[k - S->nl];
    }
    for (int e_ = 0; e_ < S->ne_nex; e_++) { const int *rw = S->ne_exrows[e_]; for (int a = 0; a < S->ne_exr[e_]; a++) for (int b_ = 0; b_ <= a; b_++) { const size_t f = schol_offset(sc, (size_t)rw[a], (size_t)rw[b_]); if (f == (size_t)-1) bad = 1; S->ne_off[o++] = f; } }
    for (int i = 0; i < m; i++) { S->ne_doff[i] = schol_offset(sc, (size_t)i, (size_t)i); if (S->ne_doff[i] == (size_t)-1) bad = 1; }
    if (S->ne_nkd > 0) {         /* the kept columns: signs (rows +, columns -), their diagonals' and A_d's places */
        signed char *sg = (signed char *)malloc((size_t)mm + 1); for (int i = 0; i < mm; i++) sg[i] = i < m ? 1 : -1;
        schol_set_signs(sc, sg); free(sg);
        S->ne_kdoff = (size_t *)malloc(sizeof(size_t) * (size_t)S->ne_nkd); S->ne_kaoff = (size_t *)malloc(sizeof(size_t) * ((size_t)S->Ap[n] + 1));
        if (S->ne_keig) {
            long tu_ = 0; for (int a = 0; a < S->ne_nkd; a++) if (S->ne_kun[a] > 0) tu_ += S->ne_kun[a];
            S->ne_kuo = (long *)malloc(sizeof(long) * (size_t)S->ne_nkd); S->ne_kuoff = (size_t *)malloc(sizeof(size_t) * ((size_t)tu_ + 1));
            if (!S->ne_kuo || !S->ne_kuoff) bad = 1;
            long o_ = 0;
            for (int a = 0; !bad && a < S->ne_nkd; a++) { S->ne_kuo[a] = o_; if (S->ne_kun[a] < 0) continue;
                for (int e = 0; e < S->ne_kun[a]; e++) { const size_t f = schol_offset(sc, (size_t)(m + a), (size_t)S->ne_kurow[S->ne_kus[a] + e]); if (f == (size_t)-1) bad = 1; S->ne_kuoff[o_++] = f; } }
        }
        for (int a = 0; a < S->ne_nkd; a++) {
            const int j = S->ne_kdcol[a]; S->ne_kdoff[a] = schol_offset(sc, (size_t)(m + a), (size_t)(m + a)); if (S->ne_kdoff[a] == (size_t)-1) bad = 1;
            if (S->ne_keig && S->ne_kun[a] >= 0) continue;
            for (int p = S->Ap[j]; p < S->Ap[j + 1]; p++) { S->ne_kaoff[p] = schol_offset(sc, (size_t)(m + a), (size_t)S->Ai[p]); if (S->ne_kaoff[p] == (size_t)-1) bad = 1; }
        }
        if (S->ne_kdcone && !S->ne_keig) {      /* (BRISK_CONEPARTNE) each kept cone's lower triangle off the diagonal, row by row */
            long nc_ = 0; for (int a = 0; a < S->ne_nkd; a++) if (S->ne_kdcone[a] >= 0) nc_ += S->ne_kdpos[a];
            S->ne_kcoff = (size_t *)malloc(sizeof(size_t) * ((size_t)nc_ + 1)); long o_ = 0;
            if (!S->ne_kcoff) bad = 1;
            for (int a = 0; S->ne_kcoff && a < S->ne_nkd; a++) { if (S->ne_kdcone[a] < 0) continue;
                for (int b_ = a - S->ne_kdpos[a]; b_ < a; b_++) { S->ne_kcoff[o_] = schol_offset(sc, (size_t)(m + a), (size_t)(m + b_)); if (S->ne_kcoff[o_] == (size_t)-1) bad = 1; o_++; } }
        }
        double *sw = (double *)realloc(S->swork2, sizeof(double) * ((size_t)mm + 8)); if (sw) S->swork2 = sw; else bad = 1;
        /* (13 November 2026, BRISK_CONEPARTNE) with kept cones a pivot of the wrong sign in a node of kept unknowns is replaced (1e-7, as
         * in the full system's supernodal factorization), not a failure: the tiny-pivot rule covers only the nodes of rows. GMRES refines
         * against the exact operator */
        if (S->ne_nkc > 0) schol_set_dynpiv(sc, 1e-7);
    }
    S->ne_nu0 = 0; S->ne_ntf = 0; S->ne_wcap = 0;
    /* (8 November 2026, the SOCP developer, a test: BRISK_CONENESPLIT=1) a large cone formed into the normal equations as a dense
     * block: eta^2 H^-1 = 2 u u' - J (u = J w) has the eigenvalues lmax = (w0 + n)^2, lmin = 1 / lmax (n = |w_tail|) on g = A (e0 + t)
     * / sqrt2 and g' = A (e0 - t) / sqrt2 (t the unit tail of u) and 1 elsewhere, so A H^-1 A' = eta^-2 [2 a0 a0' - C_k - g g' - (1 - lmin)
     * g' g''] + eta^-2 lmax g g': the first part positive semidefinite and of the size of A A', the second a positive rank-one term,
     * applied by the Woodbury update. Formed as eta^-2 (2 v v' - C_k), the term lmax g g' dominates the factored matrix near the end,
     * the directions across it are lost (heterogeneous cones A05_00, SOCP: the homogeneous model's fixed column solved to 6e-6 in the
     * cone's rows at iteration 7, three iterations lost; the full system 8 iterations, as MOSEK) */
    { static int nspl_on = -1; if (nspl_on < 0) { const char *e_ = getenv("BRISK_CONENESPLIT"); nspl_on = e_ ? atoi(e_) : 0; }
      S->ne_nspl = 0;
      if (nspl_on && S->ne_r && !(S->nl == 0 && S->nf == 0 && S->nq == 1)) {     /* (one cone alone: the route of its constant part, below) */
          int c_ = 0; for (int k = S->nl; k < S->nl + S->nq; k++) if (S->q[k - S->nl] > NE_MAXCONE && S->ne_r[k] > 0) c_++;
          if (c_ > 0 && S->ne_wrank + c_ <= NE_MAXWOODRANK) {
              S->ne_splk = (int *)malloc(sizeof(int) * (size_t)c_); S->ne_splg = (double *)calloc((size_t)mm * c_, sizeof(double)); S->ne_spld = (double *)malloc(sizeof(double) * (size_t)c_);
              if (S->ne_splk && S->ne_splg && S->ne_spld) { for (int k = S->nl; k < S->nl + S->nq; k++) if (S->q[k - S->nl] > NE_MAXCONE && S->ne_r[k] > 0) S->ne_splk[S->ne_nspl++] = k;
                  if (dbg_on()) socp_printf("  [normal equations: %d large cones' largest eigenvalue by the Woodbury update]\n", S->ne_nspl); }
          } } }
    if (S->ne_nsp > 0 || S->ne_nwl > 0 || S->ne_nspl > 0) {         /* the Woodbury columns: v and a_0 of each sparse cone (a_0 only when it has rows), the dense linear columns */
        int nu = S->ne_wrank + S->ne_ndt + S->ne_nspl; for (int t_ = 0; t_ < S->ne_nsp; t_++) { const int kc = S->ne_spk[t_]; nu += 1 + (S->Ap[S->qs[kc] + 1] > S->Ap[S->qs[kc]]); }
        S->ne_nu = S->ne_nu0 = nu; const int nuc = nu; S->ne_wcap = nuc;     /* (the tiny pivots' columns get room when they occur, ne_factor) */
        S->ne_wU = (double *)malloc(sizeof(double) * (size_t)mm * nuc); S->ne_wW = (double *)malloc(sizeof(double) * (size_t)mm * nuc);
        S->ne_wC = (double *)malloc(sizeof(double) * (size_t)nuc * nuc); S->ne_wpiv = (int *)malloc(sizeof(int) * (size_t)nuc);
        if (dbg_on()) socp_printf("  [normal equations: %d large cones by their sparse form, %d tail columns (%d dense ones in the update), %d dense blocks, a Woodbury update of rank %d]\n", S->ne_nsp, S->ne_nex, S->ne_ndt, S->ne_nwl, nu);
    }
    /* (2 November 2026, the SOCP developer) one second-order cone over all the rows and nothing else in the normal equations:
     * its block is eta^-2 (K - a_0 a_0' + 2 v v'), K = A_t A_t' (the tail columns' Gram matrix: constant), v = A_k u changing with
     * the iterate. K (with the regularization) is factored once and each factorization becomes a rank-two Woodbury update of
     * that factor, the solves scaled by eta^2 (ne_factor_onec): no dense Cholesky an iteration. CBLIB 2013_firL2a (SOCP, a cone
     * of 10,002 entries over 10,001 rows): 3.3e11 flops a factorization. BRISK_CONEONEC=0: off */
    S->ne_onec = -1; S->ne_onec_ok = 0; S->ne_scale = 1.0;
    { static int oc = -1; if (oc < 0) { const char *e_ = getenv("BRISK_CONEONEC"); oc = !(e_ && atoi(e_) == 0); }
      if (oc && S->ne_dfull && S->nl == 0 && S->nf == 0 && S->nq == 1 && S->q[0] > NE_MAXCONE && S->ne_r && S->ne_r[0] == m
          && S->ne_nsp == 0 && S->ne_nwl == 0 && S->ne_nkd == 0 && S->ne_ndt == 0 && S->ne_md == 0 && S->ne_nfix == 0 && S->ne_nex == 0 && !S->ne_elim && S->ne_nu0 == 0) {
          S->ne_wU = (double *)malloc(sizeof(double) * (size_t)m * 2); S->ne_wW = (double *)malloc(sizeof(double) * (size_t)m * 2);
          S->ne_wC = (double *)malloc(sizeof(double) * 4); S->ne_wpiv = (int *)malloc(sizeof(int) * 2);
          if (S->ne_wU && S->ne_wW && S->ne_wC && S->ne_wpiv) { S->ne_onec = 0; S->ne_wcap = 2;
              if (dbg_on()) socp_printf("  [normal equations: one cone over all %d rows: its constant part factored once, a rank-two update an iteration]\n", m); }
      } }
    if (S->ne_md > 0) {          /* the dense block's lower triangle, by its columns */
        const int md = S->ne_md; S->ne_doffd = (size_t *)malloc(sizeof(size_t) * ((size_t)md * (md + 1) / 2));
        size_t e = 0;
        for (int b_ = 0; b_ < md; b_++) for (int a = b_; a < md; a++) {
            const int i = S->ne_drow[a], j = S->ne_drow[b_];
            const size_t f = schol_offset(sc, (size_t)(i > j ? i : j), (size_t)(i > j ? j : i)); if (f == (size_t)-1) bad = 1; S->ne_doffd[e++] = f;
        }
    }
#undef NE_BYFORMULA
    if (bad) { ne_free(S); return 0; }
    const double tne3_ = sc_time();
    /* the cones' columns of A as dense r x d matrices (by rows) */
    S->ne_G = (double *)calloc((size_t)gsz + 1, sizeof(double));
    int *posr = (int *)malloc(sizeof(int) * (size_t)m);
    long g = 0;
    long csz = 0;
    for (int k = S->nl; k < nb; k++) if (S->q[k - S->nl] > NE_MAXCONE && r[k] > 0) csz += (long)r[k] * (r[k] + 1) / 2;
    if (csz) S->ne_C = (double *)calloc((size_t)csz, sizeof(double));
    /* (1 November 2026) with the pattern complete the C_k are packed by columns (column b: rows b .. r-1), so that the
     * iterations add each column into the dense factor's column contiguously; packed by rows, each addition went to another
     * column of the factor (a stride of m doubles: A047, SOCP, 2e8 such additions per cone and factorization). The same
     * entries and sums in either packing */
    const int ccol_ = S->ne_dfull;
#define NE_CCOL(b_, a_, rr_) ((size_t)(b_) * (size_t)(rr_) - (size_t)(b_) * ((size_t)(b_) - 1) / 2 + (size_t)((a_) - (b_)))
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
                if (ccol_) { for (int b_ = 0; b_ < rr; b_++) memcpy(C + NE_CCOL(b_, b_, rr), Cf + (size_t)b_ * rr + b_, sizeof(double) * (size_t)(rr - b_)); }
                else for (int a = 0; a < rr; a++) for (int b_ = 0; b_ <= a; b_++) C[(long)a * (a + 1) / 2 + b_] = Cf[(size_t)b_ * rr + a];
                free(Cf); free(Wp);
                cpos += (long)r[k] * (r[k] + 1) / 2;
                continue;
            }
            free(Cf); free(Wp);
            /* (1 November 2026, the SOCP developer; the owner's SOCP benchmark, wide cones) row by row when the products are many:
             * row a of C_k is the sum over its columns t (increasing) of J_tt a_t(a) times column t's entries of rows b <= a,
             * gathered in a dense row of the block (in cache) and copied into the packed triangle; each entry gets the same
             * products in the same order as in the loop below, so the bits are the same, and the rows are independent (on the
             * threads, the same bits at any thread count). The loop scattered each product into the triangle: A052 (SOCP, 2.4e7
             * products into 4,891 rows) 1.28 s, about 50 ns a product. BRISK_CONECKROW=0: the loop */
            static int ckrow = -1; if (ckrow < 0) { const char *e_ = getenv("BRISK_CONECKROW"); ckrow = !(e_ && atoi(e_) == 0); }
            if (ckrow && lsum >= 1e6 && !elim) {      /* (no eliminated rows: posr holds the block's rows only) */
                const int rr = r[k]; const long nz = S->Ap[c0 + d] - S->Ap[c0];
                int *rp_ = (int *)calloc((size_t)rr + 1, sizeof(int)), *rt_ = (int *)malloc(sizeof(int) * (size_t)(nz + 1)), *rq_ = (int *)malloc(sizeof(int) * (size_t)(nz + 1));
                if (rp_ && rt_ && rq_) {
                    for (long p = S->Ap[c0]; p < S->Ap[c0 + d]; p++) rp_[posr[S->Ai[p]] + 1]++;
                    for (int a = 0; a < rr; a++) rp_[a + 1] += rp_[a];
                    { int *f = (int *)malloc(sizeof(int) * (size_t)(rr + 1)); memcpy(f, rp_, sizeof(int) * (size_t)rr);
                      for (int t = 0; t < d; t++) for (int p = S->Ap[c0 + t]; p < S->Ap[c0 + t + 1]; p++) { const int q_ = f[posr[S->Ai[p]]]++; rt_[q_] = t; rq_[q_] = p; }
                      free(f); }
                    const int nt_ = (g_cone_nt > 1 && !omp_in_parallel() && lsum >= 1e7) ? g_cone_nt : 1;
                    #pragma omp parallel num_threads(nt_)
                    {   double *acc = (double *)calloc((size_t)rr + 1, sizeof(double));
                        #pragma omp for schedule(dynamic, 16)
                        for (int a = 0; a < rr; a++) {
                            for (int q_ = rp_[a]; q_ < rp_[a + 1]; q_++) {
                                const int t = rt_[q_], p = rq_[q_]; const double xa = (t == 0 ? 1.0 : -1.0) * S->Ax[p];
                                if (ccol_) { for (int p2 = S->Ap[c0 + t]; p2 < S->Ap[c0 + t + 1]; p2++) { const int rb = posr[S->Ai[p2]]; if (rb >= a) acc[rb] += xa * S->Ax[p2]; } }
                                else for (int p2 = S->Ap[c0 + t]; p2 < S->Ap[c0 + t + 1]; p2++) { const int rb = posr[S->Ai[p2]]; if (rb <= a) acc[rb] += xa * S->Ax[p2]; }
                            }
                            if (ccol_) {         /* (column a: rows a .. r-1; the products commute, the order over t is the loop's) */
                                double *Ca = C + NE_CCOL(a, a, rr);
                                for (int b_ = a; b_ < rr; b_++) { Ca[b_ - a] = acc[b_]; acc[b_] = 0.0; }
                            } else {
                            double *Ca = C + (long)a * (a + 1) / 2;
                            for (int b_ = 0; b_ <= a; b_++) { Ca[b_] = acc[b_]; acc[b_] = 0.0; }
                            }
                        }
                        free(acc);
                    }
                    free(rp_); free(rt_); free(rq_);
                    cpos += (long)r[k] * (r[k] + 1) / 2;
                    continue;
                }
                free(rp_); free(rt_); free(rq_);
            }
            for (int t = 0; t < d; t++) {
                const double jt = t == 0 ? 1.0 : -1.0;
                for (int p = S->Ap[c0 + t]; p < S->Ap[c0 + t + 1]; p++) {
                    const long ra = posr[S->Ai[p]]; const double xa = jt * S->Ax[p];
                    for (int p2 = S->Ap[c0 + t]; p2 < S->Ap[c0 + t + 1]; p2++) { const int rb = posr[S->Ai[p2]]; if (rb <= ra) C[ccol_ ? NE_CCOL(rb, ra, r[k]) : (size_t)(ra * (ra + 1) / 2 + rb)] += xa * S->Ax[p2]; }
                }
            }
            cpos += (long)r[k] * (r[k] + 1) / 2;
            continue;
        }
        for (int t = 0; t < d; t++) for (int p = S->Ap[c0 + t]; p < S->Ap[c0 + t + 1]; p++) if (!NE_ELIM(S->Ai[p])) S->ne_G[g + (long)posr[S->Ai[p]] * d + t] += S->Ax[p];
        g += (long)r[k] * d;
    }
    free(posr);
#undef NE_CCOL
    if (dbg_on()) socp_printf("  [normal equations: the offsets %.3fs, the large cones' products %.3fs]\n", tne3_ - tne2_, sc_time() - tne3_);
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
/* (13 November 2026, the SOCP developer, BRISK_CONEPARTNE=t) the partial normal equations chosen at mu: the small cones whose own scaling
 * spread (w0 + |w1|)^4 exceeds (1/mu)^t keep their unknowns, and the normal equations are set up again with them (ne_setup_fix). Returns
 * the cones kept (0: the normal equations refused, S->ne NULL: the full system). The selection follows mu: as mu falls, more cones lose
 * strict complementarity (db-joint-soerensen, SOCP: chosen at mu 2.2e-9, the factor broke at mu 6.8e-11) */
/* (BRISK_CONEPARTNEEIG) the eigenbasis of a small cone's block H_k = eta^2 [ (1/l) p p' + l q q' + (I - p p' - q q') ], l = (w0 + |w1|)^2,
 * p = (1, -v)/sqrt2, q = (1, v)/sqrt2 (v the unit direction of w1): Q's column t (in qcol + t NE_MAXCONE) and its eigenvalue lam[t]; the
 * columns from 2 on (0, the tail's complement of v) from the Householder reflection that maps e1 to v */
static void partne_eig(const Socp *S, int kc, double *qcol, double *lam) {
    const int d = S->q[kc]; const double *w = S->wbar + S->qs[kc], e2 = S->eta[kc] * S->eta[kc];
    double nw = 0; for (int i = 1; i < d; i++) nw += w[i] * w[i]; nw = sqrt(nw);
    const double l = (w[0] + nw) * (w[0] + nw), r2 = sqrt(0.5);
    double v[NE_MAXCONE]; for (int i = 1; i < d; i++) v[i] = nw > 0 ? w[i] / nw : (i == 1 ? 1.0 : 0.0);
    for (int t = 0; t < d; t++) for (int c = 0; c < d; c++) qcol[(size_t)t * NE_MAXCONE + c] = 0.0;
    qcol[0] = r2; qcol[NE_MAXCONE] = r2;
    for (int i = 1; i < d; i++) { qcol[i] = -r2 * v[i]; qcol[NE_MAXCONE + i] = r2 * v[i]; }
    lam[0] = e2 / l; lam[1] = d > 1 ? e2 * l : 0.0;
    if (d > 2) {     /* u = v - e1 over the tail; H = I - 2 u u' / u'u; its columns 2..d-1 (tail indices) are orthonormal and orthogonal to v */
        double u[NE_MAXCONE], uu = 0; for (int i = 1; i < d; i++) { u[i] = v[i] - (i == 1 ? 1.0 : 0.0); uu += u[i] * u[i]; }
        for (int t = 2; t < d; t++) { lam[t] = e2;
            for (int i = 1; i < d; i++) qcol[(size_t)t * NE_MAXCONE + i] = (i == t ? 1.0 : 0.0) - (uu > 1e-300 ? 2.0 * u[i] * u[t] / uu : 0.0); }
    }
}
const int *schol_iperm(const SChol *S);
static int partne_build(Socp *S, double mu, double t) {
    if (!S->ne_kcsel) S->ne_kcsel = (unsigned char *)calloc((size_t)S->nq + 1, 1);
    if (!S->ne_kcsel) return 0;
    const double lim_ = t * log(1.0 / mu); int nsel = 0;
    for (int kc = 0; kc < S->nq; kc++) {
        const int d = S->q[kc]; S->ne_kcsel[kc] = 0; if (d > NE_MAXCONE) continue;
        const double *w = S->wbar + S->qs[kc]; double nw = 0; for (int i = 1; i < d; i++) nw += w[i] * w[i];
        if (4.0 * log(w[0] + sqrt(nw)) > lim_) { S->ne_kcsel[kc] = 1; nsel++; }
    }
    if (nsel == 0) return 0;
    ne_free(S); S->ne_ok = 0; ne_setup(S); S->ne_part = 1; S->ne_part_mu = mu;
    if (S->ne && S->ne_keig && dbg_on()) {      /* (debug: where the order puts the kept unknowns against their rows) */
        const int *ip = schol_iperm(S->ne); long bef = 0, aft = 0, mix = 0;
        for (int a = 0; a < S->ne_nkd; a++) { if (S->ne_kun[a] < 0) continue; int nb_ = 0, na_ = 0;
            for (int e = 0; e < S->ne_kun[a]; e++) { if (ip[S->ne_kurow[S->ne_kus[a] + e]] < ip[S->m + a]) nb_++; else na_++; }
            if (nb_ == 0) bef++; else if (na_ == 0) aft++; else mix++; }
        socp_printf("      [the kept unknowns in the order: %ld before all their rows, %ld after all, %ld between]\n", bef, aft, mix);
    }
    return S->ne ? S->ne_nkc : 0;
}
/* assembles and factors; 0: done, 1: failed */
/* (2 November 2026, the SOCP developer) the normal equations of one second-order cone over all rows (ne_setup's ne_onec):
 * M = eta^-2 (K + U D' U') + delta I with K = A_t A_t' = a_0 a_0' - C_k, U = [v a_0], D' = diag(2, -1). The factored matrix is
 * eta^-2 (K + delta eta_0^2 I) (eta_0 that of the factorization: the regularization's share at that iterate; the difference to
 * delta I is left to the refinement against the full operator, as every regularization), factored again only when eta^2
 * has moved by more than 1e4 times since; the update and the scale eta^2 are applied in ne_wood_apply. Returns 0, 1 (failure
 * of the update) or -1 (the general route instead: a factorization that met tiny pivots, double-double). */
static int ne_factor_onec(Socp *S) {
    const int m = S->m, s = S->qs[0], d = S->q[0];
    const double eta2 = S->eta[0] * S->eta[0];
    if (S->use_dd || !S->ne_C) { S->ne_onec_ok = 0; return -1; }
    if (!S->ne_onec_ok || fabs(log(eta2 / S->ne_onec_eta2)) > log(1e4)) {   /* (K does not depend on eta: only the regularization's share does, delta (eta_0^2 / eta^2 - 1) against delta = 1e-9, left to the refinement up to a ratio of 1e4) */
        schol_zero(S->ne);
        double *pm = schol_values(S->ne), *a0 = (double *)calloc((size_t)m, sizeof(double));
        if (!a0) { S->ne_onec_ok = 0; return -1; }
        for (int p = S->Ap[s]; p < S->Ap[s + 1]; p++) a0[S->Ai[p]] += S->Ax[p];
        double *pd = pm + S->ne_dbase; const double *Cb0 = S->ne_C; const int *rw = S->ne_rows[0]; const int rk = m;
        #pragma omp parallel for schedule(dynamic, 64) num_threads(g_cone_nt) if (g_cone_nt > 1 && !omp_in_parallel())
        for (int b_ = 0; b_ < rk; b_++) {
            const double *cb = Cb0 + (size_t)b_ * rk - (size_t)b_ * ((size_t)b_ - 1) / 2; double *col = pd + (size_t)rw[b_] * m; const double ab = a0[rw[b_]];
            for (int a = b_; a < rk; a++) col[rw[a]] += a0[rw[a]] * ab - cb[a - b_];
        }
        free(a0);
        for (int i = 0; i < m; i++) pm[S->ne_doff[i]] += S->sreg0 * eta2;
        schol_set_tinypiv(S->ne, 1e-14);
        const int bad = schol_factor(S->ne, 0.0);
        if (bad || schol_ntiny(S->ne) > 0) {
            if (dbg_on()) socp_printf("      [one cone's constant part: the factorization %s: the general route]\n", bad ? "failed" : "met tiny pivots");
            S->ne_onec = -1; S->ne_onec_ok = 0; S->ne_nu = 0; return -1;
        }
        S->ne_onec_ok = 1; S->ne_onec_eta2 = eta2;
        if (dbg_on()) socp_printf("      [one cone's constant part factored (eta^2 %.3e)]\n", eta2);
    }
    double *U = S->ne_wU, *W = S->ne_wW, *C = S->ne_wC; const double *w = S->wbar + s;
    for (int i = 0; i < 2 * m; i++) U[i] = 0.0;
    for (int t = 0; t < d; t++) { const double ut = t == 0 ? w[0] : -w[t]; for (int p = S->Ap[s + t]; p < S->Ap[s + t + 1]; p++) U[S->Ai[p]] += ut * S->Ax[p]; }
    for (int p = S->Ap[s]; p < S->Ap[s + 1]; p++) U[m + S->Ai[p]] += S->Ax[p];
    const double dinv[2] = { 0.5, -1.0 };
    { static int wmany = -1; if (wmany < 0) { const char *e_ = getenv("BRISK_CONEWOODMANY"); wmany = e_ ? atoi(e_) : 0; }     /* (a test: the two columns in one solve, as in ne_factor) */
      if (wmany) { extern void schol_solve_many(const SChol *S, double *Bm, int nrhs, double *work); memcpy(W, U, sizeof(double) * 2 * (size_t)m); schol_solve_many(S->ne, W, 2, S->swork2); }
      else
      for (int f = 0; f < 2; f++) { memcpy(W + (size_t)f * m, U + (size_t)f * m, sizeof(double) * (size_t)m); schol_set_solve_seq(cone_seqsolve()); schol_set_solve_nt(g_cone_nt); schol_solve(S->ne, W + (size_t)f * m, S->swork2); schol_set_solve_seq(0); } }
    for (int f = 0; f < 2; f++) for (int g = 0; g < 2; g++) { double v = 0; const double *uf = U + (size_t)f * m, *wg = W + (size_t)g * m; for (int i = 0; i < m; i++) v += uf[i] * wg[i]; C[f * 2 + g] = v + (f == g ? dinv[f] : 0.0); }
    for (int f = 0; f < 2; f++) {
        int pv = f; for (int g = f + 1; g < 2; g++) if (fabs(C[g * 2 + f]) > fabs(C[pv * 2 + f])) pv = g;
        S->ne_wpiv[f] = pv; if (pv != f) for (int g = 0; g < 2; g++) { const double t = C[f * 2 + g]; C[f * 2 + g] = C[pv * 2 + g]; C[pv * 2 + g] = t; }
        if (C[f * 2 + f] == 0.0) { S->ne_nu = 0; return 1; }
        for (int g = f + 1; g < 2; g++) { const double l = C[g * 2 + f] / C[f * 2 + f]; C[g * 2 + f] = l; for (int h = f + 1; h < 2; h++) C[g * 2 + h] -= l * C[f * 2 + h]; }
    }
    S->ne_nu = 2; S->ne_scale = eta2;
    /* the update checked on one vector against the operator A H^-1 A' + delta I: a rank-two update that nearly cancels the
     * constant part (the cone's iterate near its boundary at the end: firL2a's last iteration, mu 7e9 after an update that
     * missed by orders) is not used; the route is left for the general one from here on (its factorization in place
     * overwrites the constant factor). Two products with A and one solve. */
    { const int n = S->n; double *x = (double *)malloc(sizeof(double) * (size_t)m), *ax = (double *)malloc(sizeof(double) * (size_t)n), *hx = (double *)malloc(sizeof(double) * (size_t)n), *t = (double *)malloc(sizeof(double) * (size_t)m + 8);
      double e_ = 1e300, xn = 0;
      if (x && ax && hx && t) {
          for (int i = 0; i < m; i++) x[i] = 1.0 + 0.5 * sin(1.0 + i);
          for (int j = 0; j < n; j++) ax[j] = 0.0;
          At_mul(S, 1.0, x, ax); Hinv_apply(S, ax, hx);
          for (int i = 0; i < m; i++) t[i] = 0.0;
          A_mul(S, 1.0, hx, t);
          for (int i = 0; i < m; i++) t[i] += S->sreg0 * x[i];
          schol_set_solve_seq(cone_seqsolve()); schol_set_solve_nt(g_cone_nt); schol_solve(S->ne, t, S->swork2); schol_set_solve_seq(0); ne_wood_apply(S, t);
          e_ = 0; for (int i = 0; i < m; i++) { e_ = fmax(e_, fabs(t[i] - x[i])); xn = fmax(xn, fabs(x[i])); }
          e_ /= fmax(xn, 1e-300);
      }
      free(x); free(ax); free(hx); free(t);
      if (dbg_on()) socp_printf("      [one cone's update checked: |M^-1 M x - x| / |x| %.1e]\n", e_);
      static double octol = -1; if (octol < 0) { const char *e2_ = getenv("BRISK_CONEONECTOL"); octol = e2_ ? atof(e2_) : 1e-2; }   /* (the preconditioner's miss allowed: the refinement against the operator removes it) */
      if (!(e_ <= octol)) { S->ne_onec = -1; S->ne_onec_ok = 0; S->ne_nu = 0; S->ne_scale = 1.0; return -1; } }
    return 0;
}
static int ne_factor(Socp *S) {
    const int m = S->m, nb = S->nl + S->nq;
    if (S->ne_onec >= 0) { const int r_ = ne_factor_onec(S); if (r_ >= 0) return r_; }
    S->ne_scale = 1.0;
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
            int spi_ = -1; for (int t_ = 0; t_ < S->ne_nspl; t_++) if (S->ne_splk[t_] == k) spi_ = t_;
            if (spi_ >= 0) {         /* (BRISK_CONENESPLIT, ne_setup) eta^-2 [2 a0 a0' - C_k - g g' - (1 - lmin) g' g''] here, eta^-2 lmax g g' by the update */
                double nt = 0; for (int t = 1; t < d; t++) nt += w[t] * w[t]; nt = sqrt(nt);
                const double lmax = (w[0] + nt) * (w[0] + nt), lmin = 1.0 / lmax, r2 = sqrt(0.5);
                double *a0b = (double *)calloc((size_t)rk + 1, sizeof(double)), *gb = (double *)malloc(sizeof(double) * ((size_t)rk + 1)), *hb = (double *)malloc(sizeof(double) * ((size_t)rk + 1));
                if (!a0b || !gb || !hb) { free(a0b); free(gb); free(hb); free(vb); free(posb); return 1; }
                for (int p = S->Ap[s]; p < S->Ap[s + 1]; p++) a0b[posb[S->Ai[p]]] += S->Ax[p];
                for (int a = 0; a < rk; a++) { const double tb = nt > 0 ? (vb[a] - w[0] * a0b[a]) / nt : 0.0; gb[a] = r2 * (a0b[a] + tb); hb[a] = r2 * (a0b[a] - tb); }
                double *G_ = S->ne_splg + (size_t)spi_ * S->ne_mm; for (size_t i = 0; i < (size_t)S->ne_mm; i++) G_[i] = 0.0;
                for (int a = 0; a < rk; a++) G_[S->ne_rows[k][a]] = gb[a];
                /* (2: one unit of g g' kept in the factored part, which is then eta^-2 A (I - (1 - lmin) u' u'') A', positive definite as A A',
                 * and eta^-2 (lmax - 1) g g' by the update; with all of it removed the factored part was near singular where the cone spans
                 * the rows: A05_00 9 iterations but 113 passes over the factor, A02_00 14 -> 21) */
                static int spl2 = -1; if (spl2 < 0) { const char *e_ = getenv("BRISK_CONENESPLIT"); spl2 = e_ ? atoi(e_) : 0; }
                const double kg_ = spl2 == 2 ? 0.0 : 1.0;
                S->ne_spld[spi_] = ie2 * (lmax - (spl2 == 2 ? 1.0 : 0.0));
                const size_t ntri = (size_t)rk * (rk + 1) / 2; const double om = 1.0 - lmin;
                if (S->ne_dfull) {
                    double *pd = pm + S->ne_dbase; const double *Cb0 = Cb; const int *rw = S->ne_rows[k];
                    #pragma omp parallel for schedule(dynamic, 64) num_threads(g_cone_nt) if (g_cone_nt > 1 && ntri >= 1000000 && !omp_in_parallel())
                    for (int b_ = 0; b_ < rk; b_++) {
                        const double *cb = Cb0 + (size_t)b_ * rk - (size_t)b_ * ((size_t)b_ - 1) / 2; double *col = pd + (size_t)rw[b_] * m;
                        const double ab = a0b[b_], gb_ = gb[b_], hb_ = hb[b_];
                        for (int a = b_; a < rk; a++) col[rw[a]] += ie2 * ((2.0 * a0b[a] * ab - cb[a - b_]) - kg_ * gb[a] * gb_ - om * hb[a] * hb_);
                    }
                    Cb += ntri;
                } else {
                    const size_t *off0 = off; const double *Cb0 = Cb;
                    #pragma omp parallel for schedule(dynamic, 64) num_threads(g_cone_nt) if (g_cone_nt > 1 && ntri >= 1000000 && !omp_in_parallel())
                    for (int a = 0; a < rk; a++) {
                        const size_t o = (size_t)a * (a + 1) / 2; const size_t *of = off0 + o; const double *cb = Cb0 + o;
                        for (int b_ = 0; b_ <= a; b_++) pm[of[b_]] += ie2 * ((2.0 * a0b[a] * a0b[b_] - cb[b_]) - kg_ * gb[a] * gb[b_] - om * hb[a] * hb[b_]);
                    }
                    off += ntri; Cb += ntri;
                }
                free(a0b); free(gb); free(hb);
                continue;
            }
            /* (28 October 2026, the SOCP developer) the rows on the threads when the term is large (row a's entries start at
             * a(a+1)/2 in off and in C_k; each entry gets its one addition, so the bits are those of the loop): a cone of 3,073
             * entries over 3,075 rows, 4.7 million entries an iteration (H_A_scaling_3072, SOCP) */
            const size_t ntri = (size_t)rk * (rk + 1) / 2;
            if (S->ne_dfull) {          /* (1 November 2026) the pattern complete: places by formula (the block's rows sorted) */
                double *pd = pm + S->ne_dbase; const double *Cb0 = Cb; const int *rw = S->ne_rows[k];
                /* (C_k packed by columns: column b's entries added to the factor's column rw[b], contiguous; each entry the same
                 * expression as below, one addition per cone) */
                #pragma omp parallel for schedule(dynamic, 64) num_threads(g_cone_nt) if (g_cone_nt > 1 && ntri >= 1000000 && !omp_in_parallel())
                for (int b_ = 0; b_ < rk; b_++) {
                    const double *cb = Cb0 + (size_t)b_ * rk - (size_t)b_ * ((size_t)b_ - 1) / 2; double *col = pd + (size_t)rw[b_] * m; const double vbb = vb[b_];
                    for (int a = b_; a < rk; a++) col[rw[a]] += ie2 * ((2.0 * vb[a]) * vbb - cb[a - b_]);
                }
                Cb += ntri;
                continue;
            }
            if (ntri >= 1000000 && g_cone_nt > 1 && !omp_in_parallel()) {
                const size_t *off0 = off; const double *Cb0 = Cb;
                #pragma omp parallel for schedule(dynamic, 64) num_threads(g_cone_nt)
                for (int a = 0; a < rk; a++) {
                    const size_t o = (size_t)a * (a + 1) / 2; const size_t *of = off0 + o; const double *cb = Cb0 + o; const double va = 2.0 * vb[a];
                    for (int b_ = 0; b_ <= a; b_++) pm[of[b_]] += ie2 * (va * vb[b_] - cb[b_]);
                }
                off += ntri; Cb += ntri;
            } else
            for (int a = 0; a < rk; a++) { const double va = 2.0 * vb[a]; for (int b_ = 0; b_ <= a; b_++) pm[*off++] += ie2 * (va * vb[b_] - *Cb++); }
            continue;
        }
        /* (13 November 2026, the SOCP developer, a test: BRISK_CONENEEIGHI=1) H_k^-1 formed in its eigenbasis, eta^-2 [ l p p' + (1/l) q q'
         * + (I - p p' - q q') ] (l = (w0 + |w1|)^2, p = (1, -v)/sqrt2, q = (1, v)/sqrt2), not as eta^-2 (2 u u' - J): there the small eigenvalue
         * 1/l comes out of a cancellation between terms of size l, with a relative error of about eps l^2 (near the boundary at mu 1e-11,
         * l^2 about 1e10 on db-joint-soerensen, SOCP) */
        static int neeig = -1; if (neeig < 0) { const char *e_ = getenv("BRISK_CONENEEIGHI"); neeig = e_ ? atoi(e_) : 0; }
        if (neeig) {
            double nw = 0; for (int i = 1; i < d; i++) nw += w[i] * w[i]; nw = sqrt(nw);
            const double l = (w[0] + nw) * (w[0] + nw), r2 = sqrt(0.5); double pv[NE_MAXCONE], qv[NE_MAXCONE];
            pv[0] = r2; qv[0] = r2;
            for (int i = 1; i < d; i++) { const double vi = nw > 0 ? w[i] / nw : (i == 1 ? 1.0 : 0.0); pv[i] = -r2 * vi; qv[i] = r2 * vi; }
            for (int i = 0; i < d; i++) for (int j = 0; j < d; j++)
                Hi[i * d + j] = ie2 * ((i == j ? 1.0 : 0.0) + (l - 1.0) * pv[i] * pv[j] + (1.0 / l - 1.0) * qv[i] * qv[j]);
        } else
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
      double dl_ = dlf && S->ne_dl > 0 ? fmin(S->sreg0, S->ne_dl) : S->sreg0;
      /* (13 November 2026, the SOCP developer, a test: BRISK_CONEPARTNEDR=r) with kept cones the rows regularized in the factor at r at
       * least: the kept unknowns' pivots, -(lambda + delta) - a' M^-1 a, are formed through the inverse of the rows' block, whose smallest
       * eigenvalue is the rows' regularization; a larger one bounds that inverse (GMRES refines against the exact operator) */
      { static double pdr = -1; if (pdr < 0) { const char *e_ = getenv("BRISK_CONEPARTNEDR"); pdr = e_ ? atof(e_) : 0.0; }
        if (pdr > 0 && S->ne_nkc > 0) dl_ = fmax(dl_, pdr); }
      for (int i = 0; i < m; i++) pm[S->ne_doff[i]] += dl_;
      long oc_ = 0;
      for (int a = 0; a < S->ne_nkd; a++) {          /* (23 October 2026) a kept dense column: A_d's entries and -(h_j + delta) */
          const int j = S->ne_kdcol[a];
          if (S->ne_keig && S->ne_kun[a] >= 0) {     /* (BRISK_CONEPARTNEEIG) the cone's unknowns at once, at its first */
              if (S->ne_kdpos[a] > 0) continue;
              const int kc = S->ne_kdcone[a], d = S->q[kc], s0 = S->qs[kc]; double lam[NE_MAXCONE];
              partne_eig(S, kc, S->ne_kq + (size_t)a * NE_MAXCONE, lam);
              static double pdk2 = -1; if (pdk2 < 0) { const char *e_ = getenv("BRISK_CONEPARTNEDK"); pdk2 = e_ ? atof(e_) : 0.0; }
              const double dk_ = pdk2 > 0 ? pdk2 : fmax(S->sregx, S->sreg0);
              const int *ru = S->ne_kurow + S->ne_kus[a], nu = S->ne_kun[a];
              for (int c = 0; c < d; c++) for (int p = S->Ap[s0 + c]; p < S->Ap[s0 + c + 1]; p++) {
                  int lo = 0, hi = nu - 1; while (lo < hi) { const int mid = (lo + hi) >> 1; if (ru[mid] < S->Ai[p]) lo = mid + 1; else hi = mid; }
                  for (int t = 0; t < d; t++) pm[S->ne_kuoff[S->ne_kuo[a + t] + lo]] += S->Ax[p] * S->ne_kq[(size_t)(a + t) * NE_MAXCONE + c];
              }
              for (int t = 0; t < d; t++) pm[S->ne_kdoff[a + t]] = -(lam[t] + dk_);
              continue;
          }
          for (int p = S->Ap[j]; p < S->Ap[j + 1]; p++) pm[S->ne_kaoff[p]] += S->Ax[p];
          const int kc = S->ne_kdcone ? S->ne_kdcone[a] : -1;
          if (kc < 0) { pm[S->ne_kdoff[a]] = -(S->h[j - S->nf] + S->sregx); continue; }
          /* (BRISK_CONEPARTNE) a kept cone's block -(H_k + delta), H_k in its eigenbasis (stable: the small eigenvalue not formed by
           * cancellation): eta^2 [ (1/l) p p' + l q q' + (I - p p' - q q') ], l = (w0 + |w1|)^2, p = (1, -v)/sqrt2, q = (1, v)/sqrt2 */
          const int t = S->ne_kdpos[a], d = S->q[kc], s0 = S->qs[kc]; const double *w = S->wbar + s0, e2 = S->eta[kc] * S->eta[kc];
          double nw = 0; for (int i = 1; i < d; i++) nw += w[i] * w[i]; nw = sqrt(nw);
          const double l = (w[0] + nw) * (w[0] + nw), r2 = sqrt(0.5);
          double pv[NE_MAXCONE], qv[NE_MAXCONE];
          pv[0] = r2; qv[0] = r2;
          for (int i = 1; i < d; i++) { const double vi = nw > 0 ? w[i] / nw : (i == 1 ? 1.0 : 0.0); pv[i] = -r2 * vi; qv[i] = r2 * vi; }
          for (int b_ = 0; b_ <= t; b_++) {
              const double hv = e2 * ((b_ == t ? 1.0 : 0.0) + (1.0 / l - 1.0) * pv[t] * pv[b_] + (l - 1.0) * qv[t] * qv[b_]);
              /* (the kept block regularized as the rows, at the dual regularization, not the cone block's 1e-14: a cone near its boundary
               * has eigenvalues of H far below it and the pivot fell under the tiny-pivot threshold (db-joint-soerensen, SOCP, at the
               * switch); GMRES refines against the exact operator, so this only shapes the preconditioner) */
              static double pdk = -1; if (pdk < 0) { const char *e_ = getenv("BRISK_CONEPARTNEDK"); pdk = e_ ? atof(e_) : 0.0; }   /* (a test: the kept blocks' regularization) */
              const double dk_ = pdk > 0 ? pdk : fmax(S->sregx, S->sreg0);
              if (b_ == t) pm[S->ne_kdoff[a]] = -(hv + dk_); else pm[S->ne_kcoff[oc_++]] = -hv;
          }
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
    static double nereg_ = -1; if (nereg_ < 0) { const char *e_ = getenv("BRISK_CONENEREG"); nereg_ = e_ ? atof(e_) : 0.0; }
    /* (4 November 2026, with BRISK_CONENEBACK: once the solve came back to the normal equations and keeps them, a tiny pivot is
     * replaced by 1e-13 of its diagonal entry and left to the refinement; the correction through the Woodbury update cut the
     * steps of db-joint-soerensen, SOCP, to 0.05. Before, the default's correction stays: with the replacement from the start
     * db-plate-yield-line-fox, SOCP, 17.7 -> 47.5 s) */
    /* (BRISK_CONENEBACKDROP=1, a test: after the return, a tiny pivot dropped (1e128, its column zeroed; Wright 1999) instead) */
    static int nbd = -1; if (nbd < 0) { const char *e_ = getenv("BRISK_CONENEBACKDROP"); nbd = e_ ? atoi(e_) : 0; }
    const int keepdrop = S->ne_back && nbd && nereg_ <= 0;
    const double nereg = nereg_ > 0 ? nereg_ : (S->ne_back && !keepdrop ? 1e-13 : 0.0);   /* (a test: a tiny pivot replaced by nereg times its diagonal and left to the refinement, BRISK_CONENEREG=1e-10) */
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
    const int tf_on = tfix && nereg <= 0 && !S->use_dd && (S->ne_wU != NULL || !tfw) && !S->ne_depmode && !keepdrop;
    if (tf_on) {
        const int mm_ = S->ne_mm > 0 ? S->ne_mm : m;
        if (!S->ne_dg) S->ne_dg = (double *)malloc(sizeof(double) * ((size_t)mm_ + 1));
        for (int i = 0; i < mm_; i++) S->ne_dg[i] = 0.0;
        for (int i = 0; i < m; i++) S->ne_dg[i] = pm[S->ne_doff[i]];
        for (int a = 0; a < S->ne_nkd; a++) S->ne_dg[m + a] = pm[S->ne_kdoff[a]];
        schol_set_tinyflags(S->ne, 1); schol_set_tinyreg(1.0);
    }
    if (nereg > 0) { schol_set_tinyreg(nereg); schol_set_tinyfloor(S->sreg0); }     /* (the floor: the normal matrix's diagonal regularization, the bound of its pivots) */
    const double tf0_ = sc_time();
    S->ne_ndr = 0;
    { static int nnd3 = -1; if (nnd3 < 0) nnd3 = getenv("BRISK_CONENANDBG") != NULL;     /* (debug: the assembled normal matrix) */
      if (nnd3) { const size_t nz = schol_nnz(S->ne); long bad_ = 0; double dmin = 1e300, dmax = 0; long dneg = 0;
        for (size_t q = 0; q < nz; q++) if (!isfinite(pm[q])) bad_++;
        for (int i = 0; i < m; i++) { const double d = pm[S->ne_doff[i]]; if (d <= 0) dneg++; else { dmin = fmin(dmin, d); dmax = fmax(dmax, d); } }
        socp_printf("        [the normal matrix: %zu entries, %ld non-finite; diagonal %.1e .. %.1e, %ld not positive; nereg %.0e, tf_on %d]\n", nz, bad_, dmin, dmax, dneg, nereg, tf_on); } }
    const int bad = schol_factor(S->ne, 0.0);
    { static int nnd4 = -1; if (nnd4 < 0) nnd4 = getenv("BRISK_CONENANDBG") != NULL; if (nnd4) socp_printf("        [the normal equations' factorization: returned %d, %ld tiny pivots]\n", bad, (long)schol_ntiny(S->ne)); }
    if (getenv("BRISK_CONEFTDBG")) socp_printf("      [the normal equations' factorization alone %.3fs]\n", sc_time() - tf0_);
    if (nereg > 0 || tf_on) { schol_set_tinyreg(0.0); schol_set_tinyfloor(0.0); }
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
        if (dbg_on() >= 2) {   /* (3 November 2026, a debug print: the tiny pivots' rows, their columns and those columns' cones) */
            for (int a = 0; a < k_ && a < 4; a++) { const int r_ = rows[a];
                socp_printf("      [tiny pivot: row %d, M_ii %.3e, b_i %.3e, its columns:\n", r_, S->ne_dg[r_], S->b[r_]);
                int cnt_ = 0;
                for (int j = 0; j < S->n && cnt_ < 40; j++) for (int p = S->Ap[j]; p < S->Ap[j + 1]; p++) if (S->Ai[p] == r_) {
                    int k = -1; for (int kk = 0; kk < S->nq; kk++) if (j >= S->qs[kk] && j < S->qs[kk] + S->q[kk]) { k = kk; break; }
                    if (k < 0) { socp_printf("        col %d a %.3e (free or linear) x %.3e z %.3e\n", j, S->Ax[p], S->x[j], S->z[j]); cnt_++; continue; }
                    const int s0 = S->qs[k], d = S->q[k]; double xu = 0, zu = 0; for (int t = 1; t < d; t++) { xu += S->x[s0 + t] * S->x[s0 + t]; zu += S->z[s0 + t] * S->z[s0 + t]; }
                    socp_printf("        col %d (cone %d, entry %d) a %.3e  x: t %.3e |u| %.3e  z: t %.3e |u| %.3e  eta %.3e\n", j, k, j - s0, S->Ax[p], S->x[s0], sqrt(xu), S->z[s0], sqrt(zu), S->eta ? S->eta[k] : 0.0); cnt_++; }
                socp_printf("      ]\n"); } }
        if (room && k_ == nt && k_ <= NE_TINYMAX && S->ne_nu0 + k_ <= NE_MAXWOODRANK) {
            for (int a = 0; a < k_; a++) { S->ne_tfrow[a] = rows[a]; S->ne_tfd[a] = fabs(S->ne_dg[rows[a]]); }
            S->ne_ntf = k_; S->ne_nu = S->ne_nu0 + k_;
            /* (3 November 2026, the SOCP developer) a tiny pivot on a row that is dependent to working precision on the rows
             * factored before it: the correction's entry of C = -1/Delta + e_i' (L D L')^-1 e_i cancels to rounding (the row's
             * Schur complement is zero), so the exact correction does not exist. CBLIB's db-plate-yield-line (SOCP): at
             * iteration 27 a row whose every cone has x inside its cone and z -> 0 (a rigid part of the plate, H^-1 about 1e10)
             * is a combination of other rows on those cones' columns: the dual (the velocities) is not unique in that
             * direction, which A H^-1 A' holds only as a difference of terms of 1e12. The pivot raised to the row's diagonal is
             * kept as the treatment of a dependent row (Wright 1999), and the same term Delta e_i e_i' is put in the operator the
             * refinement solves (kkt_mul), a dual regularization of that one direction (Altman and Gondzio 1999): the factor is
             * then exact for the solved system, the refinement converges, and the normal equations are kept. Before, the zero
             * entry of C left the normal equations for the full system, whose factor needs double-double there (27 s against
             * MOSEK's 12 s). BRISK_CONETINYDEP=0: as before */
            static int tdep = -1; if (tdep < 0) { const char *e_ = getenv("BRISK_CONETINYDEP"); tdep = e_ ? atoi(e_) : 2; }   /* (0: as before; 1: the raised pivot kept with the same term in the solved system; 2: the exact correction with C from w'Mw) */
            if (tdep) {
                const int mmz = S->ne_mm > 0 ? S->ne_mm : m; double *wv = (double *)malloc(sizeof(double) * ((size_t)mmz + 8)); int kk = 0;
                for (int a = 0; a < k_ && wv; a++) {
                    const int r_ = S->ne_tfrow[a]; for (int i = 0; i < mmz; i++) wv[i] = 0.0; wv[r_] = 1.0;
                    schol_set_solve_seq(cone_seqsolve()); schol_set_solve_nt(g_cone_nt); schol_solve(S->ne, wv, S->swork2); schol_set_solve_seq(0);
                    const double cd = -1.0 / S->ne_tfd[a] + wv[r_];
                    const int dep_ = r_ < m && !(fabs(cd) > 1e-8 / S->ne_tfd[a]);
                    if (dep_ && tdep == 1) { S->ne_dprow[S->ne_ndr] = r_; S->ne_dpw[S->ne_ndr] = S->ne_tfd[a]; S->ne_ndr++; }
                    else { S->ne_tfrow[kk] = r_; S->ne_tfd[kk] = S->ne_tfd[a]; S->ne_tfdep[kk] = dep_ && tdep == 2 && S->ne_nu0 == 0 && S->ne_nfix == 0 && S->ne_nkd == 0 && mmz == m; kk++; }
                }
                free(wv);
                S->ne_ntf = kk; S->ne_nu = S->ne_nu0 + kk;
                if (S->ne_ndr > 0 && dbg_on()) socp_printf("      [the normal equations' factor: %d of %d tiny pivots on rows dependent to working precision: kept raised, the same term in the solved system]\n", S->ne_ndr, k_);
                { int nd_ = 0; for (int a = 0; a < kk; a++) nd_ += S->ne_tfdep[a]; if (nd_ > 0 && dbg_on()) socp_printf("      [the normal equations' factor: %d of %d tiny pivots on rows dependent to working precision: their correction's C from w'Mw by products]\n", nd_, k_); }
            }
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
                    schol_set_solve_seq(cone_seqsolve()); schol_set_solve_nt(g_cone_nt); schol_solve(S->ne, wv, S->swork2); schol_set_solve_seq(0);
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
    if (S->ne_nkc > 0 && dbg_on()) { int schol_ndyn(const SChol *S_); static int nd_last = 0; const int nd_ = schol_ndyn(S->ne); const int nd_this = nd_ >= nd_last ? nd_ - nd_last : nd_; nd_last = nd_;
        socp_printf("      [the partial normal equations' factor: %d tiny pivots, %d pivots of the wrong sign replaced in this factorization, %d cones kept%s]\n", nt, nd_this, S->ne_nkc, S->ne_keig ? ", in their eigenbases" : ""); }
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
        for (int t_ = 0; t_ < S->ne_nspl; t_++) {      /* (BRISK_CONENESPLIT) the large cones' largest eigenvalue: eta^-2 lmax g g' */
            double *u = U + (size_t)c * mm; memcpy(u, S->ne_splg + (size_t)t_ * mm, sizeof(double) * (size_t)mm);
            dinv[c] = 1.0 / S->ne_spld[t_]; c++;
        }
        for (int a = 0; a < S->ne_ntf; a++) {          /* (26 October 2026) the raised tiny pivots: -Delta e_i e_i' */
            double *u = U + (size_t)c * mm; for (int i = 0; i < mm; i++) u[i] = 0.0;
            u[S->ne_tfrow[a]] = 1.0; dinv[c] = -1.0 / S->ne_tfd[a]; c++;
        }
        /* (8 November 2026, the SOCP developer, a test: BRISK_CONEWOODMANY=1) the nu columns in one solve with nu right-hand sides,
         * one pass over the factor instead of nu (the solves are bound by the memory: robust dose P11_06, nnz(L) 4.6e7, about 50 ms
         * a pass, two of them a factorization for the rank-two update) */
        static int wmany = -1; if (wmany < 0) { const char *e_ = getenv("BRISK_CONEWOODMANY"); wmany = e_ ? atoi(e_) : 0; }
        if (wmany && nu >= 2) { extern void schol_solve_many(const SChol *S, double *Bm, int nrhs, double *work); memcpy(W, U, sizeof(double) * (size_t)mm * nu); schol_solve_many(S->ne, W, nu, S->swork2); }
        else
        for (int f = 0; f < nu; f++) { memcpy(W + (size_t)f * mm, U + (size_t)f * mm, sizeof(double) * (size_t)mm); schol_set_solve_seq(cone_seqsolve()); schol_set_solve_nt(g_cone_nt); schol_solve(S->ne, W + (size_t)f * mm, S->swork2); schol_set_solve_seq(0); }
        if (nu > 16) {              /* C = U' W by dgemm (the sparse cones' update alone stays at most 16: the scalar sum, the same bits) (symmetric: the row-major reads below see the same entries), then D^-1 on the diagonal */
            const double one = 1.0, zero = 0.0; SC_BL(dgemm_)("T", "N", &nu, &nu, &mm, &one, U, &mm, W, &mm, &zero, C, &nu);
            for (int f = 0; f < nu; f++) C[f * nu + f] += dinv[f];
        } else
        for (int f = 0; f < nu; f++) for (int g = 0; g < nu; g++) { double v = 0; const double *uf = U + (size_t)f * mm, *wg = W + (size_t)g * mm; for (int i = 0; i < mm; i++) v += uf[i] * wg[i]; C[f * nu + g] = v + (f == g ? dinv[f] : 0.0); }
        /* (3 November 2026, the SOCP developer) a correction on a row dependent to working precision (ne_factor): with F the
         * factored matrix and w = F^-1 e_i, M w = -Delta c e_i, so c = -w'Mw / (Delta e_i'w), and w'Mw = |H^-1/2 A'w|^2 +
         * delta |w|^2 is a sum of terms of one sign formed by products with A: no cancellation, where -1/Delta + e_i'w
         * cancels to rounding (only when the corrections are the update's sole columns, F = M + sum Delta e_i e_i') */
        if (S->ne_ntf > 0 && S->ne_nu0 == 0) for (int a = 0; a < S->ne_ntf; a++) if (S->ne_tfdep[a]) {
            const int f = nu - S->ne_ntf + a, n_ = S->n; const double *w = W + (size_t)f * mm;
            double *ax = (double *)calloc((size_t)n_ + 1, sizeof(double)), *hx = (double *)malloc(sizeof(double) * ((size_t)n_ + 1));
            if (!ax || !hx) { free(ax); free(hx); continue; }
            At_mul(S, 1.0, w, ax); for (int j = 0; j < S->nf; j++) ax[j] = 0.0;
            Hinv_apply(S, ax, hx);
            double q = 0, ww = 0; for (int j = 0; j < n_; j++) q += ax[j] * hx[j]; for (int i = 0; i < m; i++) ww += w[i] * w[i];
            q += S->sreg0 * ww;
            const double ew = w[S->ne_tfrow[a]], cold = C[f * nu + f];
            if (ew != 0.0 && q > 0) { C[f * nu + f] = -q / (S->ne_tfd[a] * ew); S->ne_depused = 1; }
            if (dbg_on()) socp_printf("      [the dependent row %d: C from the factor %.3e, from w'Mw %.3e (w'Mw %.3e, e'w %.3e, 1/Delta %.3e)]\n", S->ne_tfrow[a], cold, C[f * nu + f], q, ew, 1.0 / S->ne_tfd[a]);
            free(ax); free(hx);
        }
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
            schol_set_solve_seq(cone_seqsolve()); schol_set_solve_nt(g_cone_nt); schol_solve(S->ne, t, S->swork2); schol_set_solve_seq(0); ne_wood_apply(S, t);
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
            schol_set_solve_seq(cone_seqsolve()); schol_set_solve_nt(g_cone_nt); schol_solve(S->ne, w, S->swork2); schol_set_solve_seq(0);
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
      if ((ktiny || nereg > 0 || keepdrop) && bad == 0) return 0;
      if (nt > 0 && S->ne_ntf + S->ne_ndr == nt && bad == 0) return 0;     /* (corrected exactly, or kept raised on dependent rows with the same term in the solved system: above) */
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
    if (S->ne_onec >= 0 && S->ne_onec_ok && S->ne_scale != 1.0) for (int i = 0; i < m; i++) t[i] *= S->ne_scale;   /* (ne_factor_onec: M^-1 = eta^2 (K + U D' U')^-1) */
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
/* (4 November 2026, the SOCP developer, a test: BRISK_CONENEIR=k) the normal equations' solve refined k times against
 * N = A H^-1 A' + delta I with the residual in double-double: near the end the entries of H^-1 reach 1e9 (a cone or a linear
 * column whose x is large and z small), and a product with N in double carries a rounding of about 1e-7 |dy|, which is the
 * floor of the rows' residual of every solve (db-joint-soerensen, SOCP, the normal equations kept to the end: the rows'
 * residual 1e-6 .. 1e-2 of the right-hand side from iteration 39, the primal residual rising from 1.2e-8 to 2e-7). The
 * right-hand side ry + A H^-1 rx and the primal part dx = H^-1 (A'dy - rx) are formed the same way. Three passes over A in
 * double-double for each refinement, against a factorization in double-double */
static inline sdd dd_add(sdd a, sdd b) { b.h = -b.h; b.l = -b.l; return dd_sub(a, b); }
static void ne_At_dd(const Socp *S, const double *vh, const double *vl, double *oh, double *ol) {
    #pragma omp parallel for schedule(static) num_threads(g_cone_nt) if (cone_apar(S))
    for (int j = 0; j < S->n; j++) { sdd a = { 0.0, 0.0 };
        for (int p = S->Ap[j]; p < S->Ap[j + 1]; p++) { const int i = S->Ai[p]; const sdd x = { vh[i], vl ? vl[i] : 0.0 }, c = { S->Ax[p], 0.0 }; a = dd_add(a, dd_mul(c, x)); }
        oh[j] = a.h; ol[j] = a.l; }
}
static void ne_A_dd(const Socp *S, const double *xh, const double *xl, double *oh, double *ol) {     /* (o += A x, o in double-double) */
    if (S->Rp) {
        #pragma omp parallel for schedule(static) num_threads(g_cone_nt) if (cone_apar(S))
        for (int i = 0; i < S->m; i++) { sdd a = { oh[i], ol[i] };
            for (int q = S->Rp[i]; q < S->Rp[i + 1]; q++) { const int j = S->Ri[q]; const sdd x = { xh[j], xl[j] }, c = { S->Rx[q], 0.0 }; a = dd_add(a, dd_mul(c, x)); }
            oh[i] = a.h; ol[i] = a.l; }
        return;
    }
    for (int j = 0; j < S->n; j++) { const sdd x = { xh[j], xl[j] };
        for (int p = S->Ap[j]; p < S->Ap[j + 1]; p++) { const int i = S->Ai[p]; const sdd c = { S->Ax[p], 0.0 }; sdd a = { oh[i], ol[i] }; a = dd_add(a, dd_mul(c, x)); oh[i] = a.h; ol[i] = a.l; } }
}
static void ne_Hinv_dd(const Socp *S, const double *ih, const double *il, double *oh, double *ol) {
    #pragma omp parallel for schedule(static) num_threads(g_cone_nt) if (cone_vpar(S))
    for (int i = 0; i < S->nl; i++) { const int j = S->nf + i; const sdd x = { ih[j], il[j] }; const sdd q = dd_mul(x, dd_inv((sdd){ S->h[i], 0.0 })); oh[j] = q.h; ol[j] = q.l; }
    #pragma omp parallel for schedule(static) num_threads(g_cone_nt) if (cone_vpar(S))
    for (int k = 0; k < S->nq; k++) {
        const int d = S->q[k], s = S->qs[k]; const double *w = S->wbar + s;
        const sdd et = { S->eta[k], 0.0 }, ie2 = dd_inv(dd_mul(et, et));
        sdd ux = dd_mul((sdd){ w[0], 0.0 }, (sdd){ ih[s], il[s] });
        for (int i = 1; i < d; i++) ux = dd_sub(ux, dd_mul((sdd){ w[i], 0.0 }, (sdd){ ih[s + i], il[s + i] }));
        const sdd u2 = dd_add(ux, ux);
        { const sdd o = dd_mul(ie2, dd_sub(dd_mul(u2, (sdd){ w[0], 0.0 }), (sdd){ ih[s], il[s] })); oh[s] = o.h; ol[s] = o.l; }
        for (int i = 1; i < d; i++) { const sdd o = dd_mul(ie2, dd_sub((sdd){ ih[s + i], il[s + i] }, dd_mul(u2, (sdd){ w[i], 0.0 }))); oh[s + i] = o.h; ol[s + i] = o.l; }
    }
}
static int ne_ir_steps(void) { static int k = -1; if (k < 0) { const char *e_ = getenv("BRISK_CONENEIR"); k = e_ ? atoi(e_) : 0; } return k; }
static void ne_solve_t(Socp *S, double *t) {
    { const double ts = sc_time(); if (!(ne_part() && S->ne_lnz >= 4000000L && (g_part_try++, schol_solve_part(S->ne, t, S->swork2, cone_part_nt()) == 0) && (g_part_ok++, 1))) { schol_set_solve_seq(cone_seqsolve()); schol_set_solve_nt(g_cone_nt); schol_solve(S->ne, t, S->swork2); schol_set_solve_seq(0); } S->t_solve += sc_time() - ts; S->nback++; }
    if (S->ne_nu > 0) ne_wood_apply(S, t);
}
static void ne_prec_ir(Socp *S, const double *r, double *v, int kir) {
    const int n = S->n, m = S->m;
    double *b = (double *)malloc(sizeof(double) * (size_t)(6 * (size_t)n + 4 * (size_t)m));
    double *xh = b, *xl = b + n, *zh = b + 2 * (size_t)n, *zl = b + 3 * (size_t)n, *zero = b + 4 * (size_t)n;
    double *th = b + 6 * (size_t)n, *tl = th + m, *qh = tl + m, *ql = qh + m;
    memset(zero, 0, sizeof(double) * 2 * (size_t)n);
    /* t0 = ry + A H^-1 rx, in double-double */
    ne_Hinv_dd(S, r, zero, xh, xl);
    for (int i = 0; i < m; i++) { th[i] = r[n + i]; tl[i] = 0.0; }
    ne_A_dd(S, xh, xl, th, tl);
    double *dy = S->rk;
    for (int i = 0; i < m; i++) dy[i] = th[i] + tl[i];
    ne_solve_t(S, dy);
    double *c = (double *)malloc(sizeof(double) * (size_t)m);
    const double dl = S->sreg0;
    for (int it = 0; it < kir; it++) {
        /* q = A H^-1 A' dy + delta dy; the residual t0 - q, rounded to double */
        ne_At_dd(S, dy, NULL, zh, zl);
        ne_Hinv_dd(S, zh, zl, xh, xl);
        for (int i = 0; i < m; i++) { const sdd q = dd_mul((sdd){ dl, 0.0 }, (sdd){ dy[i], 0.0 }); qh[i] = q.h; ql[i] = q.l; }
        ne_A_dd(S, xh, xl, qh, ql);
        for (int i = 0; i < m; i++) { const sdd e = dd_sub((sdd){ th[i], tl[i] }, (sdd){ qh[i], ql[i] }); c[i] = e.h + e.l; }
        ne_solve_t(S, c);
        for (int i = 0; i < m; i++) dy[i] += c[i];
    }
    /* dx = H^-1 (A' dy - rx), in double-double */
    ne_At_dd(S, dy, NULL, zh, zl);
    for (int j = 0; j < n; j++) { const sdd e = dd_sub((sdd){ zh[j], zl[j] }, (sdd){ r[j], 0.0 }); zh[j] = e.h; zl[j] = e.l; }
    ne_Hinv_dd(S, zh, zl, xh, xl);
    for (int j = 0; j < n; j++) v[j] = xh[j] + xl[j];
    for (int i = 0; i < m; i++) v[n + i] = dy[i];
    free(c); free(b);
}
/* (4 November 2026, the SOCP developer) the copies and the negation of a million entries on the threads, entry by entry (the
 * same bits): on one thread they were a part of each application's 15 ms beside the solve (db-joint-soerensen, SOCP) */
static void ne_prec(Socp *S, const double *r, double *v) {
    { const int kir = ne_ir_steps(); if (kir > 0 && S->ne_iron && S->nf == 0 && S->ne_nfix == 0 && S->ne_nkd == 0 && S->nd == 0) { ne_prec_ir(S, r, v, kir); return; } }
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
        #pragma omp parallel for schedule(static) num_threads(g_cone_nt) if (cone_vpar(S))
        for (int i = 0; i < m; i++) t[i] = r[n + i];
        A_mul(S, 1.0, hx, t);
        for (int kc = 0; S->ne_nfix > 0 && kc < S->nq; kc++) if (S->ne_fixrow[kc] >= 0) t[S->ne_fixrow[kc]] = 0.0;
        for (int a = 0; a < S->ne_nkd; a++) {
            if (S->ne_keig && S->ne_kun[a] >= 0) { const int kc = S->ne_kdcone[a], s0 = S->qs[kc]; const double *qa = S->ne_kq + (size_t)a * NE_MAXCONE;
                double v_ = 0; for (int c = 0; c < S->q[kc]; c++) v_ += qa[c] * r[s0 + c]; t[m + a] = v_; }
            else t[m + a] = r[S->ne_kdcol[a]]; }
        static int nnd2 = -1; if (nnd2 < 0) nnd2 = getenv("BRISK_CONENANDBG") != NULL;
        long c0 = 0, cr = 0; if (nnd2) { for (int i = 0; i < m; i++) if (!isfinite(t[i])) c0++; for (int j = 0; j < n + m; j++) if (!isfinite(r[j])) cr++; }
        { const double ts = sc_time(); if (!(ne_part() && S->ne_lnz >= 4000000L && (g_part_try++, schol_solve_part(S->ne, t, S->swork2, cone_part_nt()) == 0) && (g_part_ok++, 1))) { schol_set_solve_seq(cone_seqsolve()); schol_set_solve_nt(g_cone_nt); schol_solve(S->ne, t, S->swork2); schol_set_solve_seq(0); } S->t_solve += sc_time() - ts; S->nback++; }
        long c1 = 0; double cm = 0; if (nnd2) { for (int i = 0; i < m; i++) { if (!isfinite(t[i])) c1++; else cm = fmax(cm, fabs(t[i])); } }
        if (S->ne_nu > 0) ne_wood_apply(S, t);
        if (nnd2) { long c2 = 0; for (int i = 0; i < m; i++) if (!isfinite(t[i])) c2++;
            if (cr || c0 || c1 || c2) socp_printf("        [ne_prec (fixed rows): non-finite: rhs %ld, the folded right-hand side %ld, after the factor's solve %ld (largest finite %.1e), after the Woodbury update %ld (nu %d)]\n", cr, c0, c1, cm, c2, S->ne_nu); }
        for (int kc = 0; S->ne_nfix > 0 && kc < S->nq; kc++) if (S->ne_fixrow[kc] >= 0) t[S->ne_fixrow[kc]] = 0.0;
        #pragma omp parallel for schedule(static) num_threads(g_cone_nt) if (cone_vpar(S))
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
        for (int a = 0; a < S->ne_nkd; a++) {
            if (S->ne_keig && S->ne_kun[a] >= 0) { const int kc = S->ne_kdcone[a], a0 = a - S->ne_kdpos[a], c = S->ne_kdpos[a];
                double v_ = 0; for (int tt = 0; tt < S->q[kc]; tt++) v_ += S->ne_kq[(size_t)(a0 + tt) * NE_MAXCONE + c] * t[m + a0 + tt]; v[S->ne_kdcol[a]] = v_; }
            else v[S->ne_kdcol[a]] = t[m + a]; }
        #pragma omp parallel for schedule(static) num_threads(g_cone_nt) if (cone_vpar(S))
        for (int i = 0; i < m; i++) v[n + i] = t[i];
        return;
    }
    #pragma omp parallel for schedule(static) num_threads(g_cone_nt) if (cone_vpar(S))
    for (int i = 0; i < m; i++) t[i] = r[n + i];
    A_mul(S, 1.0, hx, t);
    static int nnd = -1; if (nnd < 0) nnd = getenv("BRISK_CONENANDBG") != NULL;
    long b0 = 0; if (nnd) { for (int i = 0; i < m; i++) if (!isfinite(t[i])) b0++; }
    { const double ts = sc_time(); if (!(ne_part() && S->ne_lnz >= 4000000L && (g_part_try++, schol_solve_part(S->ne, t, S->swork2, cone_part_nt()) == 0) && (g_part_ok++, 1))) { schol_set_solve_seq(cone_seqsolve()); schol_set_solve_nt(g_cone_nt); schol_solve(S->ne, t, S->swork2); schol_set_solve_seq(0); } S->t_solve += sc_time() - ts; S->nback++; }
    long b1 = 0; double tm = 0; if (nnd) { for (int i = 0; i < m; i++) { if (!isfinite(t[i])) b1++; else tm = fmax(tm, fabs(t[i])); } }
    if (S->ne_nu > 0) ne_wood_apply(S, t);
    if (nnd) { long b2 = 0, br = 0; for (int i = 0; i < m; i++) if (!isfinite(t[i])) b2++; for (int j = 0; j < n + m; j++) if (!isfinite(r[j])) br++;
        if (b0 || b1 || b2 || br) socp_printf("        [ne_prec: non-finite: rhs %ld, A H^-1 rx + ry %ld, after the factor's solve %ld (largest finite %.1e), after the Woodbury update %ld (nu %d)]\n", br, b0, b1, tm, b2, S->ne_nu); }
    double xfs[NE_MAXFREE]; double *xf = S->nf > NE_MAXFREE ? S->ne_xf : xfs;
    if (S->nf > 0) {
        const int nf = S->nf; const double *T = S->ne_T;
        for (int f = 0; f < nf; f++) { double g = -r[f]; for (int p = S->Ap[f]; p < S->Ap[f + 1]; p++) g += S->Ax[p] * t[S->Ai[p]]; xf[f] = g; }
        for (int f = 0; f < nf; f++) { double v = xf[f]; for (int k = 0; k < f; k++) v -= T[f * nf + k] * xf[k]; xf[f] = v / T[f * nf + f]; }
        for (int f = nf - 1; f >= 0; f--) { double v = xf[f]; for (int k = f + 1; k < nf; k++) v -= T[k * nf + f] * xf[k]; xf[f] = v / T[f * nf + f]; }
        for (int f = 0; f < nf; f++) { const double *w = S->ne_W + (size_t)f * m; const double a_ = xf[f]; for (int i = 0; i < m; i++) t[i] -= a_ * w[i]; }
    }
    #pragma omp parallel for schedule(static) num_threads(g_cone_nt) if (cone_vpar(S))
    for (int j = 0; j < n; j++) hx[j] = -r[j];
    At_mul(S, 1.0, t, hx);
    Hinv_apply(S, hx, v);
    for (int f = 0; f < S->nf; f++) v[f] = xf[f];
    #pragma omp parallel for schedule(static) num_threads(g_cone_nt) if (cone_vpar(S))
    for (int i = 0; i < m; i++) v[n + i] = t[i];
}

void schol_set_dynpiv(SChol *S, double p);
int schol_ndyn(const SChol *S);
static void kkt_fill(Socp *S);
/* (2 November 2026, the SOCP developer) a raise of the factored matrix's regularization in the iteration that tried the level
 * 100 times lower: the lower level failed, and the decay is held off for the next 4 factorizations. The failure shows in
 * three places (a wrong-sign pivot in the factorization, a solve that wants double-double when double-double is too large,
 * a direction that is not finite), most often in the second: the solve after the factorization flags the pivots and the
 * iteration factors again. The hold was judged only right after the first factorization of the call that tried, so it was
 * never set on that path and every iteration tried again and factored twice: db-joint-soerensen (SOCP), 50 of 75 */
static void bump_raised(Socp *S) { if (S->bump_try_it > 0 && S->bump_try_it == S->cur_it + 1) S->bump_hold = 4; }
static int kkt_factor(Socp *S) {
    Ldl *F = &S->F;
    F->dd = 0;
    { static int nandbg = -1; if (nandbg < 0) nandbg = getenv("BRISK_CONENANDBG") != NULL;     /* (debug: a non-finite entry of the assembled matrix, and its kind) */
      if (nandbg) { long bad = 0, first = -1; for (long q = 0; q < S->knz; q++) if (!isfinite(F->Ax[q])) { if (first < 0) first = q; bad++; }
        long bw = 0; int kw = -1; for (int k = 0; k < S->nq; k++) { if (!isfinite(S->eta[k])) { bw++; if (kw < 0) kw = k; } for (int i = 0; i < S->q[k]; i++) if (!isfinite(S->wbar[S->qs[k] + i])) { bw++; if (kw < 0) kw = k; } }
        if (bad) { int shown = 0; for (int j = 0; j < F->n && shown < 6; j++) for (int p = F->Ap[j]; p < F->Ap[j + 1] && shown < 6; p++) if (!isfinite(F->Ax[p])) { shown++;
              const int oj = j, oi = F->Ai[p]; socp_printf("        [non-finite entry %d: row %d column %d value %.3e (nf %d n %d naux %d)]\n", p, oi, oj, F->Ax[p], S->nf, S->n, S->naux); } }
        if (bad) socp_printf("        [knz %ld, the pattern's entries %d, n %d nk %d, sc %d]\n", S->knz, F->Ap[F->n], F->n, S->nk, S->sc != NULL);
        if (bad || bw) socp_printf("      [non-finite: %ld entries of the matrix (the first at %ld), %ld scaling values (cone %d: eta %.3e, w0 %.3e, rotated %d)]\n", bad, first, bw, kw, kw >= 0 ? S->eta[kw] : 0.0, kw >= 0 ? S->wbar[S->qs[kw]] : 0.0, kw >= 0 && S->rot ? S->rot[kw] : -1); } }
    { static int who = -2; if (who == -2) { const char *e_ = getenv("BRISK_CONEWHO"); who = e_ ? atoi(e_) : -1; }    /* (debug: what an unknown of the full system is) */
      if (who >= 0 && who < S->n) { int k = -1; for (int t = 0; t < S->nq; t++) if (who >= S->qs[t] && who < S->qs[t] + S->q[t]) { k = t; break; }
        if (k < 0) socp_printf("      [unknown %d: %s, x %.3e z %.3e]\n", who, who < S->nf ? "free" : "linear", S->x[who], S->z[who]);
        else { const double *w = S->wbar + S->qs[k]; double nq = 0; for (int i = 1; i < S->q[k]; i++) nq += w[i] * w[i]; nq = sqrt(nq);
          socp_printf("      [unknown %d: cone %d (dimension %d, entry %d), auxiliary %d, eigenbasis %d; eta %.3e, w0 %.6e, |w1| %.6e, the eigenvalue ratio %.3e]\n", who, k, S->q[k], who - S->qs[k], S->auxu[k], S->rot ? S->rot[k] : -1, S->eta[k], w[0], nq, pow(w[0] + nq, 4)); } } }
    if (S->use_dd) { S->sc_ok = 0;
        { static int ddfl = -1; if (ddfl < 0) { const char *e_ = getenv("BRISK_CONEDDFLOOR"); ddfl = e_ ? atoi(e_) : 0; }   /* (a test, off: 1 the floor; 0 the sign test alone, as before) */
          F->pivfloor = ddfl ? 0.5 * fmin(S->sreg, fmin(S->sregx, S->nf > 0 ? S->sreg : 1e300)) : 0.0; F->flaux0 = S->n; F->flaux1 = S->n + S->naux; }
        const int rdd_ = ldl_numeric_dd(F);
        if (getenv("BRISK_CONENANDBG")) { long bad = 0; int first = -1; for (int i = 0; i < F->n; i++) if (!isfinite(F->D[i]) || !isfinite(F->Dinv[i])) { bad++; if (first < 0) first = i; }
            long bl = 0; long fl = -1; for (long q = 0; q < F->lnz; q++) if (!isfinite(F->Lx[q])) { bl++; if (fl < 0) fl = q; }
            if (bad || bl) { const int o = first >= 0 ? F->perm[first] : -1; socp_printf("      [non-finite in the double-double factor: %ld pivots (the first at %d, original %d: %s, D %.3e), %ld entries of L]\n", bad, first, o, o < 0 ? "-" : o < S->nf ? "free" : o < S->n ? "cone or linear" : o < S->n + S->naux ? "auxiliary" : "row", first >= 0 ? F->D[first] : 0.0, bl); } }
        return rdd_; }
    if (!S->sc) {
        /* (4 November 2026, the SOCP developer, a test: BRISK_CONEQDBIG=1) Wright's remedy at the bound of the quasi-definite
         * matrix: a pivot below half its bound (or of the wrong sign) is not a pivot of the matrix (rounding made it); it is
         * replaced by sign x BRISK_CONEBIGPIV (set it to 1e64), which decouples its unknown instead of amplifying the rows after
         * it (a replacement of 1e-7 made L grow to 1e308 on db-joint-soerensen, SOCP) */
        { static int qdb = -1; if (qdb < 0) { const char *e_ = getenv("BRISK_CONEQDBIG"); qdb = e_ ? atoi(e_) : 0; }
          if (qdb) { F->pivfloor = 0.5 * fmin(S->sreg, S->sregx); F->flaux0 = S->n; F->flaux1 = S->n + S->naux; } }
        int f = ldl_numeric(F);
        F->pivfloor = 0.0;
        /* (3 November 2026, the SOCP developer, a test: BRISK_CONEQDCHECK=1) the double factor checked against the bound of the
         * quasi-definite matrix: with -(H + dx) in the cone and linear block and +dr in the rows' block, every pivot of the
         * first kind is at most -dx and every pivot of the second at least dr in exact arithmetic (the Schur complements stay
         * quasi-definite). A pivot of the wrong sign, or one below half its bound, means rounding destroyed the factor (its
         * growth then overflows: db-joint-soerensen, SOCP, with the free variables substituted: at a regularization of 1e-9 the
         * pivots fell to 1e-25 and 3e-26, the replaced ones by the thousand, L up to 1e308, the direction NaN). Such a factor is
         * not used: the regularization of the factored matrix is raised (GMRES removes it) and the matrix factored again */
        static int qdc_c = -2; const int qdc = cone_pkg("BRISK_CONEQDCHECK", &qdc_c);
        /* (only where double-double is refused for its size: there the factor is used as it is otherwise, and its growth
         * overflows. Where double-double is allowed the default's response stays: wrong-sign pivots send the solve to
         * double-double (joint_FC_5, SOCP: from iteration 18, 54 s OPTIMAL; with the raise instead 135 s, 127 iterations,
         * REDUCED)) */
        static double qddmax = -1; if (qddmax < 0) { const char *e_ = getenv("BRISK_CONEDDMAX"); qddmax = e_ ? atof(e_) : 5e8; }
        const int qd_dd = S->dd_on && S->cflops <= qddmax;
        if (qdc == 2 && qd_dd && !S->use_dd) {     /* (BRISK_CONEQDCHECK=2 only: socp_rocket_4000 (SOCP) 2.2 -> 17.7 s with it, pivots below the
                                                     * bound that the refinement corrects sent to double-double; joint_FC_5 needs nothing here) */
            const double fl = 0.5 * fmin(S->sreg, S->sregx); long nlow = 0;
            for (int k = 0; k < F->n; k++) { const int o = F->perm[k]; if (o >= S->n && o < S->n + S->naux) continue; if (o < S->nf) continue;
                if (!(F->sign[k] * F->D[k] >= fl)) nlow++; }
            if (nlow > F->ndyn) { F->ndyn = (int)nlow; f = F->ndyn; }
        }
        /* (4 November 2026, a test: BRISK_CONENEBACK=1) the full system's double factor below the quasi-definite bound while the
         * normal equations are parked and double-double is refused for its size: back to the normal equations, kept to the end
         * (db-joint-soerensen, SOCP: all free variables substituted, the normal matrix positive definite by construction; its
         * full system breaks the bound from iteration 37) */
        if (S->ne_park && !qd_dd && !S->use_dd) {
            const double fl = 0.5 * fmin(S->sreg, S->sregx); long nlow = 0;
            for (int k = 0; k < F->n; k++) { const int o = F->perm[k]; if (o >= S->n && o < S->n + S->naux) continue; if (o < S->nf) continue;
                if (!(F->sign[k] * F->D[k] >= fl)) nlow++; }
            if (nlow > 0 || F->ndyn > 0) {
                S->ne = S->ne_park; S->ne_park = NULL; S->ne_back = 1;
                if (dbg_on()) socp_printf("      [the full system's double factor below the quasi-definite bound (%ld pivots): back to the normal equations, kept]\n", nlow);
                S->ne_ok = !ne_factor(S);
                if (S->ne_ok) { F->ndyn = 0; F->dd = 0; return 0; }
                ne_free(S); kkt_fill(S); f = ldl_numeric(F);
            }
        }
        if (qdc && !qd_dd && S->bump_on && !S->use_dd) {
            for (int t = 0; t < 6; t++) {
                const double fl = 0.5 * fmin(S->sreg, S->sregx); long nlow = 0;
                for (int k = 0; k < F->n; k++) { const int o = F->perm[k]; if (o >= S->n && o < S->n + S->naux) continue; if (o < S->nf) continue;
                    if (!(F->sign[k] * F->D[k] >= fl)) nlow++; }
                if (nlow == 0 && F->ndyn == 0) break;
                /* (a cone's pivot below the bound is the cancellation of the small eigenvalue of its block of H that the
                 * eigenbasis removes (ISSUES 259): the switch to it is asked for, at the end of this iteration) */
                if (getenv("BRISK_ROTDBG")) socp_printf("      [bound check: rot_mode %d anyrot %d nsmallc %d]\n", S->rot_mode, S->anyrot, S->nsmallc);
                if ((S->rot_mode == 2 || (S->rot_mode == 0 && S->rot_back)) && !S->anyrot && S->nsmallc > 0) S->rot_want = 1;     /* (a replaced pivot no longer shows which
                 * block it was in; also after the switch back for its cost: the cost rule weighs the work only, and a factor below
                 * the bound is not a factor of the matrix) */
                if (S->bump >= 1e4) break;
                { static double qbf = -1; if (qbf < 0) { const char *e_ = getenv("BRISK_CONEQDSTEP"); qbf = e_ ? atof(e_) : 100.0; } S->bump *= qbf; }   /* (a test: the raise in steps of BRISK_CONEQDSTEP) */ S->nbump++; bump_raised(S);
                S->sreg = S->sreg0 * S->bump; S->sregx = S->sreg0 * S->bump; S->sregf = fmax(1e-6, 1e-8 * S->bump);
                S->qd_raised = 1; S->qd_ever = 1;
                if (dbg_on()) socp_printf("      [the double factor below the quasi-definite bound (%ld pivots, %d replaced): the factored matrix's regularization raised to %.1e]\n", nlow, F->ndyn, S->sregx);
                kkt_fill(S);
                f = ldl_numeric(F);
            }
        }
        if (getenv("BRISK_CONENANDBG")) { long bad = 0; int first = -1; double dmin = 1e300, dmax = 0; for (int i = 0; i < F->n; i++) { if (!isfinite(F->D[i]) || !isfinite(F->Dinv[i])) { bad++; if (first < 0) first = i; } else { dmin = fmin(dmin, fabs(F->D[i])); dmax = fmax(dmax, fabs(F->D[i])); } }
            long bl = 0; double lmax = 0; for (long q = 0; q < F->lnz; q++) { if (!isfinite(F->Lx[q])) bl++; else lmax = fmax(lmax, fabs(F->Lx[q])); }
            socp_printf("      [the double factor: %ld non-finite pivots (the first at %d), %ld non-finite entries of L; |D| %.1e .. %.1e, max |L| %.1e, ndyn %d]\n", bad, first, bl, dmin, dmax, lmax, F->ndyn); }
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
    if (getenv("BRISK_CONENANDBG") && S->sc_ok) {     /* (debug: the supernodal factor applied to a vector of ones) */
        const int nk_ = S->nk; double *u_ = (double *)malloc(sizeof(double) * (size_t)nk_ * 3);
        if (u_) { for (int i = 0; i < nk_; i++) u_[i] = 1.0; schol_solve(S->sc, u_, u_ + nk_);
            long bad = 0; int first = -1; for (int i = 0; i < nk_; i++) if (!isfinite(u_[i])) { bad++; if (first < 0) first = i; }
            if (bad) socp_printf("      [non-finite: the supernodal factor's solve gives %ld non-finite entries (the first at %d, original %d); replaced pivots %d]\n", bad, first, F->perm ? F->perm[first] : -1, F->ndyn);
            free(u_); } }
    if (S->sc_ok) return 0;
    /* a pivot of the wrong sign. First a larger regularization of the factored matrix (GMRES removes it:
     * the system it solves keeps the base values); the level is kept for the later iterations */
    if (S->bump_on) {
        const double r0 = S->sreg, x0 = S->sregx, f0 = S->sregf;
        for (int t = 0; t < 2 && !S->sc_ok && S->bump < 1e4; t++) {
            S->bump *= 100.0; S->nbump++; bump_raised(S);
            S->sreg = S->sreg0 * S->bump; S->sregx = S->sreg0 * S->bump; S->sregf = fmax(1e-6, 1e-8 * S->bump);
            if (dbg_on()) socp_printf("      [a pivot of the wrong sign: the factored matrix's regularization raised to %.1e]\n", S->sregx);
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
    cone_threads_adapt(S->verbose);         /* (the busy-machine rule, 28 October 2026) */
    const double ts = sc_time();
    if (S->ne) {
        S->ne_ok = !ne_factor(S);
        if (!S->ne_ok && dbg_on()) socp_printf("      [the normal equations' factorization failed (tiny pivots so far %d): the full system from here]\n", S->ne_tiny);
        if (S->ne_ok) { S->F.ndyn = 0; S->F.dd = 0; S->t_factor += sc_time() - ts; S->nfact++; if (getenv("BRISK_CONEFTDBG")) socp_printf("      [the normal equations formed and factored in %.3fs]\n", sc_time() - ts); return 0; }
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
    double b_try = 0.0; (void)b_try;
    if (bdecay && S->bump_on && S->bump > 1.0 && !S->use_dd) {
        if (S->bump_hold > 0) S->bump_hold--;
        else {
            S->bump /= 100.0; if (S->bump < 1.5) S->bump = 1.0;
            S->sreg = S->sreg0 * S->bump; S->sregx = S->sreg0 * S->bump; S->sregf = fmax(1e-6, 1e-8 * S->bump);
            kkt_fill(S); b_try = S->bump; S->bump_try_it = S->cur_it + 1;
            if (dbg_on()) socp_printf("      [the factored matrix's regularization tried 100 times lower: %.1e]\n", S->sregx);
        }
    }
    int f = kkt_factor(S);
    /* (13 November 2026, the SOCP developer, a test: BRISK_CONEROTDEF=1) a small cone's dense block -(H_k + delta) written in
     * ordinary coordinates holds its smallest eigenvalue eta^2 / l, l = (w0 + |w1|)^2, only while the rounding of its entries
     * (about eps eta^2 l) stays below eta^2 / l + delta: past that the written matrix is no longer quasi-definite, its factor has
     * pivots of the wrong sign, and double-double cannot repair a matrix rounded when it was written (socp_socopf_3012wp_k, SOCP:
     * four cones of dimension 4 at eigenvalue ratio 1e18, the double-double factor overflowed, the direction NaN). Replaced
     * pivots with such a cone, and a solve with this factor that did not refine, switch the small cones to their eigenbasis
     * (its block diagonal with the exact eigenvalues) at once, before double-double; where the solve refines, nothing changes
     * (socp_huber_20000x200, SOCP: replaced pivots with such cones, the solves refined; an eigenbasis asked for there cost 0.8 s
     * of set-up) */
    { static int rdef = -1; if (rdef < 0) { const char *e_ = getenv("BRISK_CONEROTDEF"); rdef = e_ ? atoi(e_) : 2; }   /* (2, the default since 13 November 2026: in double and double-double; 1 in double only; 0 off) */
      S->rot_def = 0;
      if (rdef && S->F.ndyn > 0 && (!S->use_dd || rdef >= 2) && !S->anyrot && !S->rot_want && S->nsmallc > 0 && S->rot && !S->ne
          && ((S->rot_mode == 2 && !S->rot_switch) || (S->rot_mode == 0 && S->rot_back))) {
          long nlost = 0;
          for (int k = 0; k < S->nq; k++) {
              const int d = S->q[k]; if (d < 2 || d > SOCP_DENSE_MAX || S->auxu[k] >= 0 || S->rot[k]) continue;
              const double *w = S->wbar + S->qs[k]; double nq = 0.0; for (int i = 1; i < d; i++) nq += w[i] * w[i];
              const double l = (w[0] + sqrt(nq)) * (w[0] + sqrt(nq)), e2 = S->eta[k] * S->eta[k];
              if (2.220446049250313e-16 * e2 * l > e2 / l + S->sregx) nlost++;
          }
          if (nlost) { S->rot_def = 1;      /* (acted on where the solve with this factor fails and double-double would follow: below, after the first solve) */
              if (dbg_on()) socp_printf("      [%ld small cones' blocks not definite as written (rounding above their smallest eigenvalue)]\n", nlost); }
      } }
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
    /* (3 November 2026, the SOCP developer, a test: BRISK_CONEDDAGAIN=1) a request for double-double that comes again after
     * one was refused for its size and answered with the raised regularization: the raised factor did not serve either
     * (db-joint-soerensen, SOCP: refused at iterations 32, 33, 38, 43, 48, every lower level failing again, the dual residual
     * stalled at 2.7e-7, TIME LIMIT; with double-double OPTIMAL). Taken when the estimated double-double factorization
     * (cflops / 3e8 s) costs at most what the solve has spent so far: a solve long enough to have reached this point
     * affords it, a short one that asks for it early (socp_huber_200kx5000, SOCP: 4.1e10 flops, about 140 s) does not */
    static int ddagain = -1; if (ddagain < 0) { const char *e_ = getenv("BRISK_CONEDDAGAIN"); ddagain = e_ ? atoi(e_) : 0; }
    const int dd_again = ddagain && S->dd_refused >= 1 && S->cflops / 3e8 <= sc_time() - S->t_begin;
    const int dd_cheap = S->cflops <= ddmax || (S->dd_force && S->cflops <= 4.0 * ddmax) || S->dd_stall || dd_again;
    if (S->dd_on && !S->use_dd && S->want_dd && S->sc && !dd_cheap && !S->dd_polish) {
        /* too large for double-double: a larger regularization of the factored matrix instead
         * (GMRES removes it: the system it solves keeps the base values) */
        S->want_dd = 0; S->dd_refused++;
        if (S->bump < 1e4) {
            S->bump *= 100.0; S->nbump++; bump_raised(S);
            S->sreg = S->sreg0 * S->bump; S->sregx = S->sreg0 * S->bump; S->sregf = fmax(1e-6, 1e-8 * S->bump);
            if (dbg_on()) socp_printf("      [double-double wanted but too large: the factored matrix's regularization raised to %.1e]\n", S->sregx);
            kkt_fill(S);
            f = kkt_factor(S);
        }
    }
    if (S->dd_on && !S->use_dd && S->want_dd) {
        if (dbg_on()) socp_printf("      [double-double from here: flops %.3g, the cap %.3g, forced %d, the shortfall rule %d, asked again %d (refused %d), polish %d, supernodal %d]\n", S->cflops, ddmax, S->dd_force, S->dd_stall, dd_again, S->dd_refused, S->dd_polish, S->sc != NULL);
        /* (14 November 2026, the SOCP developer: rows 1e-14 and cones 1e-17, from 1e-12 and 1e-14) the regularization at double-double's
         * rounding level relative to the entries it meets: late in a solve the linear and cone blocks' smallest entries fall to about
         * 1e-13 (db-joint-soerensen, SOCP: z/x 2.5e-13), of which 1e-14 is 4 %, so the factor was one of another matrix and the
         * refinement took 20-50 steps an iteration; with 1e-17 3-10, db-joint-soerensen 166 -> 119 s, the 72 CBLIB SOCPs 0.953, the 39
         * development SOCPs 0.986, no status lost, sched_50_50_orig (SOCP) OPTIMAL again; db-plate-yield-line-fox (SOCP) 13.3 -> 16.2 s
         * (two more closing steps on the equations). BRISK_CONEDDREG=rows,cones overrides */
        S->use_dd = 1; S->sreg = 1e-14; S->sregx = 1e-17;
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
static inline int lrot_on(const Socp *S, int k); static void lrot_cones(const Socp *S, int back, const double *in, double *out); static void lrot_blocks(const Socp *S, int add, const double *o, const double *t, double *out);
static void kkt_prec(Socp *S, const double *r, double *v) { const double t0 = sc_time();
    if (S->rspL) {     /* (the large cones in their eigenbasis, BRISK_CONEROTSPACE=2: the factor holds them in ordinary coordinates) */
        const int n = S->n, m = S->m; double *ri = S->lrb;
        memcpy(ri, r, sizeof(double) * ((size_t)n + m));
        lrot_cones(S, 1, r, ri);
        kkt_prec_(S, ri, v);
        double *o = S->lrb + n + m; lrot_cones(S, 0, v, o);
        lrot_blocks(S, 0, o, NULL, v);
    } else kkt_prec_(S, r, v);
    S->t_prec += sc_time() - t0; }   /* (t_prec: the whole application, the solve's permutations and rotations included) */
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
        if (S->anyrot && !S->rsp) for (int k = 0; k < S->nq; k++) if (S->rot[k]) {
            const int d = S->q[k], s = S->qs[k]; const double *Q = S->Qr + 16 * (size_t)k;
            for (int i = 0; i < d; i++) { double a = 0; for (int j = 0; j < d; j++) a += Q[j + i * d] * r[s + j]; u[s + i] = a; }
        }
        { const double ts = sc_time();
          if (!(cone_part() && schol_solve_part(S->sc, u, S->swork, cone_part_nt()) == 0)) { schol_set_solve_seq(cone_seqsolve()); schol_set_solve_nt(g_cone_nt); schol_solve(S->sc, u, S->swork); schol_set_solve_seq(0); }
          S->t_solve += sc_time() - ts; S->nback++; }
        memcpy(v, u, sizeof(double) * (size_t)n);
        memcpy(v + n, u + n + naux, sizeof(double) * (size_t)m);
        if (S->anyrot && !S->rsp) for (int k = 0; k < S->nq; k++) if (S->rot[k]) {
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
    if (S->anyrot && !S->rsp) for (int k = 0; k < S->nq; k++) if (S->rot[k]) {
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
    if (S->anyrot && !S->rsp) for (int k = 0; k < S->nq; k++) if (S->rot[k]) {
        const int d = S->q[k], s = S->qs[k]; const double *Q = S->Qr + 16 * (size_t)k;
        double xt[4];
        for (int i = 0; i < d; i++) xt[i] = t[F->iperm[s + i]];
        for (int j = 0; j < d; j++) { double a = 0; for (int i = 0; i < d; i++) a += Q[j + i * d] * xt[i]; v[s + j] = a; }
    }
}
/* w = K v for the matrix without regularization, K = [ -H A' ; A 0 ] */
/* (11 November 2026, the SOCP developer, a test: BRISK_CONEROTSPACE=2) the large cones in the eigenbasis of their scaling too, by an
 * orthogonal map applied in O(d): with w0^2 - |w1|^2 = 1 and u = w1 / |w1|, the coordinates (v0 + u'v1) / sqrt 2 and (v0 - u'v1) / sqrt 2
 * along (1, u) and (1, -u), where H = eta^2 (2 w w' - J) has the eigenvalues eta^2 lp and eta^2 / lp (lp = (w0 + |w1|)^2), and the
 * rest, the Householder reflection R (R u = -+e1) of v1 without its first entry, where H is eta^2. A direction held in ordinary
 * coordinates knows its part along the large eigenvalue only to the double precision times its size, and the product H v then
 * carries an error of that times the eigenvalue (heterogeneous cones in the dual form, SOCP, A05_00: 2e12, the cone equation's
 * residual 1e-4 against terms of 94, whatever GMRES does; ISSUES 852); in these coordinates H is diagonal and each part is held at
 * its own size. lrot: 0 ordinary -> eigenbasis, 1 back. */
static inline int lrot_cand(const Socp *S, int k) { return S->q[k] > SOCP_DENSE_MAX && !(S->rot && S->rot[k]); }
/* (a cone is held in its eigenbasis only when its eigenvalue ratio lp is at least 1e4: the ordinary product's rounding is about the double
 * precision times lp, below 1e-12 of the product otherwise, the solves' finest target; selected once a solve, lrot_norms) */
static inline int lrot_on(const Socp *S, int k) { return S->rspL && lrot_cand(S, k) && S->lrsel && S->lrsel[k]; }
static void lrot(const Socp *S, int k, int back, const double *v, double *o) {
    const int d = S->q[k]; const double *w = S->wbar + S->qs[k];
    const double nb = S->lrnb[k];         /* (|w1|, once a solve: lrot_norms) */
    if (!(nb > 0.0) || d < 3) { for (int i = 0; i < d; i++) o[i] = v[i]; return; }
    static int lbig = -1; if (lbig < 0) { const char *e_ = getenv("BRISK_LROTBIG"); lbig = e_ ? atoi(e_) : 1; }
    const int big = lbig && d >= DOTC_MIN;         /* (a huge cone: its passes in 64 fixed chunks on the threads, the same bits at any count) */
    const double r2 = sqrt(0.5), u1 = w[1] / nb, sg = u1 >= 0 ? 1.0 : -1.0, hh = 2.0 + 2.0 * fabs(u1), inb = 1.0 / nb;
    if (!back) {
        const double be = (big ? dotc(S, d - 1, w + 1, v + 1) : dotv(d - 1, w + 1, v + 1)) / nb, hv = be + sg * v[1];   /* h'v1, h = u + sg e1 */
        const double f = 2.0 * hv / hh * inb;
        if (big) {
            #pragma omp parallel for schedule(static) num_threads(g_cone_nt) if (cone_vpar(S))
            for (int i = 2; i < d; i++) o[i] = v[i] - f * w[i];
        } else for (int i = 2; i < d; i++) o[i] = v[i] - f * w[i];     /* (no parallel region per small cone: its cost is per call) */
        o[0] = r2 * (v[0] + be); o[1] = r2 * (v[0] - be);
    } else {
        const double ht = (big ? dotc(S, d - 2, w + 2, v + 2) : dotv(d - 2, w + 2, v + 2)) / nb;      /* h't, t = (0, v2, ..) */
        const double f = 2.0 * ht / hh, a = v[0], b = v[1], c = r2 * (a - b), g = (c - f) * inb;
        o[0] = r2 * (a + b);
        o[1] = c * u1 - f * (u1 + sg);
        if (big) {
            #pragma omp parallel for schedule(static) num_threads(g_cone_nt) if (cone_vpar(S))
            for (int i = 2; i < d; i++) o[i] = v[i] + g * w[i];
        } else for (int i = 2; i < d; i++) o[i] = v[i] + g * w[i];
    }
}
/* the norms |w1| of the large cones' scaling points, once a solve (the scaling does not change within it) */
static int lrot_norms(Socp *S) {
    if (!S->lrnb) S->lrnb = (double *)calloc((size_t)S->nq + 1, sizeof(double));
    if (!S->lrsel) S->lrsel = (unsigned char *)calloc((size_t)S->nq + 1, 1);
    static double lpmin = -1; if (lpmin < 0) { const char *e_ = getenv("BRISK_CONEROTLP"); lpmin = e_ ? atof(e_) : 1e4; }
    #pragma omp parallel for schedule(dynamic, 64) num_threads(g_cone_nt) if (cone_vpar(S))
    for (int k = 0; k < S->nq; k++) if (lrot_cand(S, k) && S->q[k] < DOTC_MIN) { const double *w = S->wbar + S->qs[k]; const double nb = sqrt(dotv(S->q[k] - 1, w + 1, w + 1)); S->lrnb[k] = nb; S->lrsel[k] = (w[0] + nb) * (w[0] + nb) >= lpmin; }
    int any = 0;
    for (int k = 0; k < S->nq; k++) if (lrot_cand(S, k)) {
        if (S->q[k] >= DOTC_MIN) { const double *w = S->wbar + S->qs[k]; const double nb = sqrt(dotc(S, S->q[k] - 1, w + 1, w + 1)); S->lrnb[k] = nb; S->lrsel[k] = (w[0] + nb) * (w[0] + nb) >= lpmin; }
        any |= S->lrsel[k];
    }
    return any;
}
/* every large cone's block of in mapped into out (the same offsets; out != in): the cones below DOTC_MIN on the threads, the huge
 * ones one after the other, each on the threads by chunks */
static void lrot_cones(const Socp *S, int back, const double *in, double *out) {
    #pragma omp parallel for schedule(dynamic, 64) num_threads(g_cone_nt) if (cone_vpar(S))
    for (int k = 0; k < S->nq; k++) if (lrot_on(S, k) && S->q[k] < DOTC_MIN) lrot(S, k, back, in + S->qs[k], out + S->qs[k]);
    for (int k = 0; k < S->nq; k++) if (lrot_on(S, k) && S->q[k] >= DOTC_MIN) lrot(S, k, back, in + S->qs[k], out + S->qs[k]);
}
/* the large cones' blocks of out set from o (add 0: out = o) or increased by o - t (add 1), the small ones together on the threads, a
 * huge one by its entries */
static void lrot_blocks(const Socp *S, int add, const double *o, const double *t, double *out) {
    #pragma omp parallel for schedule(dynamic, 64) num_threads(g_cone_nt) if (cone_vpar(S))
    for (int k = 0; k < S->nq; k++) if (lrot_on(S, k) && S->q[k] < DOTC_MIN) {
        const int d = S->q[k], s = S->qs[k];
        if (add) for (int i = 0; i < d; i++) out[s + i] += o[s + i] - t[s + i]; else memcpy(out + s, o + s, sizeof(double) * (size_t)d);
    }
    for (int k = 0; k < S->nq; k++) if (lrot_on(S, k) && S->q[k] >= DOTC_MIN) {
        const int d = S->q[k], s = S->qs[k];
        #pragma omp parallel for schedule(static) num_threads(g_cone_nt) if (cone_vpar(S))
        for (int i = 0; i < d; i++) out[s + i] = add ? out[s + i] + (o[s + i] - t[s + i]) : o[s + i];
    }
}

static void kkt_mul(Socp *S, const double *v, double *w) {
    const int n = S->n, m = S->m;
    const double tm0 = sc_time();
    if (S->rsp || S->rspL) {      /* (S->rsp: the small cones in their eigenbasis, BRISK_CONEROTSPACE=1; S->rspL: the large ones, =2) */
        /* (4 November 2026, the SOCP developer, a test: BRISK_CONEROTSPACE=1) the product in the eigenbasis of the small cones'
         * scaling, where the factor works: the cone's block of H is the diagonal eta^2 (lp, 1/lp, 1, ..), lp = (w0 + |w1|)^2, and
         * the rows see A Q. Formed in the original coordinates, H v = eta^2 (2 (w'v) w - J v) carries a rounding of about the
         * double precision times the largest eigenvalue in every direction, which near the end (eigenvalues 1e9 and 1e-9:
         * db-joint-soerensen, SOCP) is far above the small eigenvalue's part: GMRES then refines towards another matrix */
        double *vx = S->rspb + n, *t = S->rspb + 2 * (size_t)n;
        memcpy(vx, v, sizeof(double) * (size_t)n);
        if (S->rsp && S->rot) for (int k = 0; k < S->nq; k++) if (S->rot[k]) {
            const int d = S->q[k], s = S->qs[k]; const double *Q = S->Qr + 16 * (size_t)k;
            for (int j = 0; j < d; j++) { double a = 0; for (int i = 0; i < d; i++) a += Q[j + i * d] * v[s + i]; vx[s + j] = a; }
        }
        if (S->rspL) lrot_cones(S, 1, v, vx);
        H_apply(S, vx, w);
        if (S->rspL) {
            #pragma omp parallel for schedule(dynamic, 64) num_threads(g_cone_nt) if (cone_vpar(S))
            for (int k = 0; k < S->nq; k++) if (lrot_on(S, k) && S->q[k] < DOTC_MIN) {
                const int d = S->q[k], s = S->qs[k]; const double *wb = S->wbar + s; const double e2 = S->eta[k] * S->eta[k];
                const double lp = (wb[0] + S->lrnb[k]) * (wb[0] + S->lrnb[k]);
                w[s] = e2 * lp * v[s]; w[s + 1] = e2 / lp * v[s + 1];
                for (int i = 2; i < d; i++) w[s + i] = e2 * v[s + i];
            }
            for (int k = 0; k < S->nq; k++) if (lrot_on(S, k) && S->q[k] >= DOTC_MIN) {     /* (a huge cone: its entries on the threads) */
                const int d = S->q[k], s = S->qs[k]; const double *wb = S->wbar + s; const double e2 = S->eta[k] * S->eta[k];
                const double lp = (wb[0] + S->lrnb[k]) * (wb[0] + S->lrnb[k]);
                w[s] = e2 * lp * v[s]; w[s + 1] = e2 / lp * v[s + 1];
                #pragma omp parallel for schedule(static) num_threads(g_cone_nt) if (cone_vpar(S))
                for (int i = 2; i < d; i++) w[s + i] = e2 * v[s + i];
            }
        }
        if (S->rsp && S->rot) for (int k = 0; k < S->nq; k++) if (S->rot[k]) {
            const int d = S->q[k], s = S->qs[k]; const double *wb = S->wbar + s; const double e2 = S->eta[k] * S->eta[k];
            double nq = 0.0; for (int i = 1; i < d; i++) nq += wb[i] * wb[i];
            const double lp = (wb[0] + sqrt(nq)) * (wb[0] + sqrt(nq));
            for (int i = 0; i < d; i++) w[s + i] = e2 * (i == 0 ? lp : i == 1 ? 1.0 / lp : 1.0) * v[s + i];
        }
        for (int i = 0; i < n; i++) w[i] = -w[i];
        { static double fprox = -1; if (fprox < 0) { const char *e_ = getenv("BRISK_CONEFREEPROX"); fprox = e_ ? atof(e_) : 0.0; }
          if (fprox > 0 && !S->use_dd) { const double dp = fprox * S->sregf; for (int i = 0; i < S->nf; i++) w[i] -= dp * v[i]; } }
        if (S->fpr && !S->use_dd) for (int i = 0; i < S->nf; i++) w[i] -= S->fpr[i] * v[i];
        memset(t, 0, sizeof(double) * (size_t)n);
        At_mul(S, 1.0, v + n, t);
        for (int i = 0; i < n; i++) w[i] += t[i];
        if (S->rsp && S->rot) for (int k = 0; k < S->nq; k++) if (S->rot[k]) {
            const int d = S->q[k], s = S->qs[k]; const double *Q = S->Qr + 16 * (size_t)k;
            for (int i = 0; i < d; i++) { double a = 0; for (int j = 0; j < d; j++) a += Q[j + i * d] * t[s + j]; w[s + i] += a - t[s + i]; }
        }
        if (S->rspL) {
            double *o = S->lrb + n + m; lrot_cones(S, 0, t, o);
            lrot_blocks(S, 1, o, t, w);
        }
        const double dl = S->use_dd ? S->sreg : (S->ne && S->ne_ok && S->ne_dl > 0 ? fmin(S->sreg0, S->ne_dl) : S->sreg0);
    /* (4 November 2026, the SOCP developer, a test: BRISK_CONENEDLX=v) the rows' term of the system solved on the normal equations'
     * route set to v (0: the system without it): the rows' equation solved is A dx + dl dy = r, so a step leaves a primal
     * residual of dl |dy|, which near-dependent rows make large (db-joint-soerensen, SOCP, the normal equations kept to the
     * end: entries of dy up to 360 and more, the primal residual rising from 1.2e-8 to 2e-7) */
    static double dlx_ = -2; if (dlx_ < -1) { const char *e_ = getenv("BRISK_CONENEDLX"); dlx_ = e_ ? atof(e_) : -1.0; }
    const double dlu = dlx_ >= 0 && !S->use_dd && S->ne && S->ne_ok ? dlx_ : (S->dl_exact && !S->use_dd && S->ne && S->ne_ok ? 0.0 : dl);
        for (int i = 0; i < m; i++) w[n + i] = dlu * v[n + i];
        if (S->ne && S->ne_ok && S->ne_ndr > 0 && !S->use_dd) { static double dpe = -1; if (dpe < 0) { const char *e_ = getenv("BRISK_CONETINYDEPW"); dpe = e_ ? atof(e_) : 1.0; }
            for (int a = 0; a < S->ne_ndr; a++) w[n + S->ne_dprow[a]] += dpe * S->ne_dpw[a] * v[n + S->ne_dprow[a]]; }
        A_mul(S, 1.0, vx, w + n);
        S->t_mul += sc_time() - tm0;
        return;
    }
    H_apply(S, v, w);
    const int par = cone_vpar(S);           /* (entry by entry: the same bits on the threads) */
    #pragma omp parallel for schedule(static) num_threads(g_cone_nt) if (par)
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
    /* (4 November 2026, the SOCP developer, a test: BRISK_CONENEDLX=v) the rows' term of the system solved on the normal equations'
     * route set to v (0: the system without it): the rows' equation solved is A dx + dl dy = r, so a step leaves a primal
     * residual of dl |dy|, which near-dependent rows make large (db-joint-soerensen, SOCP, the normal equations kept to the
     * end: entries of dy up to 360 and more, the primal residual rising from 1.2e-8 to 2e-7) */
    static double dlx_ = -2; if (dlx_ < -1) { const char *e_ = getenv("BRISK_CONENEDLX"); dlx_ = e_ ? atof(e_) : -1.0; }
    const double dlu = dlx_ >= 0 && !S->use_dd && S->ne && S->ne_ok ? dlx_ : (S->dl_exact && !S->use_dd && S->ne && S->ne_ok ? 0.0 : dl);
    #pragma omp parallel for schedule(static) num_threads(g_cone_nt) if (par)
    for (int i = 0; i < m; i++) w[n + i] = dlu * v[n + i];
    if (S->ne && S->ne_ok && S->ne_ndr > 0 && !S->use_dd) { static double dpe = -1; if (dpe < 0) { const char *e_ = getenv("BRISK_CONETINYDEPW"); dpe = e_ ? atof(e_) : 1.0; }   /* (a test: the share of the raise kept in the solved system) */
        for (int a = 0; a < S->ne_ndr; a++) w[n + S->ne_dprow[a]] += dpe * S->ne_dpw[a] * v[n + S->ne_dprow[a]]; }    /* (the dependent rows' term, ne_factor) */
    if (S->nd > 0) { B_mul(S, 1.0, v, w + n); dn_mul(S, 1.0, v, w + n, v + n, w); }     /* (the dense block read once for both) */
    else A_mul(S, 1.0, v, w + n);
    S->t_mul += sc_time() - tm0;
}
/* solves [ -H A' ; A 0 ] [sx; sy] = [rx; ry]: the factorization of the regularized matrix is the
 * preconditioner of GMRES on the matrix itself (at most maxref steps; 0: the preconditioner alone).
 * One step is a step of iterative refinement with the best step length; a few steps remove the
 * directions in which the regularization is not small against the matrix. */
/* (10 November 2026, the SOCP developer; PLAN item 2) the solves counted by purpose for the debug log (BRISK_CONEDBG): 0 other (the
 * start, the polishing), 1 the tau direction p = K^-1 [c; b], 2 the predictor, 3 the corrector, 4 the extra correctors, 5 the
 * refinements against the dual or the primal equation, 6 q refined against its rows; for each the solves and the applications of the
 * factor (the preconditioned solve and each GMRES step). Plain counters the solve passes anyway: no cost to the method */
static _Thread_local int g_tag = 0;
static _Thread_local long g_tag_solves[7], g_tag_apps[7];
#define SOLVE_TAG(t) (g_tag = (t))
static void kkt_solve_(Socp *S, const double *rx, const double *ry, double *sx, double *sy, int maxref);
static double g_t_cstep = 0, g_t_ks = 0;      /* (12 November 2026, the debug log: the time in the steps to the boundary and in the linear solves) */
static void kkt_solve_w(Socp *S, const double *rx, const double *ry, double *sx, double *sy, int maxref);
static void kkt_solve(Socp *S, const double *rx, const double *ry, double *sx, double *sy, int maxref) { const double t0 = sc_time(); kkt_solve_w(S, rx, ry, sx, sy, maxref); g_t_ks += sc_time() - t0; }
static void kkt_solve_w(Socp *S, const double *rx, const double *ry, double *sx, double *sy, int maxref) {
    static int rsp = -1; if (rsp < 0) { const char *e_ = getenv("BRISK_CONEROTSPACE"); rsp = e_ ? atoi(e_) : 2; }   /* (12 November 2026: 2, the large cones in their eigenbasis, the default; 0 off) */
    int anyl = 0; if (rsp & 2) for (int k = 0; k < S->nq; k++) if (lrot_cand(S, k)) { anyl = 1; break; }
    if (anyl) { S->rspL = 1; anyl = lrot_norms(S); S->rspL = 0; }     /* (only the cones whose eigenvalue ratio needs it) */
    /* (11 November 2026: the small cones' eigenbasis only when asked for, BRISK_CONEROTSPACE=1 or 3; =2 the large ones alone: with any value
     * the small cones the solver had rotated were taken into the eigenbasis too, an untested mode that cost socp_socopf_3012wp_k, SOCP,
     * its OPTIMAL in 3 of 4 row orders) */
    const int small = (rsp & 1) && S->anyrot;
    if (!((small || anyl) && (!(S->ne && S->ne_ok) || !small) && S->nd == 0)) {     /* (the large cones alone also on the normal equations: their factor stays in ordinary coordinates) */ kkt_solve_(S, rx, ry, sx, sy, maxref); return; }
    const int n = S->n;
    if (!S->rspb) S->rspb = (double *)calloc(3 * (size_t)n, sizeof(double));
    if (anyl && !S->lrb) S->lrb = (double *)calloc(2 * ((size_t)n + S->m) + 1, sizeof(double));
    double *rr = S->rspb;      /* (the right-hand side in the eigenbasis: Q' on each rotated cone) */
    memcpy(rr, rx, sizeof(double) * (size_t)n);
    S->rspL = anyl;
    if (anyl) lrot_cones(S, 0, rx, rr);
    if (small) for (int k = 0; k < S->nq; k++) if (S->rot[k]) {
        const int d = S->q[k], s = S->qs[k]; const double *Q = S->Qr + 16 * (size_t)k;
        for (int i = 0; i < d; i++) { double a = 0; for (int j = 0; j < d; j++) a += Q[j + i * d] * rx[s + j]; rr[s + i] = a; }
    }
    S->rsp = small;
    kkt_solve_(S, rr, ry, sx, sy, maxref);     /* (rr is not touched by the product, which works in rspb's other 2 n entries) */
    S->rsp = 0;
    if (anyl) {
        double *o = S->lrb; lrot_cones(S, 1, sx, o);
        lrot_blocks(S, 0, o, NULL, sx); }
    S->rspL = 0;
    if (small) for (int k = 0; k < S->nq; k++) if (S->rot[k]) {     /* (the solution back: Q on each rotated cone) */
        const int d = S->q[k], s = S->qs[k]; const double *Q = S->Qr + 16 * (size_t)k; double xt[4];
        for (int i = 0; i < d; i++) xt[i] = sx[s + i];
        for (int j = 0; j < d; j++) { double a = 0; for (int i = 0; i < d; i++) a += Q[j + i * d] * xt[i]; sx[s + j] = a; }
    }
}
static void kkt_solve_(Socp *S, const double *rx, const double *ry, double *sx, double *sy, int maxref) {
    cone_threads_adapt(S->verbose);         /* (the busy-machine rule, 28 October 2026) */
    const int n = S->n, m = S->m, N = n + m;
    double *e = S->sk, *s = S->gs;
    S->solves++;
    const long nb_tag0 = S->nback; const int tag_ = g_tag; g_tag_solves[tag_]++;
    memcpy(e, rx, sizeof(double) * (size_t)n); memcpy(e + n, ry, sizeof(double) * (size_t)m);
    /* (3 November 2026, the SOCP developer, a test: BRISK_CONEGMWF=omega) GMRES minimizes the residual with the free
     * variables' rows weighted by omega: the error of a direction in those rows is the error of the next iterate's dual
     * residual there (no dz absorbs it), while the 2-norm of the whole residual is dominated by the other rows */
    static double gwf = -1; if (gwf < 0) { const char *e_ = getenv("BRISK_CONEGMWF"); gwf = e_ ? atof(e_) : 1.0; }
    const double wf = S->use_dd ? 1.0 : gwf;
    double bn = sqrt_dotc_(S, N, e);
    if (wf != 1.0 && S->nf > 0) { double t_ = 0; for (int i = 0; i < S->nf; i++) t_ += e[i] * e[i]; bn = sqrt(fmax(0.0, bn * bn + (wf * wf - 1.0) * t_)); }
    /* (12 November 2026, the SOCP developer, a test: BRISK_CONEGMMEAS=1) GMRES minimizes the residual in the measure of the problem as
     * given: each dual entry divided by its column's scale Ec, each row by its row's scale Dr (the errors the solver stops on), the
     * Krylov vectors unweighted before the factor is applied. On rows scaled 1e-5 .. 1e5 (multiscale rows A02_00, SOCP) a residual of
     * 1e-7 of the scaled right-hand side, whose norm the dual block sets, is 1e-3 of |b| on the rows as given: the first step from a
     * primal feasible start left a primal error of 1.1e-3 */
    static int gmm = -1; if (gmm < 0) { const char *e_ = getenv("BRISK_CONEGMMEAS"); gmm = e_ ? atoi(e_) : 0; }
    static _Thread_local double *gmw = NULL, *gmt = NULL; static _Thread_local long gmwN = 0;
    const int gm_on = gmm && wf == 1.0 && S->Ec && S->Dr;
    if (gm_on) {
        if (gmwN < N) { free(gmw); free(gmt); gmw = (double *)malloc(sizeof(double) * (size_t)N); gmt = (double *)malloc(sizeof(double) * (size_t)N); gmwN = N; }
        for (int i = 0; i < n; i++) gmw[i] = 1.0 / S->Ec[i];
        for (int i = 0; i < m; i++) gmw[n + i] = 1.0 / S->Dr[i];
        double t_ = 0; for (int i = 0; i < N; i++) t_ += (gmw[i] * e[i]) * (gmw[i] * e[i]); bn = sqrt(t_);
    }
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
        #pragma omp parallel for schedule(static) num_threads(g_cone_nt) if (N >= DOTC_MIN && cone_vpar(S))
        for (int i = 0; i < n; i++) w[i] = rx[i] - w[i];
        for (int i = 0; i < m; i++) w[n + i] = ry[i] - w[n + i];
        if (wf != 1.0) for (int i = 0; i < S->nf; i++) w[i] *= wf;
        double beta = sqrt_dotc_(S, N, w);
        S->last_res = beta / bn;
        /* (26 October 2026, a test: BRISK_CONEREFMU=4) the preconditioned solve kept without refinement when it already meets
         * the step's needs, as the QP method's targets: its residual's dual block below a tenth of the dual residual, its
         * primal block below a tenth of the primal one, and its effect on the gap sum |x_j r_j| + sum |y_i r_i| below a
         * tenth of the complementarity (largest entries; at least 1e-2 of the tolerance) */
        if (S->tg_d > 0) {
            double ed = 0, ep = 0, eg = 0;
            /* (11 November 2026, the SOCP developer) the test reads the residual entry by entry against x and the columns' scaling: with
             * the large cones in their eigenbasis (BRISK_CONEROTSPACE=2) their blocks are mapped back to ordinary coordinates first;
             * read as they were, the rotated entries were weighed by unrelated entries of x (sched_100_100_orig, SOCP: 26 -> 53
             * iterations, reduced accuracy) */
            const double *wo = w;
            if (S->rspL) { double *u = S->lrb + n + m; memcpy(u, w, sizeof(double) * (size_t)n); lrot_cones(S, 1, w, u); wo = u; }
            if (S->tg_w) {       /* (5: in the measure of the problem as given, the 2-norms of the residuals unscaled) */
                for (int i = 0; i < n; i++) { const double a = fabs(wo[i]), u = a / S->Ec[i]; ed += u * u; eg += fabs(S->x[i]) * a; }
                for (int i = 0; i < m; i++) { const double a = fabs(w[n + i]), u = a / S->Dr[i]; ep += u * u; eg += fabs(S->y[i]) * a; }
                ed = sqrt(ed); ep = sqrt(ep);
            } else {
            for (int i = 0; i < n; i++) { const double a = fabs(wo[i]); if (a > ed) ed = a; eg += fabs(S->x[i]) * a; }
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
            for (int j = 0; j <= k; j++) c[j] = dotc(S, N, V + (size_t)j * N, w);
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
                for (int j = 0; j < k; j++) { const double a = yv[j]; const double *zj = Z + (size_t)j * N; axpyc(S, N, a, zj, s); }
                for (int j = 0; j <= k; j++) { const double a = t[j]; const double *vj = V + (size_t)j * N; axpyc(S, N, -a, vj, w); }
                const double b2 = sqrt_dotc_(S, N, w);
                if (b2 < 0.9 * beta) {       /* (kept only when it gains: a least-squares combination from a stalled space is noise) */
                    S->rc_used++; S->rc_gain += log(fmax(beta / fmax(b2, 1e-300), 1e-300));
                    if (gmdbg) socp_printf("        [recycled %d vectors: residual %.2e -> %.2e]\n", k, beta / bn, b2 / bn);
                    beta = b2; S->last_res = beta / bn;
                } else {
                    for (int j = 0; j < k; j++) { const double a = yv[j]; const double *zj = Z + (size_t)j * N; axpyc(S, N, -a, zj, s); }
                    for (int j = 0; j <= k; j++) { const double a = t[j]; const double *vj = V + (size_t)j * N; axpyc(S, N, a, vj, w); }
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
        if (gm_on && beta > 0) {      /* (BRISK_CONEGMMEAS: the residual in the measure of the problem as given) */
            for (int i = 0; i < N; i++) w[i] *= gmw[i];
            beta = sqrt_dotc_(S, N, w); S->last_res = beta / bn;
        }
        if (beta > tol) {
            {
            #pragma omp parallel for schedule(static) num_threads(g_cone_nt) if (N >= DOTC_MIN && cone_vpar(S))
            for (int i = 0; i < N; i++) V[i] = w[i] / beta; }
            g[0] = beta;
            double rhist[64]; rhist[0] = beta;      /* (the residual norm after each step, for the stagnation test) */
            int k = 0;
            for (; k < kmax; k++) {
                double *zk = Z + (size_t)k * N, *vk1 = V + (size_t)(k + 1) * N;
                if (gm_on) { const double *vk = V + (size_t)k * N; for (int i = 0; i < N; i++) gmt[i] = vk[i] / gmw[i]; kkt_prec(S, gmt, zk); }
                else kkt_prec(S, V + (size_t)k * N, zk);
                kkt_mul(S, zk, vk1);
                if (wf != 1.0) for (int i = 0; i < S->nf; i++) vk1[i] *= wf;
                if (gm_on) for (int i = 0; i < N; i++) vk1[i] *= gmw[i];
                S->refine_total++;
                double *hk = Hm + (size_t)k * (kmax + 1);
                for (int j = 0; j <= k; j++) {
                    const double *vj = V + (size_t)j * N;
                    const double h = dotc(S, N, vj, vk1);
                    hk[j] = h;
                    axpyc(S, N, -h, vj, vk1);
                }
                const double hn = sqrt_dotc_(S, N, vk1);
                hk[k + 1] = hn;
                if (hn > 0) {
                    #pragma omp parallel for schedule(static) num_threads(g_cone_nt) if (N >= DOTC_MIN && cone_vpar(S))
                    for (int i = 0; i < N; i++) vk1[i] /= hn; }
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
                  /* (4 November 2026, the SOCP developer, a test: BRISK_CONEGMBACK=1) after the return to the normal equations
                   * (BRISK_CONENEBACK) the stop also where the preconditioned solve's own residual is above stagres: there every
                   * late solve starts at 1e-3 .. 1e-1 and GMRES contracts by 0.55 .. 0.8 a step to its cap (db-joint-soerensen, SOCP) */
                  static int gmback_c = -2; const int gmback = cone_back("BRISK_CONEGMBACK", &gmback_c);
                  static double stagres = -1; if (stagres < 0) { const char *e_ = getenv("BRISK_CONEGMSTAGRES"); stagres = e_ ? atof(e_) : 1e-6; }   /* (sched_100_50_orig: a solve that starts at 1e4 and gains ten orders in seven steps is still going; stopped it costs 31 -> 59 iterations) */
                  static double stagfl = -1; if (stagfl < 0) { const char *e_ = getenv("BRISK_CONEGMSTAGFLOOR"); stagfl = e_ ? atof(e_) : 1e-10; }   /* (no stop below this residual: a rounding plateau at 3e-12 made sched_100_100_orig miss its last iteration) */
                  /* (and only while GMRES has not shown that it converges: its total gain since the start below 1/stagtot; a solve that
                   *  started from a residual of order 1e4 after replaced pivots and gained ten orders in seven steps is still going) */
                  static double stagtot = -1; if (stagtot < 0) { const char *e_ = getenv("BRISK_CONEGMSTAGTOT"); stagtot = e_ ? atof(e_) : 0; }   /* (a test, off: the gate above does this job) */
                  if (stag > 0 && k + 1 >= stagmin && k >= 1 && k + 1 < 64 && (S->last_res <= stagres || (S->ne_back && gmback)) && rhist[k + 1] > stagfl * bn && rhist[k + 1] > stagtot * rhist[0] && rhist[k + 1] > stag * stag * rhist[k - 1]) { if (gmdbg) socp_printf("        [gmres stopped by stagnation at step %d]\n", k + 1); k++; break; } }
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
                    for (int j = 0; j < k; j++) { const double a = yv[j]; const double *zj = Z + (size_t)j * N; axpyc(S, N, a, zj, s); }
                    S->last_res = fabs(g[k]) / bn;
                }
            }
        }
    }
    if (gmdbg && maxref > 0 && bn > 0) {      /* (debug: the residual of the refined solve by block, from a fresh product) */
        double *w = S->lastx; kkt_mul(S, s, w);
        double rf = 0, rl = 0, rq = 0, ry_ = 0, bf = 0;
        for (int i = 0; i < S->nf; i++) { const double d_ = rx[i] - w[i]; rf += d_ * d_; bf += rx[i] * rx[i]; }
        for (int i = S->nf; i < S->nf + S->nl; i++) { const double d_ = rx[i] - w[i]; rl += d_ * d_; }
        for (int i = S->nf + S->nl; i < n; i++) { const double d_ = rx[i] - w[i]; rq += d_ * d_; }
        for (int i = 0; i < m; i++) { const double d_ = ry[i] - w[n + i]; ry_ += d_ * d_; }
        double sxf = 0, sxo = 0, syi = 0; for (int i = 0; i < n; i++) { if (i < S->nf) sxf = fmax(sxf, fabs(s[i])); else sxo = fmax(sxo, fabs(s[i])); } for (int i = 0; i < m; i++) syi = fmax(syi, fabs(s[n + i]));
        if (getenv("BRISK_CONEDBGTERMS")) {     /* (11 November 2026, debug: the cone equation's two terms, H dx and A'dy, against its residual, in the worst cone) */
            double *hx_ = (double *)calloc((size_t)n + 1, sizeof(double)), *at_ = (double *)calloc((size_t)n + 1, sizeof(double));
            H_apply(S, s, hx_); At_mul(S, 1.0, s + n, at_);
            int kw = -1; double rw = 0; for (int k = 0; k < S->nq; k++) { double t_ = 0; for (int i = 0; i < S->q[k]; i++) { const double d_ = rx[S->qs[k] + i] - w[S->qs[k] + i]; t_ += d_ * d_; } if (t_ > rw) { rw = t_; kw = k; } }
            if (kw >= 0) { const int q0 = S->qs[kw], d0 = S->q[kw]; double nh = 0, na = 0, nr = 0, nx = 0, mh = 0; int im = 0;
                for (int i = 0; i < d0; i++) { nh += hx_[q0 + i] * hx_[q0 + i]; na += at_[q0 + i] * at_[q0 + i]; nr += rx[q0 + i] * rx[q0 + i]; nx += s[q0 + i] * s[q0 + i]; if (fabs(hx_[q0 + i]) > mh) { mh = fabs(hx_[q0 + i]); im = i; } }
                const double *wb = S->wbar + q0; double nb = 0; for (int i = 1; i < d0; i++) nb += wb[i] * wb[i];
                socp_printf("        [cone %d (dimension %d): |residual| %.2e, |H dx| %.2e (largest %.2e at %d), |A'dy| %.2e, |rhs| %.2e, |dx| %.2e; eta^2 %.2e, lp %.2e]\n", kw, d0, sqrt(rw), sqrt(nh), mh, im, sqrt(na), sqrt(nr), sqrt(nx), S->eta[kw] * S->eta[kw], (wb[0] + sqrt(nb)) * (wb[0] + sqrt(nb))); }
            free(hx_); free(at_);
        }
        socp_printf("        [refined solve residual (absolute): free %.1e linear %.1e cones %.1e rows %.1e; |rhs| %.1e, its free part %.1e; the solution's largest entries: free %.1e, other columns %.1e, rows %.1e]\n", sqrt(rf), sqrt(rl), sqrt(rq), sqrt(ry_), bn, sqrt(bf), sxf, sxo, syi);
    }
    if (maxref > 0 && (S->last_res > 1e-10 || !isfinite(S->last_res)) && S->F.ndyn > 0) S->want_dd = 1;      /* replaced pivots and a solve that did not refine: double-double next */
    if (maxref > 0 && (S->last_res > S->it_maxres || !isfinite(S->last_res))) S->it_maxres = isfinite(S->last_res) ? S->last_res : 1e300;
    g_tag_apps[tag_] += S->nback - nb_tag0; g_tag = 0;
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
        /* (8 November 2026, the SOCP developer, a test: BRISK_CONEDZTARGET=f) refined when the miss exceeds f times the residual the
         * step is meant to leave, sigma max|r_D| (sigma = 1 - etaf), instead of the whole residual etaf max|r_D|: heterogeneous cones
         * (A05_00, SOCP) at iteration 7, a miss below the residual but six times the target, the dual residual falling by 6.2e-2
         * against 1.04e-2, and three iterations lost */
        { static double dzt = -1; if (dzt < 0) { const char *e_ = getenv("BRISK_CONEDZTARGET"); dzt = e_ ? atof(e_) : 0.0; }
          const double thr_ = dzt > 0 ? fmax(dzt * (1.0 - etaf) * rm / fmax(etaf, 1e-300), flo) : fmax(rm, flo);
          if (!(dm > thr_)) break; }
        for (int i = 0; i < m; i++) S->wdy[i] = 0.0;
        SOLVE_TAG(5); kkt_solve(S, wd, S->wdy, S->wdx, S->wdy2, S->refine_max);
        for (int j = 0; j < n; j++) dx[j] += S->wdx[j];
        for (int i = 0; i < m; i++) dy[i] += S->wdy2[i];
        H_apply(S, S->wdx, t2);
        for (int j = S->nf; j < n; j++) dz[j] -= t2[j];
        S->dz_count++; nref++;
    }
    return nref;
}

/* (4 November 2026, the SOCP developer, a test: BRISK_CONEPREFINE=1) the same refinement against the primal equation
 * A dx = eta rP + b dtau: the direction is q + dtau p, where p solves K p = (c, b), a right-hand side of norm about 1e2,
 * and GMRES stops p at its own relative target: near the end its rows' residual stays at 1e-6 .. 1e-4 in absolute terms
 * (db-joint-soerensen, SOCP, the normal equations kept: 2e-6 at iteration 47, 1.3e-4 at 50), far above the primal
 * residual (1e-8), and dtau carries it into the step: the primal residual rose from 2e-8 to 6e-7 in four iterations. The
 * miss is solved for with its own small right-hand side ([0; miss]: the dual equation unchanged) */
static int primal_refine(Socp *S, double tol, double etaf, const double *rP, double dtau, double *dx, double *dy, double *dz, double *t2) {
    static int on = -1; if (on < 0) { const char *e_ = getenv("BRISK_CONEPREFINE"); on = e_ ? atoi(e_) : 0; }
    if (!on) return 0;
    const int n = S->n, m = S->m;
    int nref = 0;
    double bn = 0; for (int i = 0; i < m; i++) bn = fmax(bn, fabs(S->b[i]));
    for (int rf = 0; rf < 2; rf++) {
        const double flo = 0.1 * tol * S->tau * (1.0 + bn);
        double *wp = S->wdy;
        /* (the miss of the system that was solved, whose rows carry the regularization's term dl dy by design: only what the
         * solve left is refined; measured against A dx alone, the refinement fired on healthy problems and changed their
         * path: socp_riskinf_10000x50xnear, SOCP, OPTIMAL in 18 iterations -> reduced accuracy in 52) */
        const double dl = S->use_dd ? S->sreg : (S->ne && S->ne_ok && S->ne_dl > 0 ? fmin(S->sreg0, S->ne_dl) : S->sreg0);
        for (int i = 0; i < m; i++) wp[i] = etaf * rP[i] + dtau * S->b[i] - dl * dy[i];
        A_mul(S, -1.0, dx, wp);
        double pm = 0, rm = 0;
        for (int i = 0; i < m; i++) { pm = fmax(pm, fabs(wp[i])); rm = fmax(rm, fabs(etaf * rP[i])); }
        if (dbg_on()) socp_printf("      [the direction's miss of the primal equation %.1e (its target %.1e, floor %.1e)]\n", pm, rm, flo);
        if (!(pm > fmax(rm, flo))) break;
        double *wz = S->wdz; for (int j = 0; j < n; j++) wz[j] = 0.0;
        SOLVE_TAG(5); kkt_solve(S, wz, wp, S->wdx, S->wdy2, S->refine_max);
        for (int j = 0; j < n; j++) dx[j] += S->wdx[j];
        for (int i = 0; i < m; i++) dy[i] += S->wdy2[i];
        H_apply(S, S->wdx, t2);
        for (int j = S->nf; j < n; j++) dz[j] -= t2[j];
        nref++;
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
static int scu_on(void); static void W_apply(const Socp *S, int k, int inv, const double *in, double *out);
static double cone_step_(const Socp *S, const double *x, const double *dx, double amax);
static double cone_step(const Socp *S, const double *x, const double *dx, double amax) { const double t0 = sc_time(); const double a = cone_step_(S, x, dx, amax); g_t_cstep += sc_time() - t0; return a; }
static double cone_step_(const Socp *S, const double *x, const double *dx, double amax) {
    double al = amax;
    const int par = cone_vpar(S);
    if (scu_on() && S->scu_t && (x == S->x || x == S->z)) {     /* (BRISK_CONESCALEUPD: each cone's step from its scaled point lambda and W dx or W^-1 dz) */
        const int inv = x == S->z; double *t = S->scu_t + (inv ? S->n : 0);
        for (int i = S->nf; i < S->nf + S->nl; i++) if (dx[i] < 0) { const double r = -x[i] / dx[i]; if (r < al) al = r; }
        double ac = amax;
        #pragma omp parallel for schedule(dynamic, 64) num_threads(g_cone_nt) reduction(min:ac) if (par)
        for (int k = 0; k < S->nq; k++) {
            const int d = S->q[k], s0 = S->qs[k];
            if (scu_on() == 2) {     /* (=2: the scaled point only where the cone's own margin is below 1e-4 of x0: elsewhere x0^2 - |x1|^2 formed from x keeps 12 digits) */
                double nn = 0; for (int i = 1; i < d; i++) nn += x[s0 + i] * x[s0 + i];
                if (x[s0] - sqrt(nn) >= 1e-4 * x[s0]) { const double a_ = soc_step(d, x + s0, dx + s0, amax); if (a_ < ac) ac = a_; continue; }
            }
            W_apply(S, k, inv, dx + s0, t + s0); const double a_ = soc_step(d, S->lam + s0, t + s0, amax); if (a_ < ac) ac = a_; }
        return al < ac ? al : ac;
    }
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
static int scu_on(void);
static int nt_update(Socp *S) {
    if (scu_on() && !S->scu_w) { S->scu_w = (double *)malloc(sizeof(double) * ((size_t)S->n + 1)); S->scu_eta = (double *)malloc(sizeof(double) * ((size_t)S->nq + 1));
                                 S->scu_t = (double *)malloc(sizeof(double) * (2 * (size_t)S->n + 1)); S->scu_redo = (unsigned char *)calloc((size_t)S->nq + 1, 1); S->scu_l = (double *)malloc(sizeof(double) * ((size_t)S->n + 1)); }
    for (int i = 0; i < S->nl; i++) {
        const double xi = S->x[S->nf + i], zi = S->z[S->nf + i];
        if (!(xi > 0) || !(zi > 0)) return 1;
        S->h[i] = zi / xi; S->lam[S->nf + i] = sqrt(xi * zi);
    }
    /* (29 October 2026) the cones on the threads (each cone's own entries: the same bits at any thread count); a cone that is
     * not interior is flagged (the caller stops or takes another start: the later cones' scalings are not used) */
    int bad = 0;
    #pragma omp parallel for schedule(static) num_threads(g_cone_nt) if (cone_vpar(S)) reduction(|:bad)
    for (int k = 0; k < S->nq; k++) {
        const int d = S->q[k], s = S->qs[k];
        const double *x = S->x + s, *z = S->z + s; double *w = S->wbar + s;
        int scu_k = S->scu_ok && !S->scu_redo[k];
        if (scu_k && scu_on() == 2) {     /* (=2: only a cone whose x or z lies within 1e-4 of its own size from the boundary) */
            double xn_ = 0, zn_ = 0; for (int i = 1; i < d; i++) { xn_ += x[i] * x[i]; zn_ += z[i] * z[i]; }
            if (x[0] - sqrt(xn_) >= 1e-4 * x[0] && z[0] - sqrt(zn_) >= 1e-4 * z[0]) scu_k = 0;
        }
        if (scu_k) {     /* (the scaling updated in the scaled coordinates at the step: scu_update) */
            const double e0 = S->eta[k]; double *wo = S->scu_t + s;
            memcpy(wo, w, sizeof(double) * (size_t)d);
            memcpy(w, S->scu_w + s, sizeof(double) * (size_t)d); S->eta[k] = S->scu_eta[k];
            if (scu_on() == 3) memcpy(S->lam + s, S->scu_l + s, sizeof(double) * (size_t)d); else W_apply(S, k, 0, x, S->lam + s);
            /* (a check: W x and W^-1 z agree (both are lambda); a cone where they part by more than 1e-3 of lambda, a point changed
             * outside the step or a drift, takes its scaling from x and z as before) */
            double *l2 = S->scu_t + S->n + s, df = 0, ln = 0;
            W_apply(S, k, 1, z, l2);
            for (int i = 0; i < d; i++) { const double e_ = S->lam[s + i] - l2[i]; df += e_ * e_; ln += S->lam[s + i] * S->lam[s + i]; }
            if (df <= 1e-6 * ln) continue;
            memcpy(w, wo, sizeof(double) * (size_t)d); S->eta[k] = e0;
            #pragma omp atomic
            S->scu_nredo++;
        }
        double xn = 0, zn = 0, xz = x[0] * z[0];
        for (int i = 1; i < d; i++) { xn += x[i] * x[i]; zn += z[i] * z[i]; xz += x[i] * z[i]; }
        xn = sqrt(xn); zn = sqrt(zn);
        const double xx = (x[0] - xn) * (x[0] + xn), zz = (z[0] - zn) * (z[0] + zn);
        if (!(xx > 0) || !(zz > 0) || !(x[0] > 0) || !(z[0] > 0)) { bad = 1; continue; }
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
    if (S->scu_ok) { S->scu_ok = 0; memset(S->scu_redo, 0, (size_t)S->nq + 1); }
    return bad;
}
/* (12 November 2026, the SOCP developer, a test: BRISK_CONESCALEUPD=1; ISSUES 865, 818) the cones' scaling after a step updated in
 * the scaled coordinates, never recomputed from x and z. Near the end a cone's x (or z) lies within 1e-12 of its boundary for its own
 * size, below the rounding of x0^2 - |x1|^2 formed from x held in double (friction contacts P05_07, SOCP: a cone of 813,641 entries,
 * x0 - |x1| = -6e-15 of x0 0.67 after a step that its own quadratic put inside), so the scaling built from it loses the small
 * eigenvalue. The scaled points s = lambda + alpha W dx and t = lambda + alpha W^-1 dz (lambda = W x = W^-1 z) stay of the order of
 * sqrt(mu) on every axis: their scaling (w^, eta^) is formed without cancellation, and with W = eta H(w), H(w)^2 = 2 w w' - J,
 * H(w) J H(w) = J, the new point's scaling is W+^2 = W W^^2 W = (eta eta^)^2 (2 u u' - J), u = H(w) w^: w+ = H(w) w^, eta+ = eta eta^
 * (Vandenberghe, "The CVXOPT linear and quadratic cone program solvers", 2010, the scaling's update from the scaled points). The
 * membership of the new point is judged on s and t as well. Returns 1 when the new point is not interior. */
static int scu_on(void) { static int v = -1; if (v < 0) { const char *e_ = getenv("BRISK_CONESCALEUPD"); v = e_ ? atoi(e_) : 0; } return v; }
static int scu_update(Socp *S, double alpha, const double *dx, const double *dz) {
    const int n = S->n;
    if (!S->scu_w) { S->scu_w = (double *)malloc(sizeof(double) * ((size_t)n + 1)); S->scu_eta = (double *)malloc(sizeof(double) * ((size_t)S->nq + 1));
                     S->scu_t = (double *)malloc(sizeof(double) * (2 * (size_t)n + 1)); S->scu_redo = (unsigned char *)calloc((size_t)S->nq + 1, 1); S->scu_l = (double *)malloc(sizeof(double) * ((size_t)n + 1)); }
    for (int i = 0; i < S->nl; i++) if (!(S->x[S->nf + i] > 0) || !(S->z[S->nf + i] > 0)) return 1;
    if (!(S->tau > 0) || !(S->kap > 0)) return 1;
    int bad = 0;
    #pragma omp parallel for schedule(dynamic, 64) num_threads(g_cone_nt) if (cone_vpar(S)) reduction(|:bad)
    for (int k = 0; k < S->nq; k++) {
        const int d = S->q[k], s0 = S->qs[k];
        const double *lam = S->lam + s0, *wb = S->wbar + s0; double *st = S->scu_t + s0, *zt = S->scu_t + n + s0, *u = S->scu_w + s0;
        W_apply(S, k, 0, dx + s0, st); W_apply(S, k, 1, dz + s0, zt);
        double sn = 0, zn = 0, sz = 0;
        for (int i = 0; i < d; i++) { st[i] = lam[i] + alpha * st[i]; zt[i] = lam[i] + alpha * zt[i]; }
        for (int i = 1; i < d; i++) { sn += st[i] * st[i]; zn += zt[i] * zt[i]; sz += st[i] * zt[i]; }
        sz += st[0] * zt[0]; sn = sqrt(sn); zn = sqrt(zn);
        const double ss = (st[0] - sn) * (st[0] + sn), tt = (zt[0] - zn) * (zt[0] + zn);
        if (!(ss > 0) || !(tt > 0) || !(st[0] > 0) || !(zt[0] > 0)) { bad = 1; continue; }
        const double rs = sqrt(ss), rt = sqrt(tt), g2 = 2.0 * sqrt(0.5 * (1.0 + sz / (rs * rt)));
        double qq = 0, q = 0;
        for (int i = 1; i < d; i++) { u[i] = (zt[i] / rt - st[i] / rs) / g2; qq += u[i] * u[i]; q += wb[i] * u[i]; }
        const double h0 = sqrt(1.0 + qq), a = wb[0];      /* (w^: its first entry from det w^ = 1) */
        if (scu_on() == 3) {     /* (=3: also lambda+ from the scaled points, CVXOPT's update_scaling (misc.py): with v = w^1/2 of the old scaling,
                                  * sn = t / |t|_J (BRISK's z~), zn = s / |s|_J (BRISK's x~), c = g2 / 2, q = (sn + J zn) / 2c, u = sn - J zn:
                                  * lambda0 = c, lambda1 = 2 v1 (-d v'q + v'u / 2) + sn1 (1 - d / c) / 2 + zn1 (1 + d / c) / 2,
                                  * d = (v0 v'u - sn0 / 2 + zn0 / 2) / (2 v0 v'q - q0 + 1), all times sqrt(|s|_J |t|_J): no entry of x or z enters) */
            double *lp = S->scu_l + s0; const double c = 0.5 * g2, v0 = sqrt(0.5 * (wb[0] + 1.0)), iv = 0.5 / v0;
            double vs = v0 * zt[0] / rt, vz = v0 * st[0] / rs;      /* (v'sn, v'J zn: v1 = w1 / (2 v0)) */
            for (int i = 1; i < d; i++) { vs += iv * wb[i] * zt[i] / rt; vz -= iv * wb[i] * st[i] / rs; }
            const double vq = (vs + vz) / (2.0 * c), vu = vs - vz, q0 = (zt[0] / rt + st[0] / rs) / (2.0 * c);
            const double wk0 = 2.0 * v0 * vq - q0, dd = (v0 * vu - 0.5 * zt[0] / rt + 0.5 * st[0] / rs) / (wk0 + 1.0), sc = sqrt(rs * rt);
            const double f1 = 2.0 * (-dd * vq + 0.5 * vu) * iv, fs = 0.5 * (1.0 - dd / c) / rt, fz = 0.5 * (1.0 + dd / c) / rs;
            lp[0] = sc * c;
            for (int i = 1; i < d; i++) lp[i] = sc * (f1 * wb[i] + fs * zt[i] + fz * st[i]);
        }
        const double f = h0 + q / (1.0 + a);
        double uu = 0; for (int i = 1; i < d; i++) { u[i] += f * wb[i]; uu += u[i] * u[i]; }
        u[0] = sqrt(1.0 + uu);                           /* (u = H(w) w^, det u = 1 enforced on u0 as nt_update does) */
        S->scu_eta[k] = S->eta[k] * sqrt(rt / rs);
    }
    return bad;
}

static void *xcalloc(size_t n, size_t s) { void *p = calloc(n ? n : 1, s); return p; }
/* (7 November 2026, the SOCP developer; ISSUES 853) the full system's analysis: an allocation that fails ends the solve with the
 * memory-limit status (schol_oom jumps back to socp_solve_body) instead of a segmentation fault at the first use of a NULL array
 * (2013_firL2Linfalph, SOCP: 1.2e8 entries of the pattern under a 13 GB cap) */
static void *kkt_alloc(size_t n, size_t s, int zero) {
    void *p = zero ? calloc(n ? n : 1, s) : malloc((n ? n : 1) * s);
    if (!p) { fprintf(stderr, "brisk: out of memory (the full system's analysis, %.1f GB)\n", (double)n * (double)s / 1e9); schol_oom(); }
    return p;
}

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
    /* (2 November 2026, the SOCP developer) on the threads, each its range of columns: the sparse part's places from a count
     * of each column's entries outside the dense rows, then each column's entries placed (the same arrays as one thread's).
     * 2013_firL2Linfalph (SOCP): 2,001 dense rows over 91,781 columns, 121 million entries; the block's writes 2,001 rows
     * apart, on one thread most of the setup's 13.8 s */
    const int dnt_ = !omp_in_parallel() ? g_cone_nt : 1;
    #pragma omp parallel for schedule(static) num_threads(dnt_) if (dnt_ > 1)
    for (int j = 0; j < n; j++) { int c_ = 0; for (int p = S->Ap[j]; p < S->Ap[j + 1]; p++) c_ += dmap[S->Ai[p]] < 0; Bp[j + 1] = c_; }
    Bp[0] = 0; for (int j = 0; j < n; j++) Bp[j + 1] += Bp[j];
    #pragma omp parallel for schedule(static) num_threads(dnt_) if (dnt_ > 1)
    for (int j = 0; j < n; j++) {
        int t = Bp[j];
        for (int p = S->Ap[j]; p < S->Ap[j + 1]; p++) {
            const int i = S->Ai[p];
            if (dmap[i] >= 0) { Dm[(size_t)dmap[i] * nc + dcm[j]] = S->Ax[p]; Bq[p] = -1; }
            else { Bi[t] = i; Bx[t] = S->Ax[p]; Bq[p] = t; t++; }
        }
    }
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
    if (S->scu_redo) S->scu_redo[k] = 1;      /* (BRISK_CONESCALEUPD: the boosted cone's scaling formed from its point again) */
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
    free(S->lz_ti); free(S->lz_tj); free(S->lz_pos); free(S->lz_perm); S->lz_ti = S->lz_tj = S->lz_pos = S->lz_perm = NULL; S->sc_lazy = 0;
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
    const int sup_ = S->sc || S->sc_lazy;      /* (an analysis put off is one of the supernodal factorization: kkt_sc_build) */
    const double lz = (double)(sup_ ? S->slnz : S->F.lnz);
    *tf = S->cflops / (sup_ ? 6e9 : 1.2e9); *tsv = lz * (sup_ ? 3e-9 : 1.5e-9) + (double)S->Ap[S->n] * 3e-9 + (double)(S->n + S->m) * 2e-8;
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
/* (2 November 2026) the full system's supernodal analysis from the pattern and order kkt_analyse kept (S->lz_*): the analysis,
 * the signs, the factorization in place, the values' offsets; as kkt_analyse made them at once before (see there) */
static void kkt_sc_build(Socp *S) {
    if (!S->sc_lazy) return;
    const int n = S->n, nk = S->nk; const long nz = S->lz_nz; const int *ti = S->lz_ti, *tj = S->lz_tj, *pos = S->lz_pos;
    S->sc_lazy = 0;
    int *deg = (int *)kkt_alloc((size_t)nk + 1, sizeof(int), 1), **nbr = (int **)kkt_alloc((size_t)nk + 1, sizeof(int *), 0);   /* (checked: ISSUES 853) */
    for (long q = nk; q < nz; q++) { deg[ti[q]]++; deg[tj[q]]++; }
    int *pool = (int *)kkt_alloc((size_t)(2 * (nz - nk) + 1), sizeof(int), 0);
    { long o = 0; for (int i = 0; i < nk; i++) { nbr[i] = pool + o; o += deg[i]; deg[i] = 0; } }
    for (long q = nk; q < nz; q++) { nbr[ti[q]][deg[ti[q]]++] = tj[q]; nbr[tj[q]][deg[tj[q]]++] = ti[q]; }
    const int om = schol_set_amd(2);
    schol_set_perm(S->lz_perm);
    const double tsa_ = sc_time();
    const double oam = schol_set_amalg(cone_amalg(S->lz_amalg));
    S->sc = schol_analyze_adj(nk, deg, nbr, (size_t)1 << 31);
    schol_set_amalg(oam);
    if (dbg_on()) socp_printf("  [supernodal analysis %.3fs]\n", sc_time() - tsa_);
    schol_set_perm(NULL);
    schol_set_amd(om);
    free(deg); free(nbr); free(pool);
    if (S->sc) S->slnz = (long)schol_nnz(S->sc); else S->slnz = 0;
    if (S->sc) {
        signed char *sg = (signed char *)kkt_alloc((size_t)nk + 1, 1, 0);
        for (int i = 0; i < nk; i++) sg[i] = i < n ? -1 : i < n + S->naux ? (((i - n) & 1) ? 1 : -1) : 1;
        schol_set_signs(S->sc, sg); schol_set_tinypiv(S->sc, 0.0);
        /* (23 October 2026, the SOCP developer) the factorization over the assembled values, which kkt_factor zeroes and
         * writes again before every factorization (as the LP method does): one array of the factor's size instead of two,
         * the same arithmetic (socp_svmsparse_200kx20k (SOCP): 3.2 GB each) */
        if (!getenv("BRISK_CONENOINPLACE")) schol_set_inplace(S->sc, 1);
        free(sg);
        S->soff = (size_t *)kkt_alloc((size_t)(nz + 1), sizeof(size_t), 0); S->knz = nz;
        int bad = 0; const double tso_ = sc_time();
        /* (19 October 2026, LP/SOCP: on the threads, each lookup independent; a group lasso's 4.5e6 entries 0.27 -> 0.08 s) */
        #pragma omp parallel for schedule(static) num_threads(g_cone_nt) reduction(|:bad) if (nz >= 1000000 && g_cone_nt > 1 && !omp_in_parallel())
        for (long q = 0; q < nz; q++) { S->soff[pos[q]] = schol_offset(S->sc, (size_t)ti[q], (size_t)tj[q]); if (S->soff[pos[q]] == (size_t)-1) bad = 1; }
        if (dbg_on()) socp_printf("  [offsets %.3fs]\n", sc_time() - tso_);
        if (bad) { schol_free(S->sc); S->sc = NULL; free(S->soff); S->soff = NULL; S->slnz = 0; }
        else S->swork = (double *)kkt_alloc(2 * (size_t)nk + 2, sizeof(double), 1);
    }
    free(S->lz_ti); free(S->lz_tj); free(S->lz_pos); free(S->lz_perm); S->lz_ti = S->lz_tj = S->lz_pos = S->lz_perm = NULL;
}
static int kkt_analyse(Socp *S) {
    const double tka0_ = sc_time();
    int lz_own = 0, sc_now = 0;       /* (the pattern's arrays handed to S->lz_*; the supernodal analysis made at the end) */
    const double flop_ratio = 110.0;
    double nd0 = 0;                   /* dense rows found by the first AMD pass */
    const int n = S->n, m = S->m, nk = S->nk; const long nrot_tr = S->nrot_tr;
    S->kkt_defer = 0; S->kkt_defer_md = 0;
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
    if (dbg_on()) socp_printf("  [the full system's pattern: %ld entries, %.3fs]\n", nz, sc_time() - tka0_);
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
    /* (3 November 2026, the SOCP developer) the adjacency lists of the whole graph built only when a step needs them (nested
     * dissection, off by default; BRISK_CONEAMDSTRIP=0: always, AMD then on the whole graph as before): AMD gets the graph
     * without the unknowns it sets aside as dense (below), and the other steps read the pattern's entries directly */
    int *fi = NULL, *nx = NULL;
#define KKT_FI() do { if (!fi) { fi = (int *)malloc(sizeof(int) * (size_t)(cnt[nk] + 1)); nx = (int *)malloc(sizeof(int) * (size_t)nk); \
        memcpy(nx, cnt, sizeof(int) * (size_t)nk);                                                                                 \
        for (long e = 0; e < nz; e++) { fi[nx[tj[e]]++] = ti[e]; if (ti[e] != tj[e]) fi[nx[ti[e]]++] = tj[e]; } } } while (0)
    static int amdstrip = -1; if (amdstrip < 0) { const char *e_ = getenv("BRISK_CONEAMDSTRIP"); amdstrip = e_ ? atoi(e_) : 1; }
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
    /* the reduced graph's degrees, each with its diagonal (the lengths of its lists, cntr below) */
    int *dred = (int *)xcalloc((size_t)nred + 1, sizeof(int));
    for (long e = 0; e < nz; e++) { const int a_ = rmap[ti[e]], b_ = rmap[tj[e]]; if (a_ < 0 || b_ < 0) continue; dred[a_]++; if (a_ != b_) dred[b_]++; }
    int *cntr = NULL, *fir = NULL;
#define KKT_FIR() do { if (!fir) { KKT_FI();                                                                              \
        if (nsing) {                                                                                                       \
            cntr = (int *)xcalloc((size_t)nred + 1, sizeof(int)); fir = (int *)malloc(sizeof(int) * (size_t)(cnt[nk] + 1)); \
            int w_ = 0;                                                                                                    \
            for (int i = 0; i < nk; i++) if (rmap[i] >= 0) {                                                               \
                for (int p = cnt[i]; p < cnt[i + 1]; p++) if (rmap[fi[p]] >= 0) fir[w_++] = rmap[fi[p]];                  \
                cntr[rmap[i] + 1] = w_;                                                                                     \
            }                                                                                                              \
        } else { cntr = cnt; fir = fi; } } } while (0)
    if (!amdstrip) KKT_FIR();
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
    if (dbg_on()) socp_printf("  [the graph for the ordering: %.3fs from the start]\n", sc_time() - tka0_);
    Ldl *F = &S->F;
    /* (3 November 2026, the SOCP developer; ISSUES 819) the value a replaced pivot takes is the solver's setting, not part of
     * the analysis: kept across the clearing. Cleared, every analysis after the solver's start (the full system analysed
     * when first needed; the small cones put in their eigenbasis during a solve) left it at 0, so a replaced pivot became 0,
     * its inverse infinite, and the direction NaN (joint_FC_5, SOCP, the eigenbasis form: NaN from the first double-double
     * factorization, NUMERICAL DIFFICULTIES) */
    const double dyn_keep = F->dyn_delta;
    memset(F, 0, sizeof(*F));
    F->dyn_delta = dyn_keep;
    F->n = nk;
    F->perm = (int *)kkt_alloc((size_t)nk, sizeof(int), 0); F->iperm = (int *)kkt_alloc((size_t)nk, sizeof(int), 0);
    F->Ap = (int *)kkt_alloc((size_t)nk + 1, sizeof(int), 0);
    F->Ai = (int *)kkt_alloc((size_t)nz + 1, sizeof(int), 0); F->Ax = (double *)kkt_alloc((size_t)nz + 1, sizeof(double), 1);
    int *pos = (int *)kkt_alloc((size_t)nz, sizeof(int), 0);
    F->etree = (int *)kkt_alloc((size_t)nk, sizeof(int), 0); F->Lnz = (int *)kkt_alloc((size_t)nk, sizeof(int), 0);
    F->Lp = (int *)kkt_alloc((size_t)nk + 1, sizeof(int), 0);
    F->D = (double *)kkt_alloc((size_t)nk, sizeof(double), 0); F->Dinv = (double *)kkt_alloc((size_t)nk, sizeof(double), 0);
    F->iw = (int *)kkt_alloc(3 * (size_t)nk, sizeof(int), 0); F->bw = (unsigned char *)kkt_alloc((size_t)nk, 1, 0); F->fw = (double *)kkt_alloc((size_t)nk, sizeof(double), 0);
    F->sign = (int *)kkt_alloc((size_t)nk, sizeof(int), 0);
    int *bestp = (int *)kkt_alloc((size_t)nk, sizeof(int), 0), *cand = (int *)kkt_alloc((size_t)nk, sizeof(int), 0);
    double bestfl = -1.0; long bestlnz = 0;
    int symb_fail = 0;
#define KKT_COUNT() do {                                                                                          \
        const double tkc_ = sc_time();                                                                            \
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
        if (dbg_on()) socp_printf("  [a symbolic count %.3fs]\n", sc_time() - tkc_);                             \
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
                for (int i = 0; i < nred; i++) { const double d_ = (double)(dred[i] - 1); if (d_ > ra && d_ <= rb) betw_r++; }
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
                /* (3 November 2026, the SOCP developer) AMD on the graph without the unknowns it sets aside as dense, the same
                 * order. AMD removes an unknown whose degree exceeds max(16, alpha sqrt(n)) (truncated to an integer) before it
                 * orders the rest, places those last in their natural order, and skips them in every degree update (their
                 * weight is zero); only the other unknowns' starting degrees count them. So the graph AMD gets here leaves out
                 * the entries of those unknowns, and their full starting degrees are handed to it (amd_2.c, brisk_amd_deg0):
                 * the same order, without reading and copying their entries (CBLIB's beam30, SOCP: 1,997 such unknowns of
                 * 62,590 neighbours each, 125 of the graph's 129 million entries; the graph and AMD 6.6 + 8.8 s) */
                long nent_ = 0;
                if (amdstrip) {
                    long dthr_ = (long)(int)(ctl[AMD_DENSE] * sqrt((double)nred)); if (dthr_ < 16) dthr_ = 16; if (dthr_ > nred) dthr_ = nred;
                    unsigned char *dz_ = (unsigned char *)malloc((size_t)nred + 1); int *deg0_ = (int *)malloc(sizeof(int) * ((size_t)nred + 1));
                    for (int i = 0; i < nred; i++) { deg0_[i] = dred[i] - 1; dz_[i] = (long)deg0_[i] > dthr_; }
                    int *sp_ = (int *)xcalloc((size_t)nred + 1, sizeof(int));
                    for (long e = 0; e < nz; e++) { const int a_ = rmap[ti[e]], b_ = rmap[tj[e]]; if (a_ < 0 || b_ < 0) continue;
                        if (a_ == b_) { sp_[a_ + 1]++; continue; } if (dz_[a_] || dz_[b_]) continue; sp_[a_ + 1]++; sp_[b_ + 1]++; }
                    for (int i = 0; i < nred; i++) sp_[i + 1] += sp_[i];
                    int *si_ = (int *)malloc(sizeof(int) * ((size_t)sp_[nred] + 1)), *nx_ = (int *)malloc(sizeof(int) * ((size_t)nred + 1));
                    memcpy(nx_, sp_, sizeof(int) * (size_t)nred);
                    for (long e = 0; e < nz; e++) { const int a_ = rmap[ti[e]], b_ = rmap[tj[e]]; if (a_ < 0 || b_ < 0) continue;
                        if (a_ == b_) { si_[nx_[a_]++] = a_; continue; } if (dz_[a_] || dz_[b_]) continue; si_[nx_[b_]++] = a_; si_[nx_[a_]++] = b_; }
                    free(nx_);
                    extern __thread const int *brisk_amd_deg0;
                    brisk_amd_deg0 = deg0_;
                    if (amd_order(nred, sp_, si_, cred, ctl, info) < 0) for (int i = 0; i < nred; i++) cred[i] = i;
                    brisk_amd_deg0 = NULL;
                    nent_ = sp_[nred];
                    free(sp_); free(si_); free(dz_); free(deg0_);
                } else {
                    if (amd_order(nred, cntr, fir, cred, ctl, info) < 0) for (int i = 0; i < nred; i++) cred[i] = i;
                    nent_ = cntr[nred];
                }
#ifdef HAVE_METIS
                /* (13 November 2026, the SOCP developer; BRISK_CONEKKTMETIS, the default 2) the full system ordered by METIS's nested
                 * dissection instead of AMD where it is first analysed after the normal equations were left: the end of the solve lies ahead
                 * on it, often in double-double, where a factorization costs most (db-joint-soerensen, SOCP: 1.85e9 -> 9.1e8 flops, 5.2 s of
                 * ordering on four threads; the iterations in double-double about 6 s against 7-10, DIMACS 5.5e-8 at 227 s where it took more
                 * than 300). Not where the solve starts on the full system: there nested dissection did not pay on the supernodal factorization
                 * (20 October 2026, the 122 cone programs, nd.c). 0: AMD; 1: always */
                { static int kmet = -1; if (kmet < 0) { const char *e_ = getenv("BRISK_CONEKKTMETIS"); kmet = e_ ? atoi(e_) : 2; }
                  if ((kmet == 1 || (kmet == 2 && S->ne_left > 0 && nred >= 1000)) && nred > 1) {
                      extern int METIS_NodeND(int *nvtxs, int *xadj, int *adjncy, int *vwgt, int *options, int *perm, int *iperm);
                      int *xa = (int *)xcalloc((size_t)nred + 2, sizeof(int));
                      for (long e = 0; e < nz; e++) { const int a_ = rmap[ti[e]], b_ = rmap[tj[e]]; if (a_ < 0 || b_ < 0 || a_ == b_) continue; xa[a_ + 1]++; xa[b_ + 1]++; }
                      for (int i = 0; i < nred; i++) xa[i + 1] += xa[i];
                      int *ad = (int *)malloc(sizeof(int) * ((size_t)xa[nred] + 1)), *nx2 = (int *)malloc(sizeof(int) * ((size_t)nred + 1)), *ip2 = (int *)malloc(sizeof(int) * ((size_t)nred + 1));
                      memcpy(nx2, xa, sizeof(int) * (size_t)nred);
                      for (long e = 0; e < nz; e++) { const int a_ = rmap[ti[e]], b_ = rmap[tj[e]]; if (a_ < 0 || b_ < 0 || a_ == b_) continue; ad[nx2[a_]++] = b_; ad[nx2[b_]++] = a_; }
                      int nv = nred; const double tm_ = sc_time(); int *pm2 = (int *)malloc(sizeof(int) * ((size_t)nred + 1));
                      { extern void BRISK_METIS_SetThreads(int nt); BRISK_METIS_SetThreads(g_cone_nt > 0 ? g_cone_nt : 1); }
                      if (METIS_NodeND(&nv, xa, ad, NULL, NULL, pm2, ip2) == 1) { memcpy(cred, pm2, sizeof(int) * (size_t)nred);
                          if (dbg_on()) socp_printf("  [the full system: METIS's order %.3fs]\n", sc_time() - tm_); }
                      free(xa); free(ad); free(nx2); free(ip2); free(pm2);
                  } }
#endif
                if (dbg_on()) { const double thr_ = fmax(16.0, ctl[AMD_DENSE] * sqrt((double)nred)); double P_ = 0; unsigned long long hs_ = 1469598103934665603ULL;
                    for (int i = 0; i < nred; i++) { const double d_ = (double)(dred[i] - 1); if (d_ <= thr_) P_ += d_ * d_; hs_ = (hs_ ^ (unsigned)cred[i]) * 1099511628211ULL; }
                    socp_printf("  [AMD itself %.3fs on %d unknowns, %ld entries; sum of the kept degrees squared %.2e; the order's hash %016llx]\n", sc_time() - ta, nred, nent_, P_, hs_); }
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
                for (long e = 0; e < nz; e++) { const int a_ = ti[e], b_ = tj[e]; if (a_ == b_) continue;      /* (the neighbours of the dense ones, from the pattern's entries) */
                    if (cls[a_] == 2 && cls[b_] == 0) cls[b_] = 1; else if (cls[b_] == 2 && cls[a_] == 0) cls[a_] = 1; }
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
            KKT_FIR();
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
    free(fi); free(nx); free(cand); free(owner); free(rmap); free(sing_ptr); free(sing_row); free(dred); if (nsing) { free(cntr); free(fir); }
#undef KKT_FI
#undef KKT_FIR
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
            /* (2 November 2026, the SOCP developer) the supernodal analysis put off when the normal equations may still be taken
             * (ne_setup judges them on the counts above: F->lnz and S->cflops), and made when the full system is first needed
             * (kkt_fill) or when the normal equations are refused (socp_solve); the same analysis and order then. The owner's SOCP
             * benchmark, flow P152 (SOCP): the full system's supernodal analysis 5.1 s of a 107 s solve, for a factor that the
             * normal equations replace. BRISK_CONESCLAZY=0: at once, as before */
            static int lzon = -1; if (lzon < 0) { const char *e_ = getenv("BRISK_CONESCLAZY"); lzon = !(e_ && atoi(e_) == 0); }
            /* (3 November 2026, the SOCP developer) the pattern's arrays handed over, not copied (only read from here on; CBLIB's
             * beam30, SOCP: three arrays of 64 million entries, 2.5 s), and the analysis made at once, when it is not put off,
             * after this function's last use of them (below), as when it is put off */
            S->lz_ti = ti; S->lz_tj = tj; S->lz_pos = pos; lz_own = 1;
            S->lz_perm = (int *)malloc(sizeof(int) * (size_t)nk);
            memcpy(S->lz_perm, bestp, sizeof(int) * (size_t)nk);
            S->lz_nz = nz; S->lz_amalg = (double)F->lnz / (double)nk; S->sc_lazy = 1; S->slnz = F->lnz;
            sc_now = !(lzon && S->m > 0 && !S->ne);
        }
    }
    free(bestp);
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
    if (!lz_own) { free(ti); free(tj); free(pos); }      /* (else S->lz_* hold them: kkt_sc_build frees them) */
    free(cnt); free(blk0); free(u0); free(v0);
    if (sc_now) kkt_sc_build(S);
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
    if (S->ne_park) schol_free(S->ne_park); free(S->rspb); free(S->lrb); free(S->lrnb); free(S->lrsel); free(S->scu_w); free(S->scu_eta); free(S->scu_t); free(S->scu_l); free(S->scu_redo); free(S->q); free(S->qs); free(S->Ap); free(S->Ai); free(S->Ax); free(S->Rp); free(S->Ri); free(S->Rx); free(S->R0x); free(S->Rq); free(S->axc_part);     /* (Rq: 28 October 2026, a leak found by the address sanitizer, 4 bytes per entry of A per solve at two threads or more on 100,000 entries) */
    if (S->nd > 0) { free(S->drow); free(S->dmap); free(S->dcol); free(S->dcm); free(S->Dm); free(S->Bp); free(S->Bi); free(S->Bq); free(S->Bx); free(S->dxs); free(S->dacc); free(S->dpart); } free(S->b); free(S->c);
    free(S->Dr); free(S->Ec); free(S->x); free(S->y); free(S->z); free(S->h); free(S->eta); free(S->wbar); free(S->lam); free(S->ne_kcsel); S->ne_kcsel = NULL;
    free(S->w1); free(S->w2); free(S->w3); free(S->w4); free(S->rk); free(S->sk); ne_free(S); free(S->swork2);
    free(S->lastx); free(S->gs); free(S->gV); free(S->gZ); free(S->gH); free(S->gcs); free(S->fcone);
    /* (16 November 2026, the integrator, the release check of 1.7: the rotated cones' balancing arrays were never freed) */
    if (S->bpos) for (int k = 0; k < S->nq; k++) free(S->bpos[k]);
    free(S->bpos); free(S->bth); S->bpos = NULL; S->bth = NULL;
}

void socp_default_opts(SocpOpts *o) { o->tol = 1e-8; o->maxit = 200; o->timelimit = 0; o->verbose = 1; o->equil = getenv("BRISK_CONEEQ") ? atoi(getenv("BRISK_CONEEQ")) : 1; o->step = 0; o->attempt = 0; o->wx = o->wy = o->wz = NULL; o->wlam = 0.99; o->wsave = 0; o->cutoff = INFINITY; o->noretry = 0; o->fr = 0; }
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
    #pragma omp parallel for schedule(static) num_threads(g_cone_nt) if (cone_vpar(S))
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
    SOLVE_TAG(4); kkt_solve(S, C->rhsx, C->rhsy, C->qx, C->qy, maxref);
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
static int socp_solve_core_(const SocpProb *P, const SocpOpts *opt, SocpRes *R, double t0);
static int socp_bound_count_at(const SocpProb *P, int where);
static int socp_fr_count(const SocpProb *P);
/* (31 October 2026, the SOCP developer; ISSUES 729 A6, the integrator's request) the time limit holds in the setup too: the
 * analyses of the normal equations and of the full system (the sparse Cholesky analysis, nested dissection) stop at this
 * solve's deadline, and a setup that ends past it returns TIME LIMIT after 0 iterations instead of running the iterations'
 * check only then (grouplasso_3000, SOCP, -timelimit 0.3: 1.12 s, 0.93 s of it the analysis of the normal equations). The
 * caller's stop settings are put back on return */
/* (10 November 2026, the SOCP developer; bench/socp_1110/BOUNDS.md; a test: BRISK_CONEBOXSOC=1) a box written as a row, an upper bound on a
 * nonnegative variable a x_j + t = r (t a nonnegative slack of its own, a > 0, u = r / a > 0), solved as what it is: a two-dimensional
 * second-order cone with a fixed head. With x_j = u/2 + v and a new head h fixed at u/2 by the bound's own row, (h, v) in the cone is
 * |v| <= u/2, that is 0 <= x_j <= u, and the cone's barrier -log(h^2 - v^2) is -log(x_j) - log(u - x_j), the box's. The solver's rules for
 * cones fixed by a row of their own then give a native bound: the start puts h at u/2 (the bound holds exactly from the start), the normal
 * equations eliminate the row (no row and no fill for the bound). The slack t leaves the problem; v takes x_j's column without the row,
 * the other rows' right-hand sides lose a_ij u/2. Back: x_j = u/2 + v, t = r - a x_j; the lower bound's multiplier (z_h + z_v)/2 is x_j's z,
 * the upper bound's (z_h - z_v)/(2a) is t's z, and the row's multiplier is y_b = -z_t. */
static _Thread_local int g_boxsoc_in = 0;
static _Thread_local double g_fr_tau = 1.0;     /* (the core's last tau: plan item 3, the facial reduction after a failed solve) */
static _Thread_local int g_fr_depth = 0;
static _Thread_local int g_fr1_outer = 0;    /* (the single-row reduction made before the choice of the primal or dual form: its reduced problem is being solved) */
static _Thread_local long g_fr_adv[3] = { -1, 0, 0 };   /* (11 November 2026: with the facial reduction off, the face a failed solve shows: its nonnegative variables zero, cones zero, cones on one ray; -1: none) */
long socp_face_advice(long *nkz, long *nkr) { if (nkz) *nkz = g_fr_adv[1]; if (nkr) *nkr = g_fr_adv[2]; return g_fr_adv[0]; }
/* the facial reduction asked for: BRISK_CONEFR (tests: 0 off, 1 the single rows, 2 both, 3 the face after a failed solve) or the option
 * -conefr (SocpOpts.fr: 0 none, 1 the face after a failed solve, 2 both) */
static int fr_mode(const SocpOpts *opt) {
    static int env = -2; if (env == -2) { const char *e_ = getenv("BRISK_CONEFR"); env = e_ ? atoi(e_) : -1; }
    if (env >= 0) return env;
    const int f = opt ? opt->fr : 0;
    return f == 1 ? 3 : f == 2 ? 2 : 0;
}
static _Thread_local double g_fr_face_err = -1;    /* (a result on a face read from the iterate: its measure (the primal on the problem as given, the dual on the face); -1: none) */
typedef struct { double *x, *y, *z, tau, kap; } BoxStart;
static _Thread_local BoxStart *g_box_cap = NULL, *g_box_inj = NULL;
static _Thread_local int g_box_k0 = -1, g_box_k1 = -1;     /* (the box cones' indices, [k0, k1), while the rewritten problem is solved) */
static int socp_solve_core_box(const SocpProb *P, const SocpOpts *opt, SocpRes *R, double t0, int *done) {
    static int on = -1; if (on < 0) { const char *e_ = getenv("BRISK_CONEBOXSOC"); on = e_ ? atoi(e_) : 0; }
    *done = 0;
    if (!on || g_boxsoc_in || (opt && opt->wx)) return 0;
    const int m = P->m, n = P->n, nf = P->nf, nl = P->nl;
    int *rc = (int *)calloc((size_t)m + 1, sizeof(int)), *slk = (int *)malloc(sizeof(int) * ((size_t)m + 1));
    int *bj = (int *)malloc(sizeof(int) * ((size_t)m + 1)), *jbox = (int *)malloc(sizeof(int) * ((size_t)n + 1));
    double *ba = (double *)malloc(sizeof(double) * ((size_t)m + 1));
    for (int i = 0; i < m; i++) { slk[i] = -1; bj[i] = -1; }
    for (int j = 0; j < n; j++) jbox[j] = -1;
    for (int j = 0; j < n; j++) for (int p = P->Ap[j]; p < P->Ap[j + 1]; p++) rc[P->Ai[p]]++;
    for (int j = nf; j < nf + nl; j++) if (P->Ap[j + 1] - P->Ap[j] == 1) { const int i = P->Ai[P->Ap[j]]; if (slk[i] < 0 && rc[i] == 2) slk[i] = j; }
    int nb = 0;
    for (int j = nf; j < nf + nl; j++) for (int p = P->Ap[j]; p < P->Ap[j + 1]; p++) {
        const int i = P->Ai[p]; if (slk[i] < 0 || slk[i] == j || bj[i] >= 0 || jbox[j] >= 0) continue;
        const int t = slk[i]; const double st = P->Ax[P->Ap[t]], a = P->Ax[p] / st, r = P->b[i] / st;     /* (normalized: a x_j + t = r) */
        if (!(a > 0) || !(r / a > 0) || !isfinite(r / a) || P->c[t] != 0.0) continue;
        bj[i] = j; ba[i] = a; jbox[j] = i; nb++;
    }
    free(rc);
    if (nb == 0) { free(slk); free(bj); free(jbox); free(ba); return 0; }
    /* the new problem: columns [free | nonnegative without the boxed variables and their slacks | the cones | the boxes (h, v) | rotated] */
int nqv = 0; for (int k = 0; k < P->nq; k++) nqv += P->q[k];
    const int n2 = n, nl2 = nl - 2 * nb, nq2 = P->nq + nb;     /* (each box: x_j and t leave, h and v come: the count of columns is kept) */
    int *map = (int *)malloc(sizeof(int) * ((size_t)n + 1));       /* (the new column of each old one; -1: a slack that left) */
    int *isslk = (int *)calloc((size_t)n + 1, sizeof(int)); for (int i = 0; i < m; i++) if (bj[i] >= 0) isslk[slk[i]] = 1;
    int pos = 0;
    for (int j = 0; j < nf; j++) map[j] = pos++;
    for (int j = nf; j < nf + nl; j++) map[j] = (jbox[j] >= 0 || isslk[j]) ? -1 : pos++;
    for (int j = nf + nl; j < nf + nl + nqv; j++) map[j] = pos++;
    int *hcol = (int *)malloc(sizeof(int) * ((size_t)m + 1));
    for (int i = 0; i < m; i++) if (bj[i] >= 0) { hcol[i] = pos; map[bj[i]] = pos + 1; pos += 2; }
    for (int j = nf + nl + nqv; j < n; j++) map[j] = pos++;
    int *q2 = (int *)malloc(sizeof(int) * ((size_t)nq2 + 1)); for (int k = 0; k < P->nq; k++) q2[k] = P->q[k]; for (int k = P->nq; k < nq2; k++) q2[k] = 2;
    double *b2 = (double *)malloc(sizeof(double) * ((size_t)m + 1)), *c2 = (double *)calloc((size_t)n2 + 1, sizeof(double)); memcpy(b2, P->b, sizeof(double) * (size_t)m);
    long nz2 = 0; int *cnt = (int *)calloc((size_t)n2 + 1, sizeof(int));
    double cst = 0;
    for (int j = 0; j < n; j++) {
        if (map[j] < 0) continue;
        if (jbox[j] >= 0) {
            const int ib = jbox[j]; const double u = P->b[ib] / P->Ax[P->Ap[slk[ib]]] / ba[ib];
            cst += P->c[j] * 0.5 * u;
            for (int p = P->Ap[j]; p < P->Ap[j + 1]; p++) { const int i = P->Ai[p]; if (i == ib) continue; b2[i] -= P->Ax[p] * 0.5 * u; cnt[map[j]]++; nz2++; }
            b2[ib] = 0.5 * u; cnt[hcol[ib]]++; nz2++;
        } else { cnt[map[j]] += P->Ap[j + 1] - P->Ap[j]; nz2 += P->Ap[j + 1] - P->Ap[j]; }
        c2[map[j]] = P->c[j];
    }
    int *Ap2 = (int *)malloc(sizeof(int) * ((size_t)n2 + 1)), *Ai2 = (int *)malloc(sizeof(int) * ((size_t)nz2 + 1)); double *Ax2 = (double *)malloc(sizeof(double) * ((size_t)nz2 + 1));
    Ap2[0] = 0; for (int j = 0; j < n2; j++) Ap2[j + 1] = Ap2[j] + cnt[j];
    { int *w = (int *)malloc(sizeof(int) * ((size_t)n2 + 1)); memcpy(w, Ap2, sizeof(int) * (size_t)n2);
      for (int j = 0; j < n; j++) {
          if (map[j] < 0) continue;
          for (int p = P->Ap[j]; p < P->Ap[j + 1]; p++) { const int i = P->Ai[p]; if (jbox[j] >= 0 && i == jbox[j]) continue; const int k = w[map[j]]++; Ai2[k] = i; Ax2[k] = P->Ax[p]; }
          if (jbox[j] >= 0) { const int k = w[hcol[jbox[j]]]++; Ai2[k] = jbox[j]; Ax2[k] = 1.0; }
      }
      free(w); }
    SocpProb P2 = *P; P2.n = n2; P2.nl = nl2; P2.nq = nq2; P2.q = q2; P2.Ap = Ap2; P2.Ai = Ai2; P2.Ax = Ax2; P2.b = b2; P2.c = c2;
    if (opt && opt->verbose > 1) { double umin = 1e300, umax = 0; for (int i = 0; i < m; i++) if (bj[i] >= 0) { const double u = P->b[i] / P->Ax[P->Ap[slk[i]]] / ba[i]; umin = fmin(umin, u); umax = fmax(umax, u); }
        socp_printf("  [boxes as two-dimensional cones with a fixed head: %d boxes (rows %d, columns %d -> %d), bounds %.2e .. %.2e]\n", nb, m, n, n2, umin, umax); }
    static int bxs5 = -1; if (bxs5 < 0) { const char *e_ = getenv("BRISK_CONEBOXSTART"); bxs5 = e_ && atoi(e_) == 5; }
    BoxStart cap = {0}, inj = {0};
    if (bxs5) {     /* (5: the start of the problem with its bound rows, taken by a solve of no iteration, mapped to the boxes' form) */
        cap.x = (double *)calloc((size_t)n + 1, sizeof(double)); cap.z = (double *)calloc((size_t)n + 1, sizeof(double)); cap.y = (double *)calloc((size_t)m + 1, sizeof(double));
        SocpOpts o0 = *opt; o0.maxit = 0; o0.verbose = 0; SocpRes R0; memset(&R0, 0, sizeof R0);
        g_boxsoc_in = 1; g_box_cap = &cap; socp_solve_core_(P, &o0, &R0, t0); g_box_cap = NULL; g_boxsoc_in = 0;
        free(R0.x); free(R0.y); free(R0.z); free(R0.cx); free(R0.cy); free(R0.cz);
        inj.x = (double *)calloc((size_t)n2 + 1, sizeof(double)); inj.z = (double *)calloc((size_t)n2 + 1, sizeof(double)); inj.y = (double *)calloc((size_t)m + 1, sizeof(double));
        inj.tau = cap.tau; inj.kap = cap.kap;
        for (int j = 0; j < n; j++) if (map[j] >= 0 && jbox[j] < 0) { inj.x[map[j]] = cap.x[j]; inj.z[map[j]] = cap.z[j]; }
        for (int i = 0; i < m; i++) inj.y[i] = cap.y[i];
        for (int i = 0; i < m; i++) {
            if (bj[i] < 0) continue;
            const int j = bj[i], t = slk[i], h = hcol[i], v = map[j]; const double st_ = P->Ax[P->Ap[t]], a = ba[i];
            inj.x[h] = 0.5 * (cap.x[j] + cap.x[t] / a); inj.x[v] = 0.5 * (cap.x[j] - cap.x[t] / a);
            inj.z[h] = cap.z[j] + a * cap.z[t]; inj.z[v] = cap.z[j] - a * cap.z[t];
            double s_ = 0; for (int p = P->Ap[j]; p < P->Ap[j + 1]; p++) if (P->Ai[p] != i) s_ += P->Ax[p] * cap.y[P->Ai[p]];
            inj.y[i] = 2.0 * st_ * a * cap.y[i] + s_ - P->c[j] * cap.tau;
        }
        g_box_inj = &inj;
    }
    g_boxsoc_in = 1; g_box_k0 = P->nq; g_box_k1 = nq2;
    SocpRes R2; memset(&R2, 0, sizeof R2);
    const int st = socp_solve_core_(&P2, opt, &R2, t0);
    g_boxsoc_in = 0; g_box_k0 = g_box_k1 = -1; g_box_inj = NULL;
    free(cap.x); free(cap.y); free(cap.z); free(inj.x); free(inj.y); free(inj.z);
    /* back to the problem as given */
    *R = R2; R->x = (double *)calloc((size_t)n + 1, sizeof(double)); R->z = (double *)calloc((size_t)n + 1, sizeof(double)); R->y = R2.y; R2.y = NULL;
    const int ray = st == SOCP_DINF, cert = st == SOCP_PINF;
    if (R2.x && R2.z) {
        for (int j = 0; j < n; j++) if (map[j] >= 0 && jbox[j] < 0) { R->x[j] = R2.x[map[j]]; R->z[j] = R2.z[map[j]]; }
        for (int i = 0; i < m; i++) {
            if (bj[i] < 0) continue;
            const int j = bj[i], t = slk[i], h = hcol[i], v = map[j]; const double st_ = P->Ax[P->Ap[t]], a = ba[i], u = P->b[i] / st_ / a;
            const double xj = (ray || cert ? 0.0 : 0.5 * u) + R2.x[v];
            R->x[j] = xj; R->x[t] = (ray || cert ? 0.0 : P->b[i] / st_) - a * xj;
            const double zh = R2.z[h], zv = R2.z[v];
            R->z[j] = 0.5 * (zh + zv); R->z[t] = (zh - zv) / (2.0 * a);     /* (the row divided by st, t's coefficient: a x_j + t = r; z_t = -st y_i) */
            if (R->y) R->y[i] = -R->z[t] / st_;
        }
    }
    if (!ray && !cert) { R->pobj += cst; R->dobj += cst; }
    free(R2.x); free(R2.z); free(R2.cx); free(R2.cy); free(R2.cz); R->cx = R->cy = R->cz = NULL;
    free(slk); free(bj); free(jbox); free(ba); free(map); free(isslk); free(hcol); free(q2); free(b2); free(c2); free(cnt); free(Ap2); free(Ai2); free(Ax2);
    *done = 1;
    return st;
}
/* ---- facial reduction for the SOCP solver (10 November 2026, the SOCP developer; plan item 3; tests: BRISK_CONEFR=1 the faces single rows
 * prove, removed before the solve; 2 also the face read from a failed solve's last iterate) ----
 * Every feasible x of a problem whose rows have a combination y with s = A'y in the dual cone K*, s zero on the free variables and b'y = 0
 * satisfies s'x = 0 (x in K, s in K*: each block's product is >= 0 and they sum to b'y = 0), so it lies in the face of K orthogonal to s:
 * a nonnegative variable with s_j > 0 is zero, a cone with s_k inside the cone is zero, a cone with s_k on its boundary lies on the ray
 * orthogonal to s_k ((s_0, -sbar) for a second-order cone, (s_v, s_u, -sbar) for a rotated one; the cones are self-dual). Such a problem
 * has no strictly feasible point, its optimal multipliers need not exist, and an interior-point method loses accuracy on it (Borwein and
 * Wolkowicz 1981; Pataki 2013). On the face the problem is smaller and the cause is gone. */
static int socp_solve_core_box(const SocpProb *P, const SocpOpts *opt, SocpRes *R, double t0, int *done);
static int socp_solve_core_fr(const SocpProb *P, const SocpOpts *opt, SocpRes *R, double t0, int *done);
/* the problem on a face: columns [free | nonnegative kept | one per cone on a ray | second-order cones kept | rotated kept]; a ray's
 * column is A_k r_k and its cost c_k'r_k; the rows rdel removed */
typedef struct { SocpProb P; int *cmap, *rmap, *kray, *q, *r, *Ap, *Ai; double *Ax, *b, *c; } FrFace;
static void fr_face_build(const SocpProb *P, const int *ks, const int *kd, const unsigned char *zc, const signed char *kf, const double *ray, const unsigned char *rdel, FrFace *F) {
    const int m = P->m, n = P->n, nf = P->nf, nl = P->nl, nq = P->nq, nr = P->nr, nk = nq + nr;
    F->cmap = (int *)malloc(sizeof(int) * ((size_t)n + 1)); F->rmap = (int *)malloc(sizeof(int) * ((size_t)m + 1)); F->kray = (int *)malloc(sizeof(int) * ((size_t)nk + 1));
    F->q = (int *)malloc(sizeof(int) * ((size_t)nq + 1)); F->r = (int *)malloc(sizeof(int) * ((size_t)nr + 1));
    int n3 = 0, m3 = 0, nl3 = 0, nq3 = 0, nr3 = 0;
    for (int j = 0; j < nf; j++) F->cmap[j] = n3++;
    for (int j = nf; j < nf + nl; j++) { F->cmap[j] = zc[j] ? -1 : n3++; if (!zc[j]) nl3++; }
    for (int k = 0; k < nk; k++) { F->kray[k] = -1; if (kf[k] == 2) { F->kray[k] = n3++; nl3++; } }
    for (int k = 0; k < nk; k++) { const int j0 = ks[k]; for (int t = 0; t < kd[k]; t++) F->cmap[j0 + t] = kf[k] ? -1 : n3 + t; if (!kf[k]) { n3 += kd[k]; if (k < nq) F->q[nq3++] = kd[k]; else F->r[nr3++] = kd[k]; } }
    for (int i = 0; i < m; i++) F->rmap[i] = rdel[i] ? -1 : m3++;
    long nz3 = 0; for (int j = 0; j < n; j++) if (F->cmap[j] >= 0) nz3 += P->Ap[j + 1] - P->Ap[j];
    for (int k = 0; k < nk; k++) if (kf[k] == 2) for (int t = 0; t < kd[k]; t++) nz3 += P->Ap[ks[k] + t + 1] - P->Ap[ks[k] + t];
    F->Ap = (int *)calloc((size_t)n3 + 1, sizeof(int)); F->Ai = (int *)malloc(sizeof(int) * ((size_t)nz3 + 1)); F->Ax = (double *)malloc(sizeof(double) * ((size_t)nz3 + 1));
    F->b = (double *)malloc(sizeof(double) * ((size_t)m3 + 1)); F->c = (double *)calloc((size_t)n3 + 1, sizeof(double));
    for (int i = 0; i < m; i++) if (F->rmap[i] >= 0) F->b[F->rmap[i]] = P->b[i];
    int *own = (int *)malloc(sizeof(int) * ((size_t)n3 + 1)), *isray = (int *)malloc(sizeof(int) * ((size_t)n3 + 1));
    for (int j = 0; j < n; j++) if (F->cmap[j] >= 0) { own[F->cmap[j]] = j; isray[F->cmap[j]] = -1; }
    for (int k = 0; k < nk; k++) if (F->kray[k] >= 0) { own[F->kray[k]] = -1; isray[F->kray[k]] = k; }
    double *acc = (double *)calloc((size_t)m + 1, sizeof(double)); int *lst = (int *)malloc(sizeof(int) * ((size_t)m + 1)); unsigned char *in = (unsigned char *)calloc((size_t)m + 1, 1);
    long e = 0;
    for (int c_ = 0; c_ < n3; c_++) {
        if (isray[c_] < 0) { const int j = own[c_]; for (int p = P->Ap[j]; p < P->Ap[j + 1]; p++) if (F->rmap[P->Ai[p]] >= 0) { F->Ai[e] = F->rmap[P->Ai[p]]; F->Ax[e++] = P->Ax[p]; } F->c[c_] = P->c[j]; }
        else { const int k = isray[c_], j0 = ks[k]; int nlst = 0; double cc = 0;
            for (int t = 0; t < kd[k]; t++) { const int j = j0 + t; const double rr = ray[j]; if (rr == 0.0) continue; cc += P->c[j] * rr;
                for (int p = P->Ap[j]; p < P->Ap[j + 1]; p++) { const int i = P->Ai[p]; if (!in[i]) { in[i] = 1; lst[nlst++] = i; acc[i] = 0; } acc[i] += P->Ax[p] * rr; } }
            for (int u = 0; u < nlst; u++) { const int i = lst[u]; in[i] = 0; if (F->rmap[i] >= 0 && acc[i] != 0.0) { F->Ai[e] = F->rmap[i]; F->Ax[e++] = acc[i]; } }
            F->c[c_] = cc; }
        F->Ap[c_ + 1] = (int)e;
    }
    free(own); free(isray); free(acc); free(lst); free(in);
    F->P = *P; F->P.m = m3; F->P.n = n3; F->P.nl = nl3; F->P.nq = nq3; F->P.q = F->q; F->P.nr = nr3; F->P.r = F->r;
    F->P.Ap = F->Ap; F->P.Ai = F->Ai; F->P.Ax = F->Ax; F->P.b = F->b; F->P.c = F->c;
}
static void fr_face_free(FrFace *F) { free(F->cmap); free(F->rmap); free(F->kray); free(F->q); free(F->r); free(F->Ap); free(F->Ai); free(F->Ax); free(F->b); free(F->c); }
/* x on the problem as given from x on the face */
static void fr_face_x(const SocpProb *P, const int *ks, const int *kd, const double *ray, const FrFace *F, const double *x3, double *x) {
    const int nk = P->nq + P->nr;
    for (int j = 0; j < P->n; j++) x[j] = F->cmap[j] >= 0 ? x3[F->cmap[j]] : 0.0;
    for (int k = 0; k < nk; k++) if (F->kray[k] >= 0) { const double lam = x3[F->kray[k]]; for (int t = 0; t < kd[k]; t++) x[ks[k] + t] = lam * ray[ks[k] + t]; }
}
/* the solve of a problem with no variable left (every variable fixed at zero on the face): optimal when b = 0 */
static int fr_solve_inner(const SocpProb *P, const SocpOpts *opt, SocpRes *R, double t0) {
    if (P->n == 0) {
        memset(R, 0, sizeof *R); R->x = (double *)calloc(1, sizeof(double)); R->z = (double *)calloc(1, sizeof(double)); R->y = (double *)calloc((size_t)P->m + 1, sizeof(double));
        double nb = 0; for (int i = 0; i < P->m; i++) nb += P->b[i] * P->b[i];
        if (nb > 0) { nb = sqrt(nb); for (int i = 0; i < P->m; i++) R->y[i] = P->b[i] / nb; R->status = SOCP_PINF; } else R->status = SOCP_OPTIMAL;
        return R->status;
    }
    int bx = 0; int st = socp_solve_core_box(P, opt, R, t0, &bx); if (!bx) st = socp_solve_core_(P, opt, R, t0);
    return st;
}
/* the cone violation of z = v - y a on one block (0 nonnegative, 1 second-order, 2 rotated) */
static double fr_block_viol(int kind, int d, const double *v, const double *a, double y) {
    if (kind == 0) return fmax(0.0, -(v[0] - y * a[0]));
    double b0, u2 = 0;
    if (kind == 1) { b0 = v[0] - y * a[0]; for (int t = 1; t < d; t++) { const double w = v[t] - y * a[t]; u2 += w * w; } }
    else { const double zu = v[0] - y * a[0], zv = v[1] - y * a[1]; const double rs = sqrt(0.5); b0 = rs * (zu + zv); u2 = 0.5 * (zu - zv) * (zu - zv); for (int t = 2; t < d; t++) { const double w = v[t] - y * a[t]; u2 += w * w; } }
    return fmax(0.0, sqrt(u2) - b0);
}
/* the value of one multiplier y that brings every block's z = v_b - y a_b into its cone, or nearest to it: a golden-section search on the
 * largest violation, convex in y */
static double fr_golden(int nb, const int *kind, const int *dim, const double *const *v, const double *const *a, double H) {
    double lo = -H, hi = H; const double gr = 0.5 * (sqrt(5.0) - 1.0);
    #define FR_F(y_) ({ double f_ = 0; for (int b_ = 0; b_ < nb; b_++) f_ = fmax(f_, fr_block_viol(kind[b_], dim[b_], v[b_], a[b_], (y_))); f_; })
    double a_ = hi - gr * (hi - lo), b_ = lo + gr * (hi - lo), fa = FR_F(a_), fb = FR_F(b_);
    for (int it = 0; it < 300 && hi - lo > 1e-15 * (fabs(lo) + fabs(hi)); it++) {
        if (fa <= fb) { hi = b_; b_ = a_; fb = fa; a_ = hi - gr * (hi - lo); fa = FR_F(a_); }
        else { lo = a_; a_ = b_; fa = fb; b_ = lo + gr * (hi - lo); fb = FR_F(b_); }
    }
    #undef FR_F
    return 0.5 * (a_ + b_);
}
/* the faces single rows prove, chained: a row with b_i = 0, no free variable, every part of one sign in the dual cone (a cone already on a
 * ray seen as one nonnegative variable with the coefficient a_k'r_k). Back: each removed row's multiplier, in the reverse order of the
 * reductions (the blocks a row restricted meet only the rows alive and the rows removed after it), the value that keeps the dual slack of
 * every block of the row in its cone (fr_golden); the dual is then attained where it exists. */
static int socp_solve_core_fr1(const SocpProb *P, const SocpOpts *opt, SocpRes *R, double t0, int *done, int (*inner)(const SocpProb *, const SocpOpts *, SocpRes *, double)) {
    *done = 0;
    const int m = P->m, n = P->n, nf = P->nf, nl = P->nl, nq = P->nq, nr = P->nr, nk = nq + nr;
    int *rp = (int *)calloc((size_t)m + 2, sizeof(int)); for (int p = 0; p < P->Ap[n]; p++) rp[P->Ai[p] + 1]++;
    for (int i = 0; i < m; i++) rp[i + 1] += rp[i];
    int *rj = (int *)malloc(sizeof(int) * ((size_t)P->Ap[n] + 1)); double *rx = (double *)malloc(sizeof(double) * ((size_t)P->Ap[n] + 1));
    { int *w = (int *)malloc(sizeof(int) * ((size_t)m + 1)); memcpy(w, rp, sizeof(int) * (size_t)m);
      for (int j = 0; j < n; j++) for (int p = P->Ap[j]; p < P->Ap[j + 1]; p++) { const int k = w[P->Ai[p]]++; rj[k] = j; rx[k] = P->Ax[p]; } free(w); }
    int *cone = (int *)malloc(sizeof(int) * ((size_t)n + 1)), *ks = (int *)malloc(sizeof(int) * ((size_t)nk + 1)), *kd = (int *)malloc(sizeof(int) * ((size_t)nk + 1));
    for (int j = 0; j < n; j++) cone[j] = -1;
    { int s0 = nf + nl; for (int k = 0; k < nk; k++) { kd[k] = k < nq ? P->q[k] : P->r[k - nq]; ks[k] = s0; for (int t = 0; t < kd[k]; t++) cone[s0 + t] = k; s0 += kd[k]; } }
    unsigned char *zc = (unsigned char *)calloc((size_t)n + 1, 1), *rdel = (unsigned char *)calloc((size_t)m + 1, 1);
    signed char *kf = (signed char *)calloc((size_t)nk + 1, 1);
    double *ray = (double *)calloc((size_t)n + 1, sizeof(double)), *wv = (double *)calloc((size_t)n + 1, sizeof(double));
    int *tk = (int *)malloc(sizeof(int) * ((size_t)nk + 1)), *mark = (int *)malloc(sizeof(int) * ((size_t)nk + 1)); for (int k = 0; k < nk; k++) mark[k] = -1;
    int *cert = (int *)malloc(sizeof(int) * ((size_t)m + 1)); int ncert = 0;
    int *tc = (int *)malloc(sizeof(int) * ((size_t)P->Ap[n] + 1)); double *cs0 = (double *)malloc(sizeof(double) * ((size_t)nk + 1)), *cs1 = (double *)malloc(sizeof(double) * ((size_t)nk + 1)), *ct2 = (double *)malloc(sizeof(double) * ((size_t)nk + 1)), *clam = (double *)malloc(sizeof(double) * ((size_t)nk + 1));
    const double tol = 1e-12; long nlz = 0, nkz = 0, nkr = 0; int stamp = 0;
    for (int changed = 1, rounds = 0; changed && rounds < 1000; rounds++) {
        changed = 0;
        for (int i = 0; i < m; i++) {
            if (rdel[i] || P->b[i] != 0.0) continue;
            /* (the row's entries only: a cone's sums over the entries the row touches, so that a large cone met by many rows costs its
             * entries in each row and not its whole dimension; wv is zero outside the row's entries and cleared after the row) */
            int sg = 0, ok = 1, ntk = 0, nz = 0, ntc = 0; stamp++;     /* (a stamp per visit of a row: a row visited again in a later round marked its cones with its own index, and they were then not tested at all) */
            for (int p = rp[i]; p < rp[i + 1] && ok; p++) {
                const int j = rj[p]; const double a = rx[p]; if (a == 0.0) continue;
                if (j < nf) { ok = 0; break; }
                if (j < nf + nl) { if (zc[j]) continue; nz++; const int s_ = a > 0 ? 1 : -1; if (sg && s_ != sg) ok = 0; sg = s_; continue; }
                const int k = cone[j]; if (kf[k] == 1) continue;
                nz++; if (mark[k] != stamp) { mark[k] = stamp; tk[ntk++] = k; cs0[k] = cs1[k] = ct2[k] = clam[k] = 0; }
                wv[j] += a; tc[ntc++] = j;
            }
            for (int u = 0; u < ntc; u++) { const int j = tc[u], k = cone[j], t = j - ks[k]; const double w = wv[j];
                if (kf[k] == 2) clam[k] += w * ray[j];
                if (t == 0) cs0[k] = w; else if (t == 1 && k >= nq) cs1[k] = w; else ct2[k] += w * w; }
            if (ok && nz > 0) for (int u = 0; u < ntk && ok; u++) {
                const int k = tk[u]; const double t2 = ct2[k];
                if (kf[k] == 2) { const double lam = clam[k]; if (lam == 0.0) continue; const int s_ = lam > 0 ? 1 : -1; if (sg && s_ != sg) ok = 0; else sg = s_; continue; }
                if (k < nq) { const double s0 = cs0[k];
                    const int s_ = s0 > 0 ? 1 : s0 < 0 ? -1 : 0; if (!s_) { if (t2 > 0) ok = 0; continue; }
                    if ((sg && s_ != sg) || s0 * s0 < t2 * (1.0 - tol)) ok = 0; else sg = s_; }
                else { const double s0 = cs0[k], s1 = cs1[k];
                    const int s_ = (s0 > 0 || s1 > 0) ? 1 : (s0 < 0 || s1 < 0) ? -1 : 0; if (!s_) { if (t2 > 0) ok = 0; continue; }
                    if (s0 * s_ < 0 || s1 * s_ < 0 || (sg && s_ != sg) || 2.0 * s0 * s1 < t2 * (1.0 - tol)) ok = 0; else sg = s_; }
            }
            if (!ok || nz == 0 || !sg) { for (int u = 0; u < ntc; u++) wv[tc[u]] = 0.0; continue; }
            rdel[i] = 1; cert[ncert++] = i; changed = 1;
            if (getenv("BRISK_CONEFRDBG")) { socp_printf("  [single-row face: row %d (b %.3e, sign %d):", i, P->b[i], sg); for (int p = rp[i]; p < rp[i + 1]; p++) if (rj[p] < nf + nl) socp_printf(" x%d%s %.3e", rj[p], zc[rj[p]] ? "(zero)" : "", rx[p]); else socp_printf(" x%d(cone %d entry %d state %d ray %.2e) %.3e", rj[p], cone[rj[p]], rj[p] - ks[cone[rj[p]]], kf[cone[rj[p]]], ray[rj[p]], rx[p]); socp_printf("]\n"); }
            for (int p = rp[i]; p < rp[i + 1]; p++) { const int j = rj[p]; if (rx[p] == 0.0 || j >= nf + nl || zc[j]) continue; zc[j] = 1; nlz++; }
            for (int u = 0; u < ntk; u++) {
                const int k = tk[u], j0 = ks[k], d = kd[k]; double nn = 0; for (int t = 0; t < d; t++) nn += wv[j0 + t] * wv[j0 + t]; if (nn == 0.0) continue;
                if (kf[k] == 2) { double lam = 0; for (int t = 0; t < d; t++) lam += wv[j0 + t] * ray[j0 + t]; if (lam != 0.0) { kf[k] = 1; nkz++; nkr--; } continue; }
                const double s0 = sg * wv[j0], s1 = k < nq ? 0 : sg * wv[j0 + 1]; double t2 = 0; for (int t = (k < nq ? 1 : 2); t < d; t++) t2 += wv[j0 + t] * wv[j0 + t];
                const int inter = k < nq ? (s0 * s0 > t2 * (1.0 + tol)) : (s0 > 0 && s1 > 0 && 2.0 * s0 * s1 > t2 * (1.0 + tol));
                for (int t = 0; t < d; t++) zc[j0 + t] = 1;
                if (inter) { kf[k] = 1; nkz++; }
                else { kf[k] = 2; nkr++; double rn = 0;
                    if (k < nq) { ray[j0] = s0; for (int t = 1; t < d; t++) ray[j0 + t] = -sg * wv[j0 + t]; }
                    else { ray[j0] = s1; ray[j0 + 1] = s0; for (int t = 2; t < d; t++) ray[j0 + t] = -sg * wv[j0 + t]; }
                    for (int t = 0; t < d; t++) rn += ray[j0 + t] * ray[j0 + t]; rn = sqrt(rn); for (int t = 0; t < d; t++) ray[j0 + t] /= rn; }
            }
            for (int u = 0; u < ntc; u++) wv[tc[u]] = 0.0;
        }
    }
    free(wv); free(tk); free(tc); free(cs0); free(cs1); free(ct2); free(clam);
    if (ncert == 0) { free(rp); free(rj); free(rx); free(cone); free(ks); free(kd); free(zc); free(rdel); free(kf); free(ray); free(cert); free(mark); return 0; }
    FrFace F; fr_face_build(P, ks, kd, zc, kf, ray, rdel, &F);
    if (opt && opt->verbose > 0) socp_printf("facial reduction: %d rows prove that %ld nonnegative variables and %ld cones are zero and %ld cones lie on a ray in every feasible point (rows %d -> %d, variables %d -> %d)\n",
                                             ncert, nlz, nkz, nkr, m, F.P.m, n, F.P.n);
    SNOTE("facial reduction: %d rows show that %ld nonnegative variables and %ld cones are zero and %ld cones lie on one ray in every feasible point (rows %d -> %d, variables %d -> %d)", ncert, nlz, nkz, nkr, m, F.P.m, n, F.P.n);
    SocpRes R2; memset(&R2, 0, sizeof R2);
    const int st = F.P.n == 0 ? fr_solve_inner(&F.P, opt, &R2, t0) : inner(&F.P, opt, &R2, t0);
    if (st == -77) {     /* (SOCP_DUAL_UNUSED: asked only whether the dual form is taken, and it is not: nothing solved; the caller goes on with the problem as given, and the search runs again where it is solved) */
        fr_face_free(&F); free(rp); free(rj); free(rx); free(cone); free(ks); free(kd); free(zc); free(rdel); free(kf); free(ray); free(cert); free(mark);
        *done = 0; return 0; }
    *R = R2; R->x = (double *)calloc((size_t)n + 1, sizeof(double)); R->z = (double *)calloc((size_t)n + 1, sizeof(double)); R->y = (double *)calloc((size_t)m + 1, sizeof(double));
    const int cdual = st == SOCP_PINF;      /* (a primal infeasibility certificate: the dual ray, with c = 0) */
    if (R2.x) fr_face_x(P, ks, kd, ray, &F, R2.x, R->x);
    if (R2.y) for (int i = 0; i < m; i++) if (F.rmap[i] >= 0) R->y[i] = R2.y[F.rmap[i]];
    if (R2.z) for (int j = 0; j < n; j++) if (F.cmap[j] >= 0) R->z[j] = R2.z[F.cmap[j]];
    if (st != SOCP_DINF) {     /* the removed rows' multipliers, the last reduction first */
        int *bk = (int *)malloc(sizeof(int) * ((size_t)n + 1)), *bdim = (int *)malloc(sizeof(int) * ((size_t)n + 1)), *bj0 = (int *)malloc(sizeof(int) * ((size_t)n + 1));
        double **bv = (double **)malloc(sizeof(double *) * ((size_t)n + 1)), **ba = (double **)malloc(sizeof(double *) * ((size_t)n + 1));
        for (int k = 0; k < nk; k++) mark[k] = -1;
        double ysc = 1; for (int i = 0; i < m; i++) ysc = fmax(ysc, fabs(R->y[i]));
        for (int c_ = ncert - 1; c_ >= 0; c_--) {
            const int i = cert[c_]; int nb = 0;
            for (int p = rp[i]; p < rp[i + 1]; p++) {
                const int j = rj[p]; if (rx[p] == 0.0) continue;
                int j0, d, kind;
                if (j < nf + nl) { j0 = j; d = 1; kind = 0; }
                else { const int k = cone[j]; if (mark[k] == c_) continue; mark[k] = c_; j0 = ks[k]; d = kd[k]; kind = k < nq ? 1 : 2; }
                bk[nb] = kind; bdim[nb] = d; bj0[nb] = j0; bv[nb] = (double *)malloc(sizeof(double) * (size_t)d); ba[nb] = (double *)calloc((size_t)d, sizeof(double));
                for (int t = 0; t < d; t++) { const int jj = j0 + t; double v = cdual ? 0.0 : P->c[jj];
                    for (int q = P->Ap[jj]; q < P->Ap[jj + 1]; q++) { if (P->Ai[q] == i) ba[nb][t] += P->Ax[q]; else v -= P->Ax[q] * R->y[P->Ai[q]]; } bv[nb][t] = v; }
                nb++;
            }
            R->y[i] = fr_golden(nb, bk, bdim, (const double *const *)bv, (const double *const *)ba, 1e8 * ysc);
            ysc = fmax(ysc, fabs(R->y[i]));
            for (int b = 0; b < nb; b++) { free(bv[b]); free(ba[b]); }
        }
        free(bk); free(bdim); free(bj0); free(bv); free(ba);
        for (int j = nf; j < n; j++) if (F.cmap[j] < 0) { double v = cdual ? 0.0 : P->c[j]; for (int q = P->Ap[j]; q < P->Ap[j + 1]; q++) v -= P->Ax[q] * R->y[P->Ai[q]]; R->z[j] = v; }
    }
    free(R2.x); free(R2.y); free(R2.z); free(R2.cx); free(R2.cy); free(R2.cz); R->cx = R->cy = R->cz = NULL;
    fr_face_free(&F);
    free(rp); free(rj); free(rx); free(cone); free(ks); free(kd); free(zc); free(rdel); free(kf); free(ray); free(cert); free(mark);
    *done = 1;
    return st;
}
/* the measure of a point on the problem as given: the four relative errors of the DIMACS measure and the cones' violations (x in K, z in K*,
 * relative to 1 + |b| and 1 + |c|); the largest */
static double fr_measure_parts(const SocpProb *P, const double *x, const double *y, const double *z, double *pobj, double *dobj, double *eprim) {
    const int m = P->m, n = P->n, nf = P->nf, nl = P->nl;
    double po = 0, dob = 0, nb = 0, nc = 0, rp = 0, rd = 0, xz = 0, vx = 0, vz = 0;
    double *ra = (double *)malloc(sizeof(double) * ((size_t)m + 1));
    for (int i = 0; i < m; i++) { ra[i] = -P->b[i]; dob += P->b[i] * y[i]; nb += P->b[i] * P->b[i]; }
    for (int j = 0; j < n; j++) {
        double s_ = 0; for (int p = P->Ap[j]; p < P->Ap[j + 1]; p++) { ra[P->Ai[p]] += P->Ax[p] * x[j]; s_ += P->Ax[p] * y[P->Ai[p]]; }
        const double r = P->c[j] - s_ - z[j]; rd += r * r; po += P->c[j] * x[j]; nc += P->c[j] * P->c[j]; if (j >= nf) xz += x[j] * z[j];
    }
    for (int i = 0; i < m; i++) rp += ra[i] * ra[i];
    free(ra);
    for (int j = nf; j < nf + nl; j++) { vx = fmax(vx, -x[j]); vz = fmax(vz, -z[j]); }
    int j0 = nf + nl;
    for (int k = 0; k < P->nq + P->nr; k++) {
        const int d = k < P->nq ? P->q[k] : P->r[k - P->nq];
        vx = fmax(vx, fr_block_viol(k < P->nq ? 1 : 2, d, x + j0, x + j0, 0.0)); vz = fmax(vz, fr_block_viol(k < P->nq ? 1 : 2, d, z + j0, z + j0, 0.0));
        j0 += d;
    }
    const double den = 1.0 + fabs(po) + fabs(dob);
    if (pobj) *pobj = po; if (dobj) *dobj = dob;
    if (eprim) *eprim = fmax(sqrt(rp) / (1.0 + sqrt(nb)), vx / (1.0 + sqrt(nb)));     /* (the primal part: residual and cone membership) */
    return fmax(fmax(fmax(sqrt(rp) / (1.0 + sqrt(nb)), sqrt(rd) / (1.0 + sqrt(nc))), fmax(fabs(po - dob) / den, fabs(xz) / den)),
                fmax(vx / (1.0 + sqrt(nb)), vz / (1.0 + sqrt(nc))));
}
static double fr_measure(const SocpProb *P, const double *x, const double *y, const double *z, double *pobj, double *dobj) { return fr_measure_parts(P, x, y, z, pobj, dobj, NULL); }
/* the face read from a failed solve's last iterate. When a problem has no strictly feasible point, the homogeneous model's tau and kappa
 * go to zero together and the iterate's y grows along a certificate of the face (Permenter, Friberg and Andersen, SIAM J. Optim. 27
 * (2017)): y_c = -y / |y|, s = A'y_c. Accepted when s is zero on the free variables (to 1e-5 of its largest entry), in the dual cone (to
 * 1e-5), and b'y_c = 0 (to 1e-6 of the sum of |b_i y_c,i|); the face: s_j > 1e-6 max|s| zero, a cone with s_0 - |sbar| > 1e-3 s_0 zero,
 * one with s_0 > 1e-6 max|s| otherwise on its ray; one row of the combination, implied on the face, removed (the largest |y_c,i| among the
 * rows without a free variable). Solved again (the step may repeat, at most three deep). Back: x on the face; y the face's (the removed row
 * 0) plus theta y_c, theta by the search for the smallest cone violation of z = c - A'y over the blocks s touches. Kept when its measure on
 * the problem as given is better than the first solve's. */
static int fr_face(const SocpProb *P, const SocpOpts *opt, SocpRes *R, double t0, int st, int solve) {
    if (g_fr_depth >= 3 || !R->y || !(g_fr_tau < 1e-3)) return st;
    const int m = P->m, n = P->n, nf = P->nf, nl = P->nl, nq = P->nq, nr = P->nr, nk = nq + nr;
    double ny = 0; for (int i = 0; i < m; i++) ny = fmax(ny, fabs(R->y[i])); if (!(ny > 0) || !isfinite(ny)) return st;
    double *yc = (double *)malloc(sizeof(double) * ((size_t)m + 1)), *sv = (double *)calloc((size_t)n + 1, sizeof(double));
    for (int i = 0; i < m; i++) yc[i] = -R->y[i] / ny;
    double beta = 0, bsc = 0; for (int i = 0; i < m; i++) { beta += P->b[i] * yc[i]; bsc += fabs(P->b[i] * yc[i]); }
    double smax = 0; for (int j = 0; j < n; j++) { double v = 0; for (int p = P->Ap[j]; p < P->Ap[j + 1]; p++) v += P->Ax[p] * yc[P->Ai[p]]; sv[j] = v; if (j >= nf) smax = fmax(smax, fabs(v)); }
    double sfree = 0; for (int j = 0; j < nf; j++) sfree = fmax(sfree, fabs(sv[j]));
    const double del = 1e-6 * smax;
    int ok = smax > 0 && sfree <= 1e-5 * smax && fabs(beta) <= 1e-6 * fmax(bsc, smax);
    for (int j = nf; j < n; j++) if (fabs(sv[j]) <= del) sv[j] = 0.0;     /* (the certificate's accuracy: entries at the level that counts as zero (1e-6 of the largest; the iterate's noise was 1e-7 on socp_touch) are zero, so that a ray has no stray entries) */
    unsigned char *zc = (unsigned char *)calloc((size_t)n + 1, 1), *rdel = (unsigned char *)calloc((size_t)m + 1, 1); signed char *kf = (signed char *)calloc((size_t)nk + 1, 1);
    double *ray = (double *)calloc((size_t)n + 1, sizeof(double)); long nlz = 0, nkz = 0, nkr = 0;
    for (int j = nf; ok && j < nf + nl; j++) { if (sv[j] < -10 * del) ok = 0; else if (sv[j] > del) { zc[j] = 1; nlz++; } }
    int *ks = (int *)malloc(sizeof(int) * ((size_t)nk + 1)), *kd = (int *)malloc(sizeof(int) * ((size_t)nk + 1));
    { int s0 = nf + nl; for (int k = 0; k < nk; k++) { kd[k] = k < nq ? P->q[k] : P->r[k - nq]; ks[k] = s0; s0 += kd[k]; } }
    for (int k = 0; ok && k < nk; k++) {
        const int j0 = ks[k], d = kd[k]; double t2 = 0;
        if (k < nq) { const double s0 = sv[j0]; for (int t = 1; t < d; t++) t2 += sv[j0 + t] * sv[j0 + t]; const double tn = sqrt(t2);
            if (s0 < -10 * del || tn > s0 + 10 * del) { ok = 0; break; }
            if (s0 <= del) continue;
            for (int t = 0; t < d; t++) zc[j0 + t] = 1;
            if (s0 - tn > 1e-3 * s0) { kf[k] = 1; nkz++; }
            else { kf[k] = 2; nkr++; const double rn = sqrt(s0 * s0 + t2); ray[j0] = s0 / rn; for (int t = 1; t < d; t++) ray[j0 + t] = -sv[j0 + t] / rn; }
        } else { const double su = sv[j0], sw = sv[j0 + 1]; for (int t = 2; t < d; t++) t2 += sv[j0 + t] * sv[j0 + t];
            if (su < -10 * del || sw < -10 * del || 2.0 * fmax(su, 0) * fmax(sw, 0) < t2 - 10 * del * smax) { ok = 0; break; }
            if (su <= del && sw <= del) continue;
            for (int t = 0; t < d; t++) zc[j0 + t] = 1;
            if (2.0 * su * sw - t2 > 1e-3 * 0.5 * (su + sw) * (su + sw)) { kf[k] = 1; nkz++; }
            else { kf[k] = 2; nkr++; const double rn = sqrt(su * su + sw * sw + t2); ray[j0] = sw / rn; ray[j0 + 1] = su / rn; for (int t = 2; t < d; t++) ray[j0 + t] = -sv[j0 + t] / rn; }
        }
    }
    int istar = -1;
    if (ok && nlz + nkz + nkr > 0) {
        unsigned char *hasf = (unsigned char *)calloc((size_t)m + 1, 1); for (int j = 0; j < nf; j++) for (int p = P->Ap[j]; p < P->Ap[j + 1]; p++) hasf[P->Ai[p]] = 1;
        double best = 0; for (int i = 0; i < m; i++) if (!hasf[i] && fabs(yc[i]) > best) { best = fabs(yc[i]); istar = i; }
        if (best < 1e-3) istar = -1;
        free(hasf);
    }
    if (opt && opt->verbose > 0) socp_printf("facial reduction after the solve (depth %d): tau %.1e; certificate %s (|b'y| %.1e of %.1e, free part %.1e of %.1e): %ld nonnegative variables and %ld cones zero, %ld cones on a ray\n",
        g_fr_depth + 1, g_fr_tau, ok && istar >= 0 ? "accepted" : "not accepted", fabs(beta), fmax(bsc, smax), sfree, smax, nlz, nkz, nkr);
    int st_out = st;
    if (!solve) {      /* (the facial reduction off: the face is reported, with the option that solves it, in the Summary) */
        if (ok && istar >= 0 && nlz + nkz + nkr > 0 && g_fr_depth == 0) { g_fr_adv[0] = nlz; g_fr_adv[1] = nkz; g_fr_adv[2] = nkr; }
        free(yc); free(sv); free(zc); free(rdel); free(kf); free(ray); free(ks); free(kd);
        return st;
    }
    if (ok && istar >= 0 && nlz + nkz + nkr > 0) {
        rdel[istar] = 1;
        FrFace F; fr_face_build(P, ks, kd, zc, kf, ray, rdel, &F);
        SocpRes R3; memset(&R3, 0, sizeof R3); int d3 = 0;
        g_fr_depth++; int st3 = socp_solve_core_fr(&F.P, opt, &R3, t0, &d3); if (!d3) st3 = fr_solve_inner(&F.P, opt, &R3, t0); g_fr_depth--;
        if (R3.x && R3.y && R3.z && st3 != SOCP_PINF && st3 != SOCP_DINF) {
            double *x = (double *)calloc((size_t)n + 1, sizeof(double)), *y = (double *)calloc((size_t)m + 1, sizeof(double)), *z = (double *)calloc((size_t)n + 1, sizeof(double));
            fr_face_x(P, ks, kd, ray, &F, R3.x, x);
            for (int i = 0; i < m; i++) y[i] = F.rmap[i] >= 0 ? R3.y[F.rmap[i]] : 0.0;
            /* theta: z(theta) = z0 - theta s over the blocks s touches */
            int nb = 0; int *bk = (int *)malloc(sizeof(int) * ((size_t)nl + nk + 1)), *bdim = (int *)malloc(sizeof(int) * ((size_t)nl + nk + 1));
            double **bv = (double **)malloc(sizeof(double *) * ((size_t)nl + nk + 1)), **ba = (double **)malloc(sizeof(double *) * ((size_t)nl + nk + 1));
            double *z0 = (double *)malloc(sizeof(double) * ((size_t)n + 1));
            for (int j = 0; j < n; j++) { double v = P->c[j]; for (int p = P->Ap[j]; p < P->Ap[j + 1]; p++) v -= P->Ax[p] * y[P->Ai[p]]; z0[j] = v; }
            for (int j = nf; j < nf + nl; j++) if (sv[j] != 0.0) { bk[nb] = 0; bdim[nb] = 1; bv[nb] = z0 + j; ba[nb] = sv + j; nb++; }
            for (int k = 0; k < nk; k++) { int on_ = 0; for (int t = 0; t < kd[k]; t++) if (sv[ks[k] + t] != 0.0) on_ = 1; if (on_) { bk[nb] = k < nq ? 1 : 2; bdim[nb] = kd[k]; bv[nb] = z0 + ks[k]; ba[nb] = sv + ks[k]; nb++; } }
            double ysc = 1; for (int i = 0; i < m; i++) ysc = fmax(ysc, fabs(y[i]));
            const double th = fr_golden(nb, bk, bdim, (const double *const *)bv, (const double *const *)ba, 1e8 * ysc);
            for (int i = 0; i < m; i++) y[i] += th * yc[i];
            for (int j = 0; j < n; j++) { double v = P->c[j]; for (int p = P->Ap[j]; p < P->Ap[j + 1]; p++) v -= P->Ax[p] * y[P->Ai[p]]; z[j] = j < nf ? 0.0 : v; }     /* (exact: the cleaned certificate's dropped entries times theta would show as a residual) */
            free(bk); free(bdim); free(bv); free(ba); free(z0);
            /* (judged as the SDP solver's facial reduction is: the primal on the problem as given, the dual, the gap and complementarity on
             * the face, whose multipliers exist; the problem as given has none where it has no strictly feasible point, so a rebuilt y can
             * only approach them) */
            double po3, do3, po1 = 0, do1, ep3 = 0, ef3 = 0;
            const double ea3 = fr_measure_parts(P, x, y, z, &po3, &do3, &ep3), e1 = R->x && R->z ? fr_measure(P, R->x, R->y, R->z, &po1, &do1) : INFINITY;
            ef3 = fr_measure(&F.P, R3.x, R3.y, R3.z, NULL, NULL);
            const double e3 = fmax(ep3, ef3);
            if (opt && opt->verbose > 0) socp_printf("facial reduction after the solve (depth %d): on the face %s in %d iterations; its measure %.1e (the primal on the problem as given %.1e, the face's own %.1e; the dual rebuilt on the problem as given %.1e), objective %.10e, against the first solve's %.1e (%.10e)\n",
                g_fr_depth + 1, socp_status_str(st3), R3.iters, e3, ep3, ef3, ea3, po3, e1, po1);
            if (e3 < e1) {
                free(R->x); free(R->y); free(R->z); R->x = x; R->y = y; R->z = z; x = y = z = NULL;
                R->iters += R3.iters; R->pobj = po3; R->dobj = do3; R->maxerr = e3;
                st_out = e3 <= (opt ? opt->tol : 1e-8) ? SOCP_OPTIMAL : (st3 == SOCP_OPTIMAL ? SOCP_REDUCED : st3); R->status = st_out;
                { char wh[200]; int o_ = 0; wh[0] = 0;
                  if (nlz) o_ += snprintf(wh + o_, sizeof wh - o_, "%ld nonnegative variables zero, ", nlz);
                  if (nkz) o_ += snprintf(wh + o_, sizeof wh - o_, "%ld cones zero, ", nkz);
                  if (nkr) o_ += snprintf(wh + o_, sizeof wh - o_, "%ld cones on one ray, ", nkr);
                  if (o_ >= 2) wh[o_ - 2] = 0;
                  SNOTE("facial reduction: no strictly feasible point; the first solve shows %s in every feasible point; solved again on that face (its dual errors %.1e; the problem as given has no optimal multipliers)", wh, ef3); }
                g_fr_face_err = e3;
            }
            free(x); free(y); free(z);
        }
        free(R3.x); free(R3.y); free(R3.z); free(R3.cx); free(R3.cy); free(R3.cz);
        fr_face_free(&F);
    }
    free(yc); free(sv); free(zc); free(rdel); free(kf); free(ray); free(ks); free(kd);
    return st_out;
}
static int socp_solve_core_fr(const SocpProb *P, const SocpOpts *opt, SocpRes *R, double t0, int *done) {
    /* (11 November 2026, the owner: off by default; -conefr 1 solves the face a failed solve shows, -conefr 2 also removes the faces single rows
     * prove before the solve; off, a failed solve whose last iterate shows a face is told so in the Summary, with the option) */
    const int on = fr_mode(opt);
    *done = 0;
    if (opt && opt->wx) return 0;
    if (!on) {
        if (P->n == 0 || g_fr_depth > 0) return 0;
        int st = fr_solve_inner(P, opt, R, t0); *done = 1;
        if (st != SOCP_OPTIMAL && st != SOCP_PINF && st != SOCP_DINF && st != SOCP_TIME && st != SOCP_INPUT && st != SOCP_CUTOFF) fr_face(P, opt, R, t0, st, 0);
        return st;
    }
    int d1 = 0; int st = (g_fr1_outer || on == 3) ? 0 : socp_solve_core_fr1(P, opt, R, t0, &d1, fr_solve_inner);     /* (3: the face read after a failed solve only) */     /* (done already before the choice of form: the problem here is the reduced one) */
    if (!d1) st = fr_solve_inner(P, opt, R, t0);
    *done = 1;
    if (on >= 2 && st != SOCP_OPTIMAL && st != SOCP_PINF && st != SOCP_DINF && st != SOCP_TIME && st != SOCP_INPUT && st != SOCP_CUTOFF) st = fr_face(P, opt, R, t0, st, 1);
    return st;
}
static int socp_solve_core(const SocpProb *P, const SocpOpts *opt, SocpRes *R) {
    const double t0 = sc_time();
    extern void schol_set_analysis_stop(const volatile int *f, double deadline); extern void schol_get_analysis_stop(const volatile int **f, double *deadline);
    extern void nd_set_stop_deadline(double t); extern double nd_get_stop_deadline(void);
    const volatile int *anf0 = NULL; double andl0 = 0; schol_get_analysis_stop(&anf0, &andl0); const double nddl0 = nd_get_stop_deadline();
    const double dl = opt && opt->timelimit > 0 ? t0 + opt->timelimit : 0;
    if (dl > 0) {
        schol_set_analysis_stop(anf0, andl0 > 0 ? fmin(andl0, dl) : dl);
        nd_set_stop_deadline(nddl0 > 0 ? fmin(nddl0, dl) : dl);
    }
    int frd = 0; int st = socp_solve_core_fr(P, opt, R, t0, &frd);
    if (!frd) { int boxed = 0; st = socp_solve_core_box(P, opt, R, t0, &boxed);
    if (!boxed) st = socp_solve_core_(P, opt, R, t0); }
    if (dl > 0) { schol_set_analysis_stop(anf0, andl0); nd_set_stop_deadline(nddl0); }
    return st;
}
/* the result of a setup stopped by the time limit: TIME LIMIT after 0 iterations, the point 0 */
static int socp_setup_timeout(const SocpProb *P, SocpRes *R) {
    R->x = (double *)xcalloc((size_t)P->n + 1, sizeof(double)); R->y = (double *)xcalloc((size_t)P->m + 1, sizeof(double)); R->z = (double *)xcalloc((size_t)P->n + 1, sizeof(double));
    R->iters = 0; R->status = SOCP_TIME;
    SUSER("the SOCP solver: the time limit was reached during the setup, before the first iteration\n");
    return SOCP_TIME;
}
/* (3 November 2026, the SOCP developer, a measurement: BRISK_CONENEMEASURE=1) AMD's count of the factor (entries of L and
 * multiply-subtract pairs) for three forms of the same system: the full system (rows and every column, a cone's columns a
 * clique), the normal equations of the non-free columns (each linear column or cone joins its rows into a clique), and the
 * same with the free columns as a border of unknowns joined to their rows. The edges as pairs, sorted and deduplicated per
 * vertex */
static _Thread_local double g_nem_lnz = 0;
extern __thread const volatile double *brisk_amd_cap; extern __thread int brisk_amd_capped;     /* (amd/amd_2.c: a per-thread cap on the ordering's count) */
static int nem_cmpi(const void *a, const void *b) { const int x = *(const int *)a, y = *(const int *)b; return (x > y) - (x < y); }
static double nem_amd(const char *name, int nv, long np, const int *pa, const int *pb) {
    int *cnt = (int *)calloc((size_t)nv + 1, sizeof(int));
    for (long e = 0; e < np; e++) { cnt[pa[e] + 1]++; cnt[pb[e] + 1]++; }
    for (int v = 0; v < nv; v++) cnt[v + 1] += cnt[v];
    int *adj = (int *)malloc(sizeof(int) * (size_t)(2 * np + 1)); int *fill = (int *)malloc(sizeof(int) * (size_t)nv);
    for (int v = 0; v < nv; v++) fill[v] = cnt[v];
    for (long e = 0; e < np; e++) { adj[fill[pa[e]]++] = pb[e]; adj[fill[pb[e]]++] = pa[e]; }
    int *Ap = (int *)malloc(sizeof(int) * ((size_t)nv + 1)); long w = 0; Ap[0] = 0;
    for (int v = 0; v < nv; v++) {      /* (sort and dedup the row's neighbours in place) */
        int a = cnt[v], b = fill[v];
        if (b - a > 32) qsort(adj + a, (size_t)(b - a), sizeof(int), nem_cmpi);      /* (a dense row: insertion sort is quadratic in its degree, socp_rocket_4000 16 s) */
        else for (int i = a + 1; i < b; i++) { int x = adj[i], j = i - 1; while (j >= a && adj[j] > x) { adj[j + 1] = adj[j]; j--; } adj[j + 1] = x; }
        int last = -1; for (int i = a; i < b; i++) if (adj[i] != last) { adj[w++] = adj[i]; last = adj[i]; }
        Ap[v + 1] = (int)w;
    }
    int *Pm = (int *)malloc(sizeof(int) * (size_t)nv); double ctl[AMD_CONTROL], info[AMD_INFO]; amd_defaults(ctl);
    const double t_ = sc_time(); const int st = amd_order(nv, Ap, adj, Pm, ctl, info);
    if (brisk_amd_capped) { brisk_amd_capped = 0; free(cnt); free(adj); free(fill); free(Ap); free(Pm); if (name) socp_printf("  [the measure, %s: stopped at the cap]\n", name); g_nem_lnz = 0; return 1e300; }
    if (name) socp_printf("  [the measure, %s: %d unknowns, %ld entries of the graph; AMD %s in %.2fs: nnz(L) %.4g, multiply-subtract pairs %.4g, %.0f dense]\n", name, nv, w, st < 0 ? "failed" : "done", sc_time() - t_, info[AMD_LNZ], info[AMD_NMULTSUBS_LDL], info[AMD_NDENSE]);
    free(cnt); free(adj); free(fill); free(Ap); free(Pm);
    g_nem_lnz = st < 0 ? 0 : info[AMD_LNZ];
    return st < 0 ? 1e300 : info[AMD_NMULTSUBS_LDL];
}
static void nem_measure(const Socp *S) {
    const int m = S->m, n = S->n, nf = S->nf;
    /* the groups: each linear column alone, each cone's columns together */
    int ng = S->nl + S->nq; int *gs = (int *)malloc(sizeof(int) * ((size_t)ng + 1)); int g = 0;
    for (int i = 0; i < S->nl; i++) gs[g++] = nf + i; for (int k = 0; k < S->nq; k++) gs[g++] = S->qs[k]; gs[g] = n;
    int *mk = (int *)malloc(sizeof(int) * (size_t)m); for (int i = 0; i < m; i++) mk[i] = -1; int *rl = (int *)malloc(sizeof(int) * (size_t)m);
    long np = 0; for (g = 0; g < ng; g++) { int r = 0; for (int j = gs[g]; j < gs[g + 1]; j++) for (int p = S->Ap[j]; p < S->Ap[j + 1]; p++) if (mk[S->Ai[p]] != g) { mk[S->Ai[p]] = g; r++; } np += (long)r * (r - 1) / 2; }
    const long nfree = S->Ap[nf];
    int *pa = (int *)malloc(sizeof(int) * (size_t)(np + nfree + 1)), *pb = (int *)malloc(sizeof(int) * (size_t)(np + nfree + 1));
    for (int i = 0; i < m; i++) mk[i] = -1; long e = 0;
    for (g = 0; g < ng; g++) { int r = 0; for (int j = gs[g]; j < gs[g + 1]; j++) for (int p = S->Ap[j]; p < S->Ap[j + 1]; p++) if (mk[S->Ai[p]] != g) { mk[S->Ai[p]] = g; rl[r++] = S->Ai[p]; }
        for (int a = 0; a < r; a++) for (int b = a + 1; b < r; b++) { pa[e] = rl[a]; pb[e] = rl[b]; e++; } }
    socp_printf("  [the measure: %d groups (linear columns and cones), %ld pairs of rows in the normal equations, %d free columns with %ld entries]\n", ng, np, nf, nfree);
    nem_amd("the normal equations of the non-free columns", m, e, pa, pb);
    for (int j = 0; j < nf; j++) for (int p = S->Ap[j]; p < S->Ap[j + 1]; p++) { pa[e] = m + j; pb[e] = S->Ai[p]; e++; }
    nem_amd("the normal equations with the free columns as a border", m + nf, e, pa, pb);
    free(pa); free(pb);
    /* the full system: rows m.., columns 0..n-1; A's entries and each cone's clique */
    long nfull = S->Ap[n]; for (int k = 0; k < S->nq; k++) nfull += (long)S->q[k] * (S->q[k] - 1) / 2;
    pa = (int *)malloc(sizeof(int) * (size_t)(nfull + 1)); pb = (int *)malloc(sizeof(int) * (size_t)(nfull + 1)); e = 0;
    for (int j = 0; j < n; j++) for (int p = S->Ap[j]; p < S->Ap[j + 1]; p++) { pa[e] = j; pb[e] = n + S->Ai[p]; e++; }
    for (int k = 0; k < S->nq; k++) for (int a = 0; a < S->q[k]; a++) for (int b = a + 1; b < S->q[k]; b++) { pa[e] = S->qs[k] + a; pb[e] = S->qs[k] + b; e++; }
    nem_amd("the full system", n + m, e, pa, pb);
    free(pa); free(pb); free(gs); free(mk); free(rl);
}
/* (3 November 2026, the SOCP developer, a test: BRISK_CONEPOLISH=1) polishing on the optimal face. Near the solution each
 * piece of the iterate shows which side of its complementarity holds: a linear variable x_j > z_j keeps x_j and has z_j = 0;
 * a cone's x and z are compared along the two eigen-directions (1, u) and (1, -u) of x (u = xbar / |xbar|): x larger along
 * both, the cone's x is interior and z = 0; z larger along both, x = 0; one each, x lies on the ray (1, +-u) and z on the
 * opposite ray, so x'z = 0. Fixing that structure leaves linear equations in the pieces that carry x: A_P xi = b for the
 * primal, A_P' y = c_P for the dual (z = c - A'y), with no H in them, so their conditioning is A_P's, not the 1e18 of the
 * interior-point system near the end. One factorization of K = [-I A_P'; A_P d I] (quasi-definite, d small) gives the
 * least-norm primal correction (K [a; l] = [0; b - A_P xi0], a = A_P' l) and the regularized least-norm dual one
 * (K [s; t] = [c_P - A_P' y0; 0], dy = t); x and z are then projected on their cones and the solver's own test judges the
 * point. In the solver's scaled space (S->Ax, S->b, S->c; every cone a second-order cone there). Returns 0 with the point
 * in xs, ys, zs; -1 when it does not apply (separate dense columns) or the factorization fails */
static double pol_dot(int d, const double *a, const double *b) { double s = 0; for (int i = 0; i < d; i++) s += a[i] * b[i]; return s; }
static void pol_proj(int d, double *v) {      /* the projection on the second-order cone of dimension d (d = 1: the half line) */
    if (d == 1) { if (v[0] < 0) v[0] = 0; return; }
    double nb = 0; for (int i = 1; i < d; i++) nb += v[i] * v[i]; nb = sqrt(nb);
    if (v[0] >= nb) return;
    if (v[0] <= -nb) { for (int i = 0; i < d; i++) v[i] = 0; return; }
    const double a = 0.5 * (v[0] + nb), f = nb > 0 ? a / nb : 0; v[0] = a; for (int i = 1; i < d; i++) v[i] *= f;
}
static int socp_polish(Socp *S, double *xs, double *ys, double *zs, int *npiece, const double *hx, const double *hz) {
    const int n = S->n, m = S->m, nf = S->nf, nl = S->nl;
    if (S->nd > 0 || !(S->tau > 0)) return -1;
    const double ti = 1.0 / S->tau, dlt = 1e-12;
    const double t0_ = sc_time();
    int *pp = (int *)malloc(sizeof(int) * ((size_t)n + 2)), *pj = (int *)malloc(sizeof(int) * ((size_t)n + 1));
    double *pw = (double *)malloc(sizeof(double) * ((size_t)n + 1)), *xi0 = (double *)malloc(sizeof(double) * ((size_t)n + 1)), *cpv = (double *)malloc(sizeof(double) * ((size_t)n + 1));
    int np = 0; long ne = 0; pp[0] = 0;
    /* the split between the pieces that carry x and those that carry z, scale-free: near the solution log(x/z) is about
     * log(x*^2 / mu) for the first and log(mu / z*^2) for the second, two groups far apart; the threshold by two-means
     * on those logarithms (the linear variables and the cones' two eigen-components together) */
    double lthr = 0.0;
    {   long nv = 0; double *lv = (double *)malloc(sizeof(double) * ((size_t)nl + 2 * (size_t)S->nq + 1)); double uu[64];
        for (int j = nf; j < nf + nl; j++) lv[nv++] = log(fmax(S->x[j], 1e-300) / fmax(S->z[j], 1e-300));
        for (int k = 0; k < S->nq; k++) { const int d = S->q[k]; const double *x = S->x + S->qs[k], *z = S->z + S->qs[k];
            if (d == 1) { lv[nv++] = log(fmax(x[0], 1e-300) / fmax(z[0], 1e-300)); continue; } if (d > 64) continue;
            double nx = 0, nz = 0; for (int i = 1; i < d; i++) { nx += x[i] * x[i]; nz += z[i] * z[i]; } nx = sqrt(nx); nz = sqrt(nz);
            if (nx > 0) { for (int i = 1; i < d; i++) uu[i] = x[i] / nx; } else if (nz > 0) { for (int i = 1; i < d; i++) uu[i] = -z[i] / nz; } else { for (int i = 1; i < d; i++) uu[i] = i == 1; }
            const double ux = pol_dot(d - 1, uu + 1, x + 1), uz = pol_dot(d - 1, uu + 1, z + 1);
            lv[nv++] = log(fmax(x[0] + ux, 1e-300) / fmax(z[0] + uz, 1e-300)); lv[nv++] = log(fmax(x[0] - ux, 1e-300) / fmax(z[0] - uz, 1e-300)); }
        if (nv > 0) { double lo = lv[0], hi = lv[0]; for (long q = 0; q < nv; q++) { lo = fmin(lo, lv[q]); hi = fmax(hi, lv[q]); }
            lthr = 0.5 * (lo + hi);
            for (int r = 0; r < 50; r++) { double s0 = 0, s1 = 0; long c0 = 0, c1 = 0; for (long q = 0; q < nv; q++) { if (lv[q] > lthr) { s1 += lv[q]; c1++; } else { s0 += lv[q]; c0++; } }
                if (!c0 || !c1) break; const double t_ = 0.5 * (s0 / c0 + s1 / c1); if (fabs(t_ - lthr) < 1e-9) break; lthr = t_; }
            if (dbg_on()) { long c1 = 0; for (long q = 0; q < nv; q++) c1 += lv[q] > lthr; socp_printf("      [polishing: log(x/z) from %.1f to %.1f, the split at %.1f: %ld of %ld carry x]\n", lo, hi, lthr, c1, nv); } }
        free(lv); }
    const double ethr = exp(lthr);
    #define POL_ADD1(j_) do { pj[ne] = (j_); pw[ne] = 1.0; ne++; xi0[np] = S->x[j_] * ti; cpv[np] = S->c[j_]; np++; pp[np] = (int)ne; } while (0)
    for (int j = 0; j < nf; j++) POL_ADD1(j);
    /* (with an earlier iterate hx, hz (divided by its tau), mu then 10 to 100 times larger: the trend decides, scale-free: a
     * piece that carries x keeps its size and one that goes to zero shrinks with mu; x_now / x_then against z_now / z_then) */
    #define POL_XACT(xn_, zn_, xh_, zh_) (hx ? (xn_) * fmax((zh_), 1e-300) > (zn_) * fmax((xh_), 1e-300) : (xn_) > ethr * (zn_))
    for (int j = nf; j < nf + nl; j++) if (POL_XACT(S->x[j] * ti, S->z[j] * ti, hx ? hx[j] : 0, hz ? hz[j] : 0)) POL_ADD1(j);
    double u[64];
    for (int k = 0; k < S->nq; k++) {
        const int d = S->q[k], s0 = S->qs[k];
        const double *x = S->x + s0, *z = S->z + s0;
        if (d == 1) { if (POL_XACT(x[0] * ti, z[0] * ti, hx ? hx[s0] : 0, hz ? hz[s0] : 0)) POL_ADD1(s0); continue; }
        if (d > 64) { for (int i = 0; i < d; i++) POL_ADD1(s0 + i); continue; }    /* (a large cone: kept whole, as interior) */
        double nx = 0, nz = 0; for (int i = 1; i < d; i++) { nx += x[i] * x[i]; nz += z[i] * z[i]; } nx = sqrt(nx); nz = sqrt(nz);
        if (nx > 0) { for (int i = 1; i < d; i++) u[i] = x[i] / nx; } else if (nz > 0) { for (int i = 1; i < d; i++) u[i] = -z[i] / nz; } else { for (int i = 1; i < d; i++) u[i] = i == 1; }
        const double ux = pol_dot(d - 1, u + 1, x + 1), uz = pol_dot(d - 1, u + 1, z + 1);
        double hux = 0, huz = 0, hx0 = 0, hz0 = 0;
        if (hx) { hux = pol_dot(d - 1, u + 1, hx + s0 + 1); huz = pol_dot(d - 1, u + 1, hz + s0 + 1); hx0 = hx[s0]; hz0 = hz[s0]; }
        const int a1 = POL_XACT((x[0] + ux) * ti, (z[0] + uz) * ti, hx0 + hux, hz0 + huz), a2 = POL_XACT((x[0] - ux) * ti, (z[0] - uz) * ti, hx0 - hux, hz0 - huz);
        if (a1 && a2) { for (int i = 0; i < d; i++) POL_ADD1(s0 + i); }
        else if (a1 || a2) {
            const double sg = a1 ? 1.0 : -1.0, r2 = 1.0 / sqrt(2.0);
            double xv = 0, cv = 0;
            for (int i = 0; i < d; i++) { const double w = (i == 0 ? 1.0 : sg * u[i]) * r2; pj[ne] = s0 + i; pw[ne] = w; ne++; xv += w * x[i]; cv += w * S->c[s0 + i]; }
            xi0[np] = xv * ti; cpv[np] = cv; np++; pp[np] = (int)ne;
        }
    }
    #undef POL_ADD1
    #undef POL_XACT
    *npiece = np;
    /* A_P by columns (each piece's columns merged) */
    long cap = 0; for (int t = 0; t < np; t++) for (long q = pp[t]; q < pp[t + 1]; q++) cap += S->Ap[pj[q] + 1] - S->Ap[pj[q]];
    int *Pp = (int *)malloc(sizeof(int) * ((size_t)np + 1)), *Pi = (int *)malloc(sizeof(int) * ((size_t)cap + 1)); double *Px = (double *)malloc(sizeof(double) * ((size_t)cap + 1));
    int *mk = (int *)malloc(sizeof(int) * ((size_t)m + 1)); double *acc = (double *)calloc((size_t)m + 1, sizeof(double));
    for (int i = 0; i < m; i++) mk[i] = -1;
    long pe = 0; Pp[0] = 0;
    for (int t = 0; t < np; t++) {
        const long st = pe;
        for (long q = pp[t]; q < pp[t + 1]; q++) { const int j = pj[q]; const double w = pw[q];
            for (int p = S->Ap[j]; p < S->Ap[j + 1]; p++) { const int i = S->Ai[p]; if (mk[i] != t) { mk[i] = t; Pi[pe++] = i; acc[i] = 0; } acc[i] += w * S->Ax[p]; } }
        for (long q = st; q < pe; q++) Px[q] = acc[Pi[q]];
        Pp[t + 1] = (int)pe;
    }
    /* K's pattern: piece t joined to its rows (np + i) */
    const int N = np + m;
    int *deg = (int *)calloc((size_t)N + 1, sizeof(int)), **nbr = (int **)malloc(sizeof(int *) * ((size_t)N + 1));
    for (int t = 0; t < np; t++) for (int q = Pp[t]; q < Pp[t + 1]; q++) { deg[t]++; deg[np + Pi[q]]++; }
    int *pool = (int *)malloc(sizeof(int) * (2 * (size_t)pe + 1));
    { long o = 0; for (int v = 0; v < N; v++) { nbr[v] = pool + o; o += deg[v]; deg[v] = 0; } }
    for (int t = 0; t < np; t++) for (int q = Pp[t]; q < Pp[t + 1]; q++) { const int r = np + Pi[q]; nbr[t][deg[t]++] = r; nbr[r][deg[r]++] = t; }
    const int om = schol_set_amd(2);
    SChol *sc = schol_analyze_adj(N, deg, nbr, (size_t)1 << 31);
    schol_set_amd(om);
    free(deg); free(nbr); free(pool);
    int rc = -1;
    double *rhs = NULL, *sol = NULL, *wk = NULL, *res = NULL;
    if (sc) {
        signed char *sg = (signed char *)malloc((size_t)N + 1); for (int v = 0; v < N; v++) sg[v] = v < np ? -1 : 1;
        schol_set_signs(sc, sg); free(sg);
        schol_set_dynpiv(sc, 1e-7);
        schol_zero(sc); double *pm = schol_values(sc);
        for (int t = 0; t < np; t++) pm[schol_offset(sc, (size_t)t, (size_t)t)] = -1.0;
        for (int i = 0; i < m; i++) pm[schol_offset(sc, (size_t)(np + i), (size_t)(np + i))] = dlt;
        for (int t = 0; t < np; t++) for (int q = Pp[t]; q < Pp[t + 1]; q++) pm[schol_offset(sc, (size_t)(np + Pi[q]), (size_t)t)] += Px[q];
        const int fr = schol_factor(sc, 0.0);
        if (fr == 0) {
            rhs = (double *)malloc(sizeof(double) * (size_t)N); sol = (double *)malloc(sizeof(double) * (size_t)N); res = (double *)malloc(sizeof(double) * (size_t)N);
            wk = (double *)malloc(sizeof(double) * (2 * (size_t)N + 2));
            /* K v for the refinement */
            #define POL_KMV(v_, o_) do { for (int t_ = 0; t_ < np; t_++) { double s_ = -(v_)[t_]; for (int q_ = Pp[t_]; q_ < Pp[t_ + 1]; q_++) s_ += Px[q_] * (v_)[np + Pi[q_]]; (o_)[t_] = s_; } \
                for (int i_ = 0; i_ < m; i_++) (o_)[np + i_] = dlt * (v_)[np + i_]; \
                for (int t_ = 0; t_ < np; t_++) { const double a_ = (v_)[t_]; for (int q_ = Pp[t_]; q_ < Pp[t_ + 1]; q_++) (o_)[np + Pi[q_]] += Px[q_] * a_; } } while (0)
            #define POL_SOLVE(b_) do { memcpy(sol, (b_), sizeof(double) * (size_t)N); schol_solve(sc, sol, wk); \
                for (int r_ = 0; r_ < 3; r_++) { POL_KMV(sol, res); double e_ = 0; for (int v_ = 0; v_ < N; v_++) { res[v_] = (b_)[v_] - res[v_]; e_ = fmax(e_, fabs(res[v_])); } \
                    if (!(e_ > 0)) break; schol_solve(sc, res, wk); for (int v_ = 0; v_ < N; v_++) sol[v_] += res[v_]; } } while (0)
            /* primal: the least-norm correction of the pieces */
            double *bb = (double *)malloc(sizeof(double) * (size_t)N);
            for (int t = 0; t < np; t++) bb[t] = 0;
            for (int i = 0; i < m; i++) bb[np + i] = S->b[i];
            for (int t = 0; t < np; t++) for (int q = Pp[t]; q < Pp[t + 1]; q++) bb[np + Pi[q]] -= Px[q] * xi0[t];
            POL_SOLVE(bb);
            double *xi = (double *)malloc(sizeof(double) * ((size_t)np + 1));
            for (int t = 0; t < np; t++) xi[t] = xi0[t] + sol[t];
            /* dual: y0 + dy with A_P' dy = c_P - A_P' y0 */
            for (int t = 0; t < np; t++) { double s_ = cpv[t]; for (int q = Pp[t]; q < Pp[t + 1]; q++) s_ -= Px[q] * S->y[Pi[q]] * ti; bb[t] = s_; }
            for (int i = 0; i < m; i++) bb[np + i] = 0;
            POL_SOLVE(bb);
            for (int i = 0; i < m; i++) ys[i] = S->y[i] * ti + sol[np + i];
            #undef POL_SOLVE
            #undef POL_KMV
            if (dbg_on()) {     /* (diagnostics: the two least-norm systems' residuals before the projection, rows without a piece) */
                double rp_ = 0, nb_ = 0, rd_ = 0, nc_ = 0; long er_ = 0, erb_ = 0;
                double *ax_ = (double *)calloc((size_t)m + 1, sizeof(double));
                for (int t = 0; t < np; t++) for (int q = Pp[t]; q < Pp[t + 1]; q++) ax_[Pi[q]] += Px[q] * xi[t];
                for (int i = 0; i < m; i++) { const double r_ = ax_[i] - S->b[i]; rp_ += r_ * r_; nb_ += S->b[i] * S->b[i]; }
                for (int i = 0; i < m; i++) ax_[i] = 0; for (int t = 0; t < np; t++) for (int q = Pp[t]; q < Pp[t + 1]; q++) ax_[Pi[q]] = 1;
                for (int i = 0; i < m; i++) if (!ax_[i]) { er_++; if (S->b[i] != 0) erb_++; }
                for (int t = 0; t < np; t++) { double s_ = cpv[t]; for (int q = Pp[t]; q < Pp[t + 1]; q++) s_ -= Px[q] * ys[Pi[q]]; rd_ += s_ * s_; nc_ += cpv[t] * cpv[t]; }
                socp_printf("      [polishing: before the projection, |A_P xi - b| / (1 + |b|) %.2e, |c_P - A_P' y| / (1 + |c_P|) %.2e; rows without a piece %ld (%ld with b != 0); pivots replaced %d]\n",
                            sqrt(rp_) / (1 + sqrt(nb_)), sqrt(rd_) / (1 + sqrt(nc_)), er_, erb_, schol_ndyn(sc));
                free(ax_); }
            /* the point: x from the pieces, z = c - A'y, both projected on their cones */
            for (int j = 0; j < n; j++) xs[j] = 0;
            for (int t = 0; t < np; t++) for (long q = pp[t]; q < pp[t + 1]; q++) xs[pj[q]] += pw[q] * xi[t];
            for (int j = 0; j < n; j++) { double s_ = S->c[j]; for (int p = S->Ap[j]; p < S->Ap[j + 1]; p++) s_ -= S->Ax[p] * ys[S->Ai[p]]; zs[j] = s_; }
            if (dbg_on()) {     /* (diagnostics: the pieces outside their cone before the projection) */
                long lxn = 0, lzn = 0, cxo = 0, czo = 0; double lxm = 0, lzm = 0, cxm = 0, czm = 0, xmx = 0, zmx = 0;
                for (int j = nf; j < n; j++) { xmx = fmax(xmx, fabs(xs[j])); zmx = fmax(zmx, fabs(zs[j])); }
                for (int j = nf; j < nf + nl; j++) { if (xs[j] < 0) { lxn++; lxm = fmax(lxm, -xs[j]); } if (zs[j] < 0) { lzn++; lzm = fmax(lzm, -zs[j]); } }
                for (int k = 0; k < S->nq; k++) { const int d = S->q[k], s0 = S->qs[k]; if (d < 2) continue;
                    double a = 0, b = 0; for (int i = 1; i < d; i++) { a += xs[s0 + i] * xs[s0 + i]; b += zs[s0 + i] * zs[s0 + i]; }
                    const double vx = sqrt(a) - xs[s0], vz = sqrt(b) - zs[s0]; if (vx > 0) { cxo++; cxm = fmax(cxm, vx); } if (vz > 0) { czo++; czm = fmax(czm, vz); } }
                socp_printf("      [polishing: outside before the projection: linear x %ld (max %.1e), linear z %ld (max %.1e), cone x %ld (max %.1e), cone z %ld (max %.1e); |x| max %.1e, |z| max %.1e]\n",
                            lxn, lxm, lzn, lzm, cxo, cxm, czo, czm, xmx, zmx); }
            for (int j = nf; j < nf + nl; j++) { if (xs[j] < 0) xs[j] = 0; if (zs[j] < 0) zs[j] = 0; }
            for (int j = 0; j < nf; j++) zs[j] = 0;
            for (int k = 0; k < S->nq; k++) { pol_proj(S->q[k], xs + S->qs[k]); pol_proj(S->q[k], zs + S->qs[k]); }
            free(xi); free(bb);
            rc = 0;
        }
        if (dbg_on()) socp_printf("      [polishing on the optimal face: %d pieces (of %d unknowns), A_P %ld entries, K of %d unknowns, factor %s; %.2fs]\n", np, n, pe, N, fr == 0 ? "done" : "failed", sc_time() - t0_);
        schol_free(sc);
    } else if (dbg_on()) socp_printf("      [polishing on the optimal face: the analysis failed]\n");
    free(rhs); free(sol); free(wk); free(res);
    free(pp); free(pj); free(pw); free(xi0); free(cpv); free(Pp); free(Pi); free(Px); free(mk); free(acc);
    return rc;
}
static _Thread_local int g_meas_ok; static _Thread_local double g_meas_pairs, g_meas_lnz;     /* (socp_solve_elim's count of the full system, BRISK_CONEMEASDEFER) */
static int xzdbg_cmpd(const void *a, const void *b) { const double u = *(const double *)a, v = *(const double *)b; return (u < v) - (u > v); }     /* (descending; the x'z debug of ISSUES 865) */
static int socp_solve_core_(const SocpProb *P, const SocpOpts *opt, SocpRes *R, double t0) {
    SocpOpts od; if (!opt) { socp_default_opts(&od); opt = &od; }
    if (socp_bound_count_at(P, 1)) { memset(R, 0, sizeof *R); R->status = SOCP_INPUT; return SOCP_INPUT; }
    if (!g_boxsoc_in && socp_fr_count(P)) { memset(R, 0, sizeof *R); R->status = SOCP_INPUT; return SOCP_INPUT; }
    Socp S_; Socp *S = &S_;
    memset(S, 0, sizeof(*S));
    memset(g_tag_solves, 0, sizeof g_tag_solves); memset(g_tag_apps, 0, sizeof g_tag_apps); g_tag = 0;
    S->t_begin = t0;
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
        /* (29 October 2026, a test: BRISK_CONEEQCOL1=1; ISSUES 803) the columns first brought to a largest entry of 1 (a cone as one
         * block, the slack columns left), the passes below from there: the passes' fixed point depends on where they start, and
         * this start is the same however the variables were scaled (A D, c D with D constant on each cone), so the scaled data
         * are those of the problem as if its variables had not been scaled; on columns already balanced it changes nothing */
        static int eqc1 = -1; if (eqc1 < 0) eqc1 = getenv("BRISK_CONEEQCOL1") ? atoi(getenv("BRISK_CONEEQCOL1")) : 2;
        /* (29 October 2026, the SOCP developer; ISSUES 803; BRISK_CONEEQCOL1=0: the passes from the matrix as given alone, 1: from
         * the columns brought to 1 alone) the passes reach one of many fixed points, which depends on how the variables were
         * scaled: a control SOCP with its variables scaled by 10^[-3,3] (a cone by one factor) ends with entries spread over
         * 7.5e-7 .. 1 against the original's 3.1e-4 .. 1, and 197 iterations against 9. When the passes from the matrix as given
         * leave the entries widely spread (the mean of log^2 |a_ij| above 4: a typical entry e^2 below its row's and column's
         * largest; this mean is the quantity the passes are meant to make small, Curtis and Reid's objective), they are run
         * again from the columns brought to 1, a start that does not depend on the variables' scaling, and that result is kept
         * when its spread is a tenth smaller (two runs that reach nearly the same point differ by under 5 %, and moving to such
         * a point only changes the rounding). Below 4 the matrix is balanced either way and the start is left as it was. */
        const double teq0_ = sc_time();
        /* (2 November 2026, the SOCP developer) the passes on the threads above a million entries (2013_firL2a, SOCP: 50 million
         * entries, 5.5 s on one thread); BRISK_CONEEQNT=1: one thread */
        int eqnt_ = (S->Ap[n] >= 1000000 && !omp_in_parallel()) ? g_cone_nt : 1;
        { const char *e_ = getenv("BRISK_CONEEQNT"); if (e_ && atoi(e_) > 0 && atoi(e_) < eqnt_) eqnt_ = atoi(e_); }
        double *eqrt_ = eqnt_ > 1 ? (double *)malloc(sizeof(double) * (size_t)eqnt_ * (size_t)m) : NULL; if (!eqrt_) eqnt_ = 1;
        const int ncand = eqc1 >= 2 ? 2 : 1; const size_t nnzA_ = (size_t)S->Ap[n];
        /* (2 November 2026, the SOCP developer) one copy of the entries, not five: the start's entries copied once; the second
         * start runs in that copy and the better result is kept by exchanging the arrays (the same values as the copies gave).
         * 2013_firL2Linfalph (SOCP, 121 million entries): 13.3 s of the equilibration's 17 s went to the copies */
        double *Ax0 = ncand > 1 ? (double *)malloc(sizeof(double) * (nnzA_ + 1)) : NULL, *Axb = NULL; double eqbb_ = 1.0, eqcb_ = 1.0;
        double *DE0 = ncand > 1 ? (double *)malloc(sizeof(double) * ((size_t)m + n + 1)) : NULL, *DEb = ncand > 1 ? (double *)malloc(sizeof(double) * ((size_t)m + n + 1)) : NULL;
        double tcp_ = 0, tps_ = 0, tlg_ = 0; double tq_ = sc_time();
        if (ncand > 1) {
            #pragma omp parallel for schedule(static) num_threads(eqnt_) if (eqnt_ > 1)
            for (int c_ = 0; c_ < 64; c_++) { const size_t p0 = nnzA_ * (size_t)c_ / 64, p1 = nnzA_ * (size_t)(c_ + 1) / 64; memcpy(Ax0 + p0, S->Ax + p0, sizeof(double) * (p1 - p0)); }
            memcpy(DE0, S->Dr, sizeof(double) * m); memcpy(DE0 + m, S->Ec, sizeof(double) * n); }
        tcp_ += sc_time() - tq_;
        double qbest = INFINITY; int cbest = 0, npb = 0; double devb = 0;
        for (int cand = 0; cand < ncand; cand++) {
        if (cand > 0 && !(qbest > 4.0)) break;      /* (the matrix as given balanced: the second start not tried) */
        if (cand > 0) { Axb = S->Ax; S->Ax = Ax0; Ax0 = NULL; memcpy(S->Dr, DE0, sizeof(double) * m); memcpy(S->Ec, DE0 + m, sizeof(double) * n); eqb = eqc = 1.0; }     /* (the best so far kept in Axb) */
        if ((eqc1 == 1 && cand == 0) || cand == 1) {
            #pragma omp parallel for schedule(static) num_threads(eqnt_) if (eqnt_ > 1)
            for (int ch_ = 0; ch_ < ENT_CH; ch_++) for (int j = ent_lo(S->Ap, n, ch_), j1_ = ent_lo(S->Ap, n, ch_ + 1); j < j1_; j++) { double s_ = 0; for (int p = S->Ap[j]; p < S->Ap[j + 1]; p++) { const double v = fabs(S->Ax[p]); if (v > s_) s_ = v; } cn[j] = s_; }
            for (int k = 0; k < S->nq; k++) { double s_ = 0; for (int i = S->qs[k]; i < S->qs[k + 1]; i++) if (cn[i] > s_) s_ = cn[i]; for (int i = S->qs[k]; i < S->qs[k + 1]; i++) cn[i] = s_; }
            for (int j = 0; j < n; j++) cn[j] = (colblk && colblk[j] >= 0) || !(cn[j] > 0) ? 1.0 : 1.0 / cn[j];
            #pragma omp parallel for schedule(static) num_threads(eqnt_) if (eqnt_ > 1)
            for (int ch_ = 0; ch_ < ENT_CH; ch_++) for (int j = ent_lo(S->Ap, n, ch_), j1_ = ent_lo(S->Ap, n, ch_ + 1); j < j1_; j++) { for (int p = S->Ap[j]; p < S->Ap[j + 1]; p++) S->Ax[p] *= cn[j]; S->Ec[j] *= cn[j]; }
        }
        tq_ = sc_time();
        for (int pass = 0; pass < eqpass; pass++) {
            npass_ = pass + 1;
            for (int i = 0; i < m; i++) rn[i] = 0.0;
            if (eqnt_ > 1) {         /* (2 November 2026) on the threads: each its range of columns and its own row maxima, merged by max (exact: the same bits) */
                #pragma omp parallel num_threads(eqnt_)
                { const int t_ = omp_get_thread_num(), T_ = omp_get_num_threads(); double *rt = eqrt_ + (size_t)t_ * m;
                  for (int i = 0; i < m; i++) rt[i] = 0.0;
                  const int j0 = ent_part(S->Ap, n, t_, T_), j1 = ent_part(S->Ap, n, t_ + 1, T_);   /* (2 November 2026: ranges of about equal entries, not columns: beam7's dense columns sit together) */
                  for (int j = j0; j < j1; j++) { double s = 0; const int sl = colblk && colblk[j] >= 0; for (int p = S->Ap[j]; p < S->Ap[j + 1]; p++) { const double v = fabs(S->Ax[p]); if (v > s) s = v; if (!sl && v > rt[S->Ai[p]]) rt[S->Ai[p]] = v; } cn[j] = s; }
                  #pragma omp barrier
                  const int i0 = (int)((long long)m * t_ / T_), i1 = (int)((long long)m * (t_ + 1) / T_);
                  for (int i = i0; i < i1; i++) { double v = 0; for (int u = 0; u < T_; u++) { const double w_ = eqrt_[(size_t)u * m + i]; if (w_ > v) v = w_; } rn[i] = v; } }
            } else
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
            #pragma omp parallel for schedule(static) num_threads(eqnt_) if (eqnt_ > 1)
            for (int ch_ = 0; ch_ < ENT_CH; ch_++) for (int j = ent_lo(S->Ap, n, ch_), j1_ = ent_lo(S->Ap, n, ch_ + 1); j < j1_; j++) for (int p = S->Ap[j]; p < S->Ap[j + 1]; p++) S->Ax[p] *= rn[S->Ai[p]] * cn[j];
            for (int i = 0; i < m; i++) S->Dr[i] *= rn[i];
            for (int j = 0; j < n; j++) S->Ec[j] *= cn[j];
            dev_ = dev;
            if (dev < 1e-2) break;
        }
        tps_ += sc_time() - tq_; tq_ = sc_time();
        if (ncand > 1) {
            double q_ = 0, qc_[64];     /* (64 fixed chunks summed in their order: the same bits at any thread count) */
            #pragma omp parallel for schedule(static) num_threads(eqnt_) if (eqnt_ > 1)
            for (int c_ = 0; c_ < 64; c_++) { double a_ = 0; const size_t p0 = nnzA_ * (size_t)c_ / 64, p1 = nnzA_ * (size_t)(c_ + 1) / 64; for (size_t p = p0; p < p1; p++) { const double v = fabs(S->Ax[p]); if (v > 0) { const double l_ = log(v); a_ += l_ * l_; } } qc_[c_] = a_; }
            for (int c_ = 0; c_ < 64; c_++) q_ += qc_[c_];
            q_ /= fmax((double)nnzA_, 1.0);
            tlg_ += sc_time() - tq_; tq_ = sc_time();
            if (dbg_on()) socp_printf("  [equilibration from %s: %d passes, mean log^2 |a| %.3f]\n", cand ? "the columns brought to 1" : "the matrix as given", npass_, q_);
            if (cand == 0 || q_ < 0.9 * qbest) { qbest = q_; cbest = cand; npb = npass_; devb = dev_;
                memcpy(DEb, S->Dr, sizeof(double) * m); memcpy(DEb + m, S->Ec, sizeof(double) * n); eqbb_ = eqb; eqcb_ = eqc; }
        }
        }
        if (ncand > 1) {
            if (Axb && cbest == 0) { free(S->Ax); S->Ax = Axb; Axb = NULL; } else { free(Axb); Axb = NULL; }     /* (cbest 1: S->Ax holds it) */
            memcpy(S->Dr, DEb, sizeof(double) * m); memcpy(S->Ec, DEb + m, sizeof(double) * n); eqb = eqbb_; eqc = eqcb_;
            npass_ = npb; dev_ = devb;
            if (dbg_on()) socp_printf("  [equilibration: the start %s kept]\n", cbest ? "with the columns brought to 1" : "from the matrix as given");
            if (dbg_on()) socp_printf("  [equilibration's parts: the passes %.3fs, the spreads %.3fs, the copies and the rest %.3fs]\n", tps_, tlg_, sc_time() - teq0_ - tps_ - tlg_);
            free(Ax0); free(Axb); free(DE0); free(DEb);
        }
        if (dbg_on()) socp_printf("  [equilibration: %d passes, the last change of a factor %.2e (%.3fs)]\n", npass_, dev_, sc_time() - teq0_);
        free(eqrt_);
        free(slrow); free(slblk); free(colblk); free(bmx);
        free(rn); free(cn);
    }
    for (int i = 0; i < m; i++) S->b[i] *= S->Dr[i];
    for (int j = 0; j < n; j++) S->c[j] *= S->Ec[j];
    const double trc0_ = sc_time();
    if (S->Ap[n] >= 100000 && g_cone_nt > 1) {     /* the row copy of the scaled A, for A x on the threads */
        const int nz = S->Ap[n];
        S->Rp = (int *)xcalloc((size_t)m + 2, sizeof(int)); S->Ri = (int *)malloc(sizeof(int) * ((size_t)nz + 1)); S->Rx = (double *)malloc(sizeof(double) * ((size_t)nz + 1));
        S->Rq = (int *)malloc(sizeof(int) * ((size_t)nz + 1));
        S->R0x = (double *)malloc(sizeof(double) * ((size_t)nz + 1));
        /* (2 November 2026, the SOCP developer) above a million entries on the threads: each thread counts its range of columns
         * by rows, each row's places are given out to the threads in the order of their ranges, and each thread places its
         * entries; a row's entries then come in the order of their columns, as one thread places them: the same copy.
         * 2013_firL2a (SOCP, 50 million entries): 2.2 s on one thread */
        const int rt_ = (nz >= 1000000 && !omp_in_parallel()) ? g_cone_nt : 1;
        int *rc_ = rt_ > 1 ? (int *)malloc(sizeof(int) * (size_t)rt_ * (size_t)m) : NULL;
        if (rc_) {
            #pragma omp parallel num_threads(rt_)
            { const int t_ = omp_get_thread_num(), T_ = omp_get_num_threads(); int *c_ = rc_ + (size_t)t_ * m;
              for (int i = 0; i < m; i++) c_[i] = 0;
              const int j0 = ent_part(S->Ap, n, t_, T_), j1 = ent_part(S->Ap, n, t_ + 1, T_);   /* (2 November 2026: ranges of about equal entries, not columns: beam7's dense columns sit together) */
              for (int p = S->Ap[j0]; p < S->Ap[j1]; p++) c_[S->Ai[p]]++;
              #pragma omp barrier
              #pragma omp single
              { int o_ = 0; S->Rp[0] = 0;
                for (int i = 0; i < m; i++) { for (int u = 0; u < T_; u++) { const int l_ = rc_[(size_t)u * m + i]; rc_[(size_t)u * m + i] = o_; o_ += l_; } S->Rp[i + 1] = o_; }
                S->Rp[m + 1] = o_; }
              for (int j = j0; j < j1; j++) for (int p = S->Ap[j]; p < S->Ap[j + 1]; p++) { const int q = c_[S->Ai[p]]++; S->Ri[q] = j; S->Rx[q] = S->Ax[p]; S->Rq[p] = q; } }
            free(rc_);
        } else {
        for (int p = 0; p < nz; p++) S->Rp[S->Ai[p] + 2]++;
        for (int i = 0; i < m; i++) S->Rp[i + 2] += S->Rp[i + 1];
        for (int j = 0; j < n; j++) for (int p = S->Ap[j]; p < S->Ap[j + 1]; p++) { const int q = S->Rp[S->Ai[p] + 1]++; S->Ri[q] = j; S->Rx[q] = S->Ax[p]; S->Rq[p] = q; }
        }
        #pragma omp parallel for schedule(static) num_threads(rt_) if (rt_ > 1)
        for (int p = 0; p < nz; p++) S->R0x[S->Rq[p]] = A0x[p];
    }
    const double tdn0_ = sc_time();
    dn_build(S);
    const double tax0_ = sc_time();
    axc_setup(S);
    if (dbg_on()) socp_printf("  [the row copy %.3fs, the dense block %.3fs, the products' setup %.3fs]\n", tdn0_ - trc0_, tax0_ - tdn0_, sc_time() - tax0_);
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
    /* (3 November 2026, the SOCP developer, a test: BRISK_CONEMEASDEFER=1) the full system's analysis put off when the presolve's
     * substitution already counted it (AMD's multiply-subtract pairs and entries of L for the reduction it chose): the normal
     * equations are judged against that count, and the full system is analysed only when it is first needed. db-joint-soerensen
     * (SOCP): its analysis 1.6 s, for a factorization the solve never makes */
    { static int mdf_c = -2; const int mdf = cone_on("BRISK_CONEMEASDEFER", &mdf_c);
      S->meas_sup = -1;
      if (mdf && g_meas_ok && !S->kkt_defer) {
          const double fl_ = 2.0 * g_meas_pairs, lz_ = g_meas_lnz, nk_ = (double)S->n + (double)S->m;
          S->kkt_defer = 1; S->cflops = fl_; g_meas_lnz_kept = lz_;
          S->meas_sup = lz_ > 40000 && (lz_ > 40.0 * nk_ || fl_ > 110.0 * lz_ || (lz_ > 15.0 * nk_ && fl_ > 60.0 * lz_));    /* (the backend kkt_analyse would choose) */
          if (dbg_on()) socp_printf("  [the full system's analysis put off: the presolve counted %.3g flops, %.3g entries of L (%s)]\n", fl_, lz_, S->meas_sup ? "supernodal" : "by columns");
      }
      g_meas_ok = 0; }
    /* (28 October 2026, the SOCP developer; ISSUES 805; BRISK_CONENELB=0: off) the normal equations judged before the full system
     * is analysed, against a lower bound of the full system's work, when their pattern is small: a factor with P entries over nk
     * columns does at least P^2 / nk operations, and the full system's factor holds at least its diagonal and the entries of A
     * (P >= nk + nnz(A)). When the normal equations' pattern, at most m(m+1)/2 entries, is within three quarters of that P, the
     * normal path's other tests (the pairs against nnz(L)) decide as they would with the full system analysed, and the final test
     * (the normal equations' work within 1.5 times the full system's) holds for the true work whenever it holds for the bound;
     * when it fails, the full system is analysed and the normal equations judged again as before. An SDP of 600,000 blocks of
     * order 2 on 300 rows (r2_tiny_lmi_N600k_o2, as 600,000 rotated cones of 3): the full system's analysis 10.8 s of 43 s, for
     * 5.5e7 flops a factorization against the normal equations' 9.1e6 (the bound 4.5e7) */
    { static int lbon = -1; if (lbon < 0) { const char *e_ = getenv("BRISK_CONENELB"); lbon = e_ ? atoi(e_) != 0 : 1; }
      if (lbon && !S->kkt_defer && S->m > 0) {
          long naux = 0; for (int k = 0; k < S->nq; k++) if (S->q[k] > SOCP_DENSE_MAX) naux += 2;
          const double nk_ = (double)S->n + (double)naux + (double)S->m, P_ = nk_ + (double)S->Ap[S->n];
          if (0.5 * (double)S->m * ((double)S->m + 1.0) <= 0.75 * P_) {
              S->kkt_defer = 1; S->kkt_defer_lb = 1; S->cflops = P_ * P_ / nk_;
              if (dbg_on()) socp_printf("  [the normal equations judged first, against the bound %.3g of the full system's work]\n", S->cflops);
          }
      } }
    /* (31 October 2026, the SOCP developer; the owner's SOCP benchmark, wide cones; BRISK_CONENEMD=0: off) the normal equations
     * judged first when the minimum-degree ordering of the full system would form them anyway: an unknown of a column has the
     * column's entries as neighbours (and its cone's other entries for a cone of at most SOCP_DENSE_MAX, the two terms of its
     * low-rank form for a larger one), an unknown of a row the row's entries. When every column has fewer neighbours than every
     * row, the ordering eliminates the columns first, and their elimination leaves on the rows exactly the pattern of the
     * normal equations (each column joins its rows): the full system's factor then holds the normal equations' factor, its
     * work is at least theirs, and its analysis is not needed to judge them. A059 (SOCP: 5,869 rows of about 1,200 entries,
     * columns of about 40): the full system's ordering 42 s, twice, for a factor of 6.8e10 flops equal to the dense normal
     * equations'. When they are refused (their dense or low-rank extras above half their factor), the full system is analysed
     * and they are judged again against it, as for the lower bound below */
    { static int mdon = -1; if (mdon < 0) { const char *e_ = getenv("BRISK_CONENEMD"); mdon = !(e_ && atoi(e_) == 0); }
      if (mdon && !S->kkt_defer && S->m > 0 && S->n > 0) {
          int *rl = (int *)calloc((size_t)S->m + 1, sizeof(int)); int maxc = 0, minr = INT_MAX;
          for (int p = 0; p < S->Ap[S->n]; p++) rl[S->Ai[p]]++;
          for (int i = 0; i < S->m; i++) if (rl[i] >= 2 && rl[i] < minr) minr = rl[i];      /* (a row of one entry is eliminated without fill and changes nothing else) */
          for (int j = 0; j < S->nf + S->nl; j++) { const int d_ = S->Ap[j + 1] - S->Ap[j]; if (d_ > maxc) maxc = d_; }
          for (int k = 0; k < S->nq; k++) { const int ex = S->q[k] <= SOCP_DENSE_MAX ? S->q[k] - 1 : 2;
              for (int j = S->qs[k]; j < S->qs[k] + S->q[k]; j++) { const int d_ = S->Ap[j + 1] - S->Ap[j] + ex; if (d_ > maxc) maxc = d_; } }
          free(rl);
          if (maxc < minr) {
              const double pm = 0.5 * (double)S->m * ((double)S->m + 1.0);
              S->kkt_defer = 1; S->kkt_defer_lb = 1; S->kkt_defer_md = 1; S->cflops = pm * pm / (double)S->m;
              if (dbg_on()) socp_printf("  [the normal equations judged first: every column has fewer neighbours (at most %d) than every row (at least %d)]\n", maxc, minr);
          } else if (dbg_on()) socp_printf("  [the normal equations not judged first: a column has %d neighbours, a row %d]\n", maxc, minr);
      } }
    { const char *e_ = getenv("BRISK_CONEROT"); S->rot_mode = e_ ? (atoi(e_) == 1 ? 1 : atoi(e_) == 2 ? 2 : 0) : 2; }
    for (int k = 0; k < S->nq; k++) if (S->q[k] >= 2 && S->q[k] <= SOCP_DENSE_MAX) S->nsmallc++;
    { const double ts = sc_time(); const int rc_ = kkt_setup(S); S->t_setup = sc_time() - ts;
      if (rc_ || (opt->timelimit > 0 && sc_time() - t0 > opt->timelimit)) { socp_free(S); free(A0x); free(b0); free(c0); free(rgam); free(rrho); free(rrow); free(rsd); free(rae);
          if (opt->timelimit > 0 && sc_time() - t0 > opt->timelimit) return socp_setup_timeout(P, R);
          R->status = SOCP_INPUT; return SOCP_INPUT; } }
    const size_t wn = (size_t)S->nk + 8;
    S->w1 = (double *)xcalloc(wn, sizeof(double)); S->w2 = (double *)xcalloc(wn, sizeof(double)); S->w3 = (double *)xcalloc(wn, sizeof(double));
    S->w4 = (double *)xcalloc(wn, sizeof(double)); S->rk = (double *)xcalloc(wn, sizeof(double)); S->sk = (double *)xcalloc(wn, sizeof(double));
    S->lastx = (double *)xcalloc(2 * wn, sizeof(double)); S->lasty = S->lastx + wn;
    S->swork2 = (double *)xcalloc((size_t)m + 8, sizeof(double));
    { const double ts = sc_time(); ne_setup(S); S->t_setup += sc_time() - ts; if (dbg_on()) socp_printf("  [kkt setup %.3fs, normal equations setup %.3fs]\n", S->t_setup - (sc_time() - ts), sc_time() - ts); }
    if (S->kkt_defer && !S->ne) {           /* (the normal equations not taken after all: the full system now) */
        const double ts = sc_time(); const int rc_ = kkt_analyse(S); S->t_setup += sc_time() - ts;
        if (!rc_ && S->kkt_defer_lb) {        /* (judged against the bound only: again, against the full system's work) */
            S->kkt_defer_lb = 0; S->kkt_defer_md = 0; const double ts2 = sc_time(); ne_setup(S); S->t_setup += sc_time() - ts2;
            if (dbg_on()) socp_printf("  [the normal equations %s against the full system once analysed]\n", S->ne ? "taken" : "refused");
        }
        if (rc_) { socp_free(S); free(A0x); free(b0); free(c0); free(rgam); free(rrho); free(rrow); free(rsd); free(rae);
            if (opt->timelimit > 0 && sc_time() - t0 > opt->timelimit) return socp_setup_timeout(P, R);
            R->status = SOCP_INPUT; return SOCP_INPUT; }
    }
    if (S->sc_lazy && !S->ne) { const double ts = sc_time(); kkt_sc_build(S); S->t_setup += sc_time() - ts; }    /* (the normal equations not taken: the full system's analysis now, kkt_analyse) */
    if (opt->timelimit > 0 && sc_time() - t0 > opt->timelimit) { socp_free(S); free(A0x); free(b0); free(c0); free(rgam); free(rrho); free(rrow); free(rsd); free(rae); return socp_setup_timeout(P, R); }
    S->gm_max = 10; { const char *e_ = getenv("BRISK_CONEGMK"); if (e_ && atoi(e_) > 0 && atoi(e_) <= 60) { S->gm_max = atoi(e_); S->refine_max = S->gm_max; } }   /* (test: GMRES's step cap) */
    S->gm_tol = 1e-12; S->gm_abs = 0.0; S->tg_d = S->tg_p = S->tg_g = 0.0; S->tg_f = 1.0; S->tg_fl = S->tg_fd = S->tg_fp = 0.0; S->tg_skip = 0; S->tg_w = 0;
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
    S->sreg = 1e-9; S->sregx = 1e-14; S->sregf = 1e-6;         /* regularization of the factored matrix: dual, cone and free blocks */
    /* (31 October 2026, the SOCP developer; ISSUES 810) the cone and linear block at 1e-14 (1e-9 before): H's entries fall to
     * 1e-9 and below on cones near their boundary, the fixed 1e-9 then dominated those pivots and the factor stopped being an
     * accurate preconditioner there (a user's truss compliance SOCP: from iteration 30 about 15,000 cone pivots below 1e-6,
     * the smallest 5.6e-9; GMRES stagnated at 1e-7 .. 1e-10, the steps 0.05 .. 0.6, tau decayed to 0: 500 iterations at an
     * error of 1e-2; now OPTIMAL in 44 iterations, 6.6 s; MOSEK 16 s on the same host). H itself is positive definite by
     * construction (the spectral form, l- = 1 / l+), the block stays negative definite; 1e-14 is the double-double factor's
     * value. The rows keep 1e-9 (part of the operator GMRES refines against, ISSUES 123). The owner's 52 benchmark cases
     * 0.986 in time, the 39 development SOCPs 1.007; sched_200_100_orig 50 -> 24 and socp_socopf_3012wp_k 140 -> 67
     * iterations; slower: socp_rocket_4000 (22 -> 30 iterations, its double-double end phase), nql180 (1.18), sched_100_50_orig
     * (27 -> 32 iterations: replaced pivots switch it to double-double). Forms relative to H (to its diagonal, to each
     * cone's smallest eigenvalue, to each cone's scale) lost a scheduling SOCP or the truss (bench/socp_truss_1031/).
     * BRISK_CONEREG=1e-9,1e-9: as before */
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
        if (ok && (rotsh == 2 || rotsh == 3) && S->nq > 0) {      /* (3: as 2, the separated cones started centrally, below) */
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
                if (rotsh == 3 && sepx && (sepx[k] || sepz[k]) && !(S->fcone && S->fcone[k])) {
                    /* (8 November 2026, the SOCP developer, a test: BRISK_CONEROTSHIFT=3) a separated cone (its own shift more than ten
                     * times the rest's) started centrally instead of at the margin 1: its first entry that of the rest, max(1, its
                     * least-squares first entry + the rest's shift), its tail the least-squares tail cut to at most half of it, so
                     * that its eigenvalues are within a factor 3. At the margin 1 the cone of 414,585 entries of the friction contacts
                     * (SOCP), whose least-squares tail has norm 57, had eigenvalues 115 and 1 */
                    double a = 0, b_ = 0;
                    for (int i = 1; i < d; i++) { a += px[j + i] * px[j + i]; b_ += qx[j + i] * qx[j + i]; }
                    const double hx = fmax(1.0, px[j] + ax), hz = fmax(1.0, -qx[j] + az);
                    const double fx = sqrt(a) > 0.5 * hx ? 0.5 * hx / sqrt(a) : 1.0, fz = sqrt(b_) > 0.5 * hz ? 0.5 * hz / sqrt(b_) : 1.0;
                    S->x[j] = hx; S->z[j] = hz;
                    for (int i = 1; i < d; i++) { S->x[j + i] = fx * px[j + i]; S->z[j + i] = -fz * qx[j + i]; }
                    if (dbg_on() && d > 100) socp_printf("  [start: cone %d (dimension %d) apart: first entries %.3e %.3e, tails cut by %.3e %.3e]\n", k, d, hx, hz, fx, fz);
                    continue;
                }
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
            /* (9 November 2026, the SOCP developer, a test: BRISK_CONESTARTPC=theta) each second-order cone started from its own
             * least-squares point, not by the shift shared by all: its first entry raised to (1 + theta) times the norm of its tail
             * where it is below (the margin measured against the cone's own size, so that a cone's start does not depend on how
             * far another cone's point lies outside), on each side; then both sides of the cone dilated by one factor so that its
             * product x_k'z_k is the start's mean product (every cone at the same mu, as on the central path). The shared shift is
             * set by the one cone furthest outside: robust dose (SOCP), shift 4.4 from one cone, a primal residual 150 times the
             * dual one. A cone whose least-squares point is zero on a side keeps the shared start on that side */
            { static double pcth = -1; if (pcth < 0) { const char *e_ = getenv("BRISK_CONESTARTPC"); pcth = e_ ? atof(e_) : 0.0; }
              if (pcth > 0 && S->nq > S->nr_rot) {
                double smu = 0; long npr = 0;
                for (int i = S->nf; i < S->nf + S->nl; i++) { smu += S->x[i] * S->z[i]; npr++; }
                for (int k = 0; k < S->nq; k++) { const int j = S->qs[k], d = S->q[k]; double v_ = 0; for (int i = 0; i < d; i++) v_ += S->x[j + i] * S->z[j + i]; smu += v_; npr++; }
                const double mu_t = npr > 0 && smu > 0 && isfinite(smu) ? smu / npr : 1.0;
                int nset = 0; double fmin_ = 1e300, fmax_ = 0;
                for (int k = 0; k < S->nq - S->nr_rot; k++) {
                    if (S->fcone && S->fcone[k]) continue;
                    const int j = S->qs[k], d = S->q[k];
                    double a = 0, b_ = 0; for (int i = 1; i < d; i++) { a += px[j + i] * px[j + i]; b_ += qx[j + i] * qx[j + i]; }
                    a = sqrt(a); b_ = sqrt(b_);
                    double hx = fmax(px[j], (1.0 + pcth) * a), hz = fmax(-qx[j], (1.0 + pcth) * b_);
                    if (!(hx > 0) || !isfinite(hx)) { hx = S->x[j]; for (int i = 1; i < d; i++) S->x[j + i] = px[j + i]; }
                    if (!(hz > 0) || !isfinite(hz)) { hz = S->z[j]; }
                    S->x[j] = hx; S->z[j] = hz;
                    double pk = 0; for (int i = 0; i < d; i++) pk += S->x[j + i] * S->z[j + i];
                    if (!(pk > 0) || !isfinite(pk)) continue;
                    const double f = sqrt(mu_t / pk);
                    for (int i = 0; i < d; i++) { S->x[j + i] *= f; S->z[j + i] *= f; }
                    nset++; fmin_ = fmin(fmin_, f); fmax_ = fmax(fmax_, f);
                }
                if (dbg_on()) socp_printf("  [start: %d second-order cones each from its own least-squares point (margin %.2f of its size), dilated to mu %.3e (factors %.3e .. %.3e)]\n", nset, pcth, mu_t, fmin_, fmax_);
              } }
            for (int i = 0; i < m; i++) S->y[i] = qy[i];
            free(sepx); free(sepz); sepx = sepz = NULL;
            if (epid) {
                double mu_ = 0; for (int i = S->nf; i < S->nf + S->nl; i++) mu_ += S->x[i] * S->z[i];
                mu_ = S->nl > 0 ? mu_ / S->nl : 1.0; if (!(mu_ > 0) || !isfinite(mu_)) mu_ = 1.0;
                if (start_fixmu() && start_fixmu() != 3) {      /* (the mean of the rest of the start: start_mu_rest; 3: the fixed cones only) */
                    unsigned char *sk_ = (unsigned char *)xcalloc((size_t)S->nq + 1, 1);
                    for (int k = 0; k < S->nq; k++) sk_[k] = epid[k] > 0;
                    mu_ = start_mu_rest(S, sk_, mu_); free(sk_);
                    if (dbg_on()) socp_printf("  [start: the epigraph cones' mu from the rest of the start: %.3e]\n", mu_);
                }
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
                    /* (8 November 2026, the SOCP developer, a test: BRISK_CONEFIXBIG=1) the cones of any size: the large ones were left to the
                     * shared shift, and their scale at the start then came from the equilibration's factor of each cone, not from their fixed
                     * entries (multiscale rows, SOCP: two cones of 6,975 and 6,341 entries whose fixed first entries are 1 and 0.05 in the file
                     * and 290 and 1.7e-4 after the equilibration, both started near 1) */
                    /* (9 November 2026, the SOCP developer; ISSUES 852) the default is 7: the large cones join the dilation start when every cone
                     * of the problem is fixed (one start rule for all cones); BRISK_CONEFIXBIG=0: the small cones only, as before; 1-6: tests */
                    static int fbig = -1; if (fbig < 0) { const char *e_ = getenv("BRISK_CONEFIXBIG"); fbig = e_ ? atoi(e_) : 8; }
                    int nfx = fbig ? find_fixed_d(S, el, fr, fp, fa, INT_MAX) : ne_find_fixed(S, el, fr, fp, fa);
                    /* (9 November 2026, the SOCP developer, a test: BRISK_CONEFIXBIG=7) one start rule for all cones: the large cones at the
                     * dilation to their fixed first entry (mode 2) only when every cone of the problem has its first entry fixed by a row of
                     * its own and there are no linear or free variables; otherwise the large ones keep the shared shift as before. Mixing
                     * the two rules unbalanced the cones that are not fixed against the ones that are (heterogeneous cones, trust region,
                     * array design, SOCP: 1-5 iterations more), while problems whose every cone is fixed gained (multiscale rows 0.75,
                     * network capacity 0.82, sensor calibration 0.90) */
                    unsigned char *dfx = NULL; int ndfx = 0;     /* (BRISK_CONEFIXBIG=8: the cones fixed on the dual side) */
                    if (fbig == 7 || fbig == 8) {
                        /* (8, the default since 9 November 2026 evening, with BRISK_CONEFIX8D=1 and BRISK_CONEFIX8P=4; 7: the rule without its
                         * dual counterpart. Measured on the owner's benchmark, the library, SOCPs: near tangent 9 -> 5 iterations (0.50-0.60 of the
                         * time, fresh and transformed copies 9 -> 5-6), wide dense 5 -> 4 (0.86-0.96), trust region 6-7 -> 7-8 on two of three
                         * cases (1.09, 1.15; copies two of five +1): the shared start puts its dual-fixed cone's z_0 at 2.3 against c_0 = 1, and any
                         * move costs centring in the first steps; the 39 development SOCPs two files +1 iteration, 37 identical.)
                         * (8, a test, 9 November 2026: the rule's dual counterpart. A cone whose first column is empty and whose cost there is
                         * positive has its dual first entry fixed, z_0 = c_0 (a radius |u| <= r written with r in the cost, the dual form of a
                         * norm bound: near tangent, SOCP); its scale is given by the problem on the dual side. "Every cone fixed" then counts
                         * the cones fixed on either side, and a dual-fixed cone starts with z at the dilation of the centre to c_0 and x = mu z^-1) */
                        int nall = nfx;
                        if (fbig == 8) {
                            dfx = (unsigned char *)xcalloc((size_t)S->nq + 1, 1);
                            for (int k = 0; k < S->nq - S->nr_rot; k++) { if (fr[k] >= 0 && fp[k] == 0) continue; const int j = S->qs[k];
                                if (S->Ap[j + 1] == S->Ap[j] && S->c[j] > 0) { dfx[k] = 1; ndfx++; } }
                            nall += ndfx;
                        }
                        if (!(nall == S->nq && S->nl == 0 && S->nf == 0 && S->nr_rot == 0)) { memset(el, 0, (size_t)m + 1); nfx = ne_find_fixed(S, el, fr, fp, fa); if (dfx) { free(dfx); dfx = NULL; } ndfx = 0; }
                    }
                    double mu_ = 0; for (int i = S->nf; i < S->nf + S->nl; i++) mu_ += S->x[i] * S->z[i];
                    mu_ = S->nl > 0 ? mu_ / S->nl : 1.0; if (!(mu_ > 0) || !isfinite(mu_)) mu_ = 1.0;
                    /* (8 November 2026, the SOCP developer; ISSUES 852; the default BRISK_CONEFIXMU=3, 0 off) the fixed cones' central point
                     * at the start's mean product, the fixed cones counted at the default (start_mu_rest), when that differs from the default
                     * by more than a factor 2; the default, the linear part's mean product, is 1 when there is none: the friction contacts
                     * (SOCP) have no linear variables, 51,823 fixed cones started at products of 1 and the other 51,824 pairs, shifted by the
                     * shared dual shift of 58, at 58-75; the centering steps were then blocked for 20 iterations (P05_00: 70 -> 19
                     * iterations, 43 -> 15 s; MOSEK 14.2 s; fresh and transformed instances 55-58 -> 17-19). Mode 1, the mean of the rest
                     * alone, centred 1,400-2,000 fixed cones at the product of one large cone where it was the only other pair (group
                     * sparse, network capacity: 1.12-1.24); mode 2 also the epigraph cones (TV denoising 12 -> 14 iterations) */
                    if (start_fixmu() && nfx > 0) {
                        unsigned char *sk_ = (unsigned char *)xcalloc((size_t)S->nq + 1, 1);
                        for (int k = 0; k < S->nq - S->nr_rot; k++) if (fr[k] >= 0 && fp[k] == 0) { const double rho = S->b[fr[k]] / fa[k]; sk_[k] = rho > 0 && isfinite(rho); }
                        mu_ = start_mu_rest(S, sk_, mu_); free(sk_);
                        if (dbg_on()) socp_printf("  [start: the fixed cones' mu from the rest of the start: %.3e]\n", mu_);
                    }
                    int nset = 0, nfx6 = 0;
                    double fx6_cn = 0, fx6_gn = 0;
                    if (fbig == 6) { fx6_cn = nrm2(n, S->c); fx6_gn = fabs(dotv(n, S->c, S->x)) + fabs(dotv(m, S->b, S->y)); }
                    for (int k = 0; nfx > 0 && k < S->nq - S->nr_rot; k++) {
                        if (fr[k] < 0 || fp[k] != 0) continue;
                        const double rho = S->b[fr[k]] / fa[k]; if (!(rho > 0) || !isfinite(rho)) continue;
                        const int j = S->qs[k], d = S->q[k];
                        /* (BRISK_CONEFIX8P, a test with BRISK_CONEFIXBIG=8: 0 leaves the large primal-fixed cones at the shared start when some
                         * cone is fixed on the dual side; 4 moves their row's dual variable by the change of z_0, as mode 4) */
                        static int f8p = -1; if (f8p < 0) { const char *e_ = getenv("BRISK_CONEFIX8P"); f8p = e_ ? atoi(e_) : 4; }
                        if (dfx && ndfx > 0 && f8p == 0 && d > NE_MAXCONE) continue;
                        double t2 = 0; for (int i = 1; i < d; i++) t2 += px[j + i] * px[j + i];
                        static double fcut = -1; if (fcut < 0) { const char *e_ = getenv("BRISK_CONEFIXCUT"); fcut = e_ ? atof(e_) : 0.9; }   /* (a test: the large cones' cut of the tail, 0.9 of rho as the small ones') */
                        const double cut = fst == 2 && t2 > 0 ? fmin(1.0, (d > NE_MAXCONE ? fcut : 0.9) * rho / sqrt(t2)) : 0.0;
                        double *xo_k = NULL; if ((fbig == 2 || fbig == 4 || fbig == 6 || fbig == 7 || fbig == 8) && d > NE_MAXCONE) { xo_k = (double *)malloc(sizeof(double) * (size_t)d); memcpy(xo_k, S->x + j, sizeof(double) * (size_t)d); }
                        /* (BRISK_CONEFIXBIG=5, a test: a large cone's dual side kept as the shared-shift start made it, so that neither the dual
                         * residual nor the dual objective moves, and its primal side placed at the point central to it with the fixed first
                         * entry: x = rho (1, -zbar / z0), x o z = (rho det(z) / z0) e. Mode 2 put the change into the dual residual, mode 4 into
                         * the dual objective (the fixed row's dual variable carries b: b'y -2.1e6 on multiscale rows) */
                        if (fbig == 5 && d > NE_MAXCONE && S->z[j] > 0) {
                            double zt = 0; for (int i = 1; i < d; i++) zt += S->z[j + i] * S->z[j + i];
                            if (sqrt(zt) < S->z[j]) { S->x[j] = rho; for (int i = 1; i < d; i++) S->x[j + i] = -rho * S->z[j + i] / S->z[j]; nset++; continue; }
                        }
                        S->x[j] = rho; for (int i = 1; i < d; i++) S->x[j + i] = cut * px[j + i];
                        double u2 = 0; for (int i = 1; i < d; i++) u2 += S->x[j + i] * S->x[j + i];
                        const double det = rho * rho - u2;
                        /* (BRISK_CONEFIXBIG=2, a test: a large cone keeps its own product of the shared-shift start, x_k'z_k, and is moved to the
                         * dilation at that level: the start's balance between the primal and the dual side, which the data set, stays; correlated
                         * design, SOCP: the common mu of 1 against the start's 4.6 cut tau to 0.26 in the first step, 7 -> 9 iterations) */
                        double muk = mu_;
                        if ((fbig == 2 || fbig == 4 || fbig == 6 || fbig == 7 || fbig == 8) && d > NE_MAXCONE) { double v_ = 0; for (int i = 0; i < d; i++) v_ += xo_k[i] * S->z[j + i]; if (v_ > 0 && isfinite(v_)) muk = v_; }
                        const double z0_old = S->z[j];
                        /* (BRISK_CONEBOXSTART=1, a test with BRISK_CONEBOXSOC: a box's cone (dimension 2, head fixed at u/2) keeps the dual side
                         * of the shared start, only moved into the dual cone's interior; its primal head is exact. The central point at the
                         * common mu put z_h = 2 mu / u: a large dual residual where the boxes are narrow (socp_qflow, SOCP: 1.8e3)) */
                        static int bxs = -1; if (bxs < 0) { const char *e_ = getenv("BRISK_CONEBOXSTART"); bxs = e_ ? atoi(e_) : 0; }
                        if (bxs == 4 && k >= g_box_k0 && k < g_box_k1) { free(xo_k); continue; }     /* (4: the box's cone keeps the shared start: its head not exact at first, as the bound row's slack) */
                        if (bxs == 1 && k >= g_box_k0 && k < g_box_k1) {
                            const double zv = S->z[j + 1]; if (!(S->z[j] > 1.01 * fabs(zv))) S->z[j] = 1.01 * fabs(zv) + 1e-3 * fmax(1.0, fabs(zv));
                            nset++; free(xo_k); continue;
                        }
                        /* (BRISK_CONEBOXSTART=2: the box's cone central, z = mu_k x^-1, at its own mu_k = zeta det / rho, so that its dual head is
                         * zeta, the shared start's level of the dual head (at least 1e-3): the central point at the common mu put z_h = mu rho / det,
                         * large for a narrow box; keeping the shared start's dual point (1) left it near the cone's boundary: steps blocked at once
                         * on the SOC-OPF relaxations) */
                        if (bxs == 2 && k >= g_box_k0 && k < g_box_k1) { const double zeta = fmax(S->z[j], 1e-3); muk = zeta * det / rho; }
                        /* (BRISK_CONEBOXSTART=3: central at the common mu with the head exact, and v (x_j - u/2) the root of zeta v^2 - mu v - zeta h^2 = 0
                         * (the stable form v = -2 zeta h^2 / (mu + sqrt(mu^2 + 4 zeta^2 h^2)), |v| < h), so that the box's net multiplier z_v = -mu v / det
                         * equals zeta, the shared start's z_v: the dual residual of x_j's column as the shared start had it, the box central) */
                        if (bxs == 3 && k >= g_box_k0 && k < g_box_k1) {
                            const double zeta = S->z[j + 1], hh = rho, v = -2.0 * zeta * hh * hh / (muk + sqrt(muk * muk + 4.0 * zeta * zeta * hh * hh));
                            S->x[j + 1] = v; const double det3 = hh * hh - v * v;
                            S->z[j] = muk * hh / det3; S->z[j + 1] = -muk * v / det3;
                            nset++; free(xo_k); continue;
                        }
                        if (!(fbig == 3 && d > NE_MAXCONE)) { S->z[j] = muk * rho / det; for (int i = 1; i < d; i++) S->z[j + i] = -muk * S->x[j + i] / det; }
                        /* (BRISK_CONEFIXBIG=4, a test: mode 2, and the fixed row's own dual variable takes the change of z_0: the row touches only
                         * x_0, so y_r enters the dual equations only through z_0 = c_0 - a y_r - r_0, and moving y_r by (z_0,old - z_0) / a leaves the
                         * dual residual as it was; mode 2 put the change into the dual residual: trust region, SOCP, 4.0 -> 49) */
                        if ((fbig == 4 || (dfx && ndfx > 0 && f8p == 4)) && d > NE_MAXCONE && fa[k] != 0.0) S->y[fr[k]] += (z0_old - S->z[j]) / fa[k];
                        /* (BRISK_CONEFIXBIG=6, a test: modes 2 and 4 chosen cone by cone by where the change of z_0 costs less in the start's
                         * relative measures: moved into the fixed row's dual variable (mode 4) it changes the dual objective by rho dz0, measured
                         * against 1 + |c'x| + |b'y|; left in the dual residual (mode 2) it is dz0 against 1 + |c|. Mode 4 when
                         * rho (1 + |c|) < 1 + |c'x| + |b'y|) */
                        if (fbig == 6 && d > NE_MAXCONE && fa[k] != 0.0 && rho * (1.0 + fx6_cn) < 1.0 + fx6_gn) { S->y[fr[k]] += (z0_old - S->z[j]) / fa[k]; nfx6++; }   /* (BRISK_CONEFIXBIG=3, a test: the primal side alone moved; the dual side of the shared-shift start kept, nearer to dual feasibility) */
                        if (dbg_on() > 1) { double zn = 0; for (int i = 0; i < d; i++) zn += S->z[j + i] * S->z[j + i]; socp_printf("  [start: cone %d (order %d) fixed on the primal side: rho %.3e, mu_k %.3e, z0 %.3e -> %.3e, |z| %.3e]\n", k, d, rho, muk, z0_old, S->z[j], sqrt(zn)); }
                        free(xo_k);
                        nset++;
                    }
                    if (dfx) {          /* (the dual-fixed cones: z at the dilation to c_0, its tail the shared start's cut to 0.9 c_0; x = mu_k z^-1 at the cone's own product) */
                        for (int k = 0; k < S->nq - S->nr_rot; k++) {
                            if (!dfx[k]) continue;
                            const int j = S->qs[k], d = S->q[k]; const double c0 = S->c[j];
                            /* (BRISK_CONEFIX8Z=2, the default since 10 November 2026: a dual-fixed cone takes the rule's start only where it is less
                             * dually infeasible than the shared start's: the shared start's infeasibility in this cone is its shift of z_0 (z_0 - c_0,
                             * the column being empty), the rule's the change of the tail |zbar + c_0 xbar / x_0|. Where the shared start was already
                             * dual feasible or nearly (single-cone least squares: socp_bigcone_*, socp_rlsinf_*, SOCPs) the rule had moved the tail and
                             * lost 1-2 iterations; near tangent (z_0 20 against c_0 2.8e-6) keeps 9 -> 5. 0: the rule everywhere)
                             * (BRISK_CONEFIX8Z=1, a test: a cone whose dual first entry the shared start already put at its fixed value (no dual
                             * shift: the column is empty, so z_0 = c_0 exactly) is left as the start made it; the rule's purpose, the fixed entry at
                             * its value, is met, and moving the tail gives up the start's dual feasibility: socp_bigcone_b5k, SOCP, dual residual
                             * 6.5e-17 -> 0.51, 5 -> 6 iterations) */
                            static int f8z = -1; if (f8z < 0) { const char *e_ = getenv("BRISK_CONEFIX8Z"); f8z = e_ ? atoi(e_) : 2; }
                            if (dbg_on() > 1 || f8z == 2) {   /* (the two starts' dual infeasibility in this cone: the shared start's is its shift of z_0, z_0 - c_0;
                                                                 * the rule's is the change of the tail, |zbar + c_0 xbar / x_0| (the least-squares tail is zbar)) */
                                double rt = 0; for (int i = 1; i < d; i++) { const double v = S->z[j + i] + c0 * S->x[j + i] / S->x[j]; rt += v * v; }
                                rt = sqrt(rt); const double rs = S->z[j] - c0;
                                if (dbg_on() > 1) socp_printf("  [start: cone %d (order %d) fixed on the dual side: the shared start's dual infeasibility %.3e, the rule's %.3e]\n", k, d, rs, rt);
                                if (f8z == 2 && !(rt < rs)) continue;
                            }
                            if (f8z == 1 && S->z[j] == c0) { if (dbg_on() > 1) socp_printf("  [start: cone %d (order %d) fixed on the dual side, already at c0 %.3e: left]\n", k, d, c0); continue; }
                            /* (BRISK_CONEFIX8D=1, a test: x kept as the shared start made it, z placed central to it with the fixed first entry,
                             * z = c0 (1, -xbar / x0), so that the primal residual does not move) */
                            static int f8d = -1; if (f8d < 0) { const char *e_ = getenv("BRISK_CONEFIX8D"); f8d = e_ ? atoi(e_) : 1; }
                            /* (BRISK_CONEFIX8D=2, a test: the cone's first column is empty, so x_0 enters no row and is free: the tail of x kept as the
                             * shared start made it (the primal residual does not move), z_0 = c_0, and x_0 the root of c_0 x_0^2 - mu x_0 - c_0 |xbar|^2 = 0,
                             * so that x and z = mu x^-1 are a central pair at the cone's own product mu with z_0 = c_0 exactly) */
                            if (f8d == 2) { double xt = 0, v2 = 0; for (int i = 1; i < d; i++) xt += S->x[j + i] * S->x[j + i];
                                for (int i = 0; i < d; i++) v2 += S->x[j + i] * S->z[j + i];
                                const double mk = v2 > 0 && isfinite(v2) ? v2 : mu_;
                                const double x0 = (mk + sqrt(mk * mk + 4.0 * c0 * c0 * xt)) / (2.0 * c0), dt = x0 * x0 - xt;
                                if (x0 > 0 && dt > 0 && isfinite(x0)) { S->x[j] = x0; S->z[j] = mk * x0 / dt; for (int i = 1; i < d; i++) S->z[j + i] = -mk * S->x[j + i] / dt;
                                    if (dbg_on() > 1) socp_printf("  [start: cone %d (order %d) fixed on the dual side, x_0 free: c0 %.3e, mu_k %.3e, x0 %.3e, z0 %.3e]\n", k, d, c0, mk, x0, S->z[j]);
                                    nset++; continue; } }
                            if (f8d == 1 && S->x[j] > 0) { double xt = 0; for (int i = 1; i < d; i++) xt += S->x[j + i] * S->x[j + i];
                                if (sqrt(xt) < S->x[j]) { S->z[j] = c0; for (int i = 1; i < d; i++) S->z[j + i] = -c0 * S->x[j + i] / S->x[j];
                                    if (dbg_on() > 1) socp_printf("  [start: cone %d (order %d) fixed on the dual side, central to x: c0 %.3e, x0 %.3e]\n", k, d, c0, S->x[j]);
                                    nset++; continue; } }
                            double v_ = 0; for (int i = 0; i < d; i++) v_ += S->x[j + i] * S->z[j + i];
                            double t2 = 0; for (int i = 1; i < d; i++) t2 += S->z[j + i] * S->z[j + i];
                            const double cut = t2 > 0 ? fmin(1.0, 0.9 * c0 / sqrt(t2)) : 0.0;
                            S->z[j] = c0; for (int i = 1; i < d; i++) S->z[j + i] *= cut;
                            double u2 = 0; for (int i = 1; i < d; i++) u2 += S->z[j + i] * S->z[j + i];
                            const double det = c0 * c0 - u2, muk = v_ > 0 && isfinite(v_) ? v_ : mu_;
                            S->x[j] = muk * c0 / det; for (int i = 1; i < d; i++) S->x[j + i] = -muk * S->z[j + i] / det;
                            if (dbg_on() > 1) socp_printf("  [start: cone %d (order %d) fixed on the dual side: c0 %.3e, mu_k %.3e, x0 %.3e, tail cut %.3f]\n", k, d, c0, muk, S->x[j], cut);
                            nset++;
                        }
                        free(dfx); dfx = NULL;
                    }
                    if (dbg_on()) socp_printf("  [start: %d second-order cones at the dilation to their fixed first entry (mu %.3e)%s%s]\n", nset, mu_, fbig == 6 ? (nfx6 ? "; the fixed rows' dual variables moved for some" : "; no fixed row's dual variable moved") : "", ndfx ? "; some on the dual side" : "");
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
    /* (BRISK_CONEBOXSOC with BRISK_CONEBOXSTART=5, a test: the boxes' cone form started from the start the problem with its bound rows
     * would have. The box as a two-dimensional cone is the bound row's pair (x_j, t) in other coordinates (h = (x_j + t/a)/2,
     * v = (x_j - t/a)/2; z_h = z_j + a z_t, z_v = z_j - a z_t; h z_h + v z_v = x_j z_j + t z_t), and the other rows' shift by a_ij u/2 is a row
     * operation: from the corresponding start the interior-point iterates are the same, and only the linear algebra differs. g_box_cap: the
     * start of this solve taken out in the problem's own coordinates; g_box_inj: a start given in them, set exactly) */
    if (g_box_cap || g_box_inj) {
        double *gx = g_box_cap ? g_box_cap->x : g_box_inj->x, *gz = g_box_cap ? g_box_cap->z : g_box_inj->z, *gy = g_box_cap ? g_box_cap->y : g_box_inj->y;
        const double rs2 = sqrt(0.5);
        if (g_box_cap) {
            for (int j = 0; j < n; j++) { gx[j] = S->x[j] * S->Ec[j] * S->sb; gz[j] = S->z[j] * S->scl / S->Ec[j]; }
            for (int i = 0; i < m; i++) gy[i] = S->y[i] * S->Dr[i] * S->scl;
            if (rgam) for (int k = P->nq; k < S->nq; k++) if (rrow[k] >= 0) gy[rrow[k]] *= rrho[k];
            for (int k = P->nq; k < S->nq; k++) {
                const int j = S->qs[k]; const double g = rgam ? rgam[k] : 1.0;
                const double a = rs2 * (gx[j] + gx[j + 1]), b = rs2 * (gx[j] - gx[j + 1]), c = rs2 * (gz[j] + gz[j + 1]), d = rs2 * (gz[j] - gz[j + 1]);
                gx[j] = a / g; gx[j + 1] = b * g; gz[j] = c * g; gz[j + 1] = d / g;
            }
            g_box_cap->tau = S->tau; g_box_cap->kap = S->kap * S->sb * S->scl;
        } else {
            for (int j = 0; j < n; j++) { S->x[j] = gx[j]; S->z[j] = gz[j]; }
            for (int k = P->nq; k < S->nq; k++) {
                const int j = S->qs[k]; const double g = rgam ? rgam[k] : 1.0;
                const double a = g * S->x[j], b = S->x[j + 1] / g, c = S->z[j] / g, d = g * S->z[j + 1];
                S->x[j] = rs2 * (a + b); S->x[j + 1] = rs2 * (a - b); S->z[j] = rs2 * (c + d); S->z[j + 1] = rs2 * (c - d);
            }
            for (int j = 0; j < n; j++) { S->x[j] /= S->Ec[j] * S->sb; S->z[j] *= S->Ec[j] / S->scl; }
            for (int i = 0; i < m; i++) S->y[i] = gy[i] / (S->Dr[i] * S->scl);
            if (rgam) for (int k = P->nq; k < S->nq; k++) if (rrow[k] >= 0) S->y[rrow[k]] /= rrho[k];
            S->tau = g_box_inj->tau; S->kap = g_box_inj->kap / (S->sb * S->scl);
        }
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
        socp_printf("BRISK SOCP solver: m = %d, n = %d (free %d, linear %d, cones %d", m, n, S->nf, S->nl, S->nq);
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
        if (getenv("BRISK_CONENEMEASURE")) nem_measure(S);
        if (dbg_on() && S->nq > 0) {     /* (3 November 2026, debug: how many distinct rows each cone's columns touch, against its dimension) */
            int *mk_ = (int *)malloc(sizeof(int) * ((size_t)m + 1)); for (int i = 0; i < m; i++) mk_[i] = -1;
            long hist[8] = {0}; long over = 0;
            for (int k = 0; k < S->nq; k++) { int r_ = 0; for (int j = S->qs[k]; j < S->qs[k] + S->q[k]; j++) for (int p = S->Ap[j]; p < S->Ap[j + 1]; p++) if (mk_[S->Ai[p]] != k) { mk_[S->Ai[p]] = k; r_++; }
                const int ex = r_ - S->q[k]; hist[ex < -1 ? 0 : ex > 5 ? 7 : ex + 2]++; if (ex > 0) over++; }
            socp_printf("  [cones by rows touched minus dimension: <-1 %ld, -1 %ld, 0 %ld, 1 %ld, 2 %ld, 3 %ld, 4 %ld, >4 %ld; %ld touch more rows than their dimension]\n", hist[0], hist[1], hist[2], hist[3], hist[4], hist[5], hist[6], hist[7], over);
            free(mk_);
        }
        {   extern int schol_panel_threads(void); const int pt = schol_panel_threads();   /* (17 October 2026: the large blocks of the factorization) */
            socp_threads_used = pt > 1 ? pt : 1;
            if (g_log_brief && !g_user_meth) SUSER("Number of threads: %d%s\nMethod: interior-point method, homogeneous self-dual embedding, Nesterov-Todd scaling\n", pt > 1 ? pt : 1, brisk_threads_label(pt > 1 ? pt : 1));
            else socp_printf("Number of threads: %d%s\n", pt > 1 ? pt : 1, brisk_threads_label(pt > 1 ? pt : 1)); }   /* (31 October 2026, ISSUES 727: the owner's wording) */
        if (g_log_brief) { g_user_meth = 1; brisk_log_head(1); }     /* (the table's head at each solve: a second attempt or a caller's re-solve starts its own table) */
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
    /* (9 November 2026, a test: BRISK_CONEMCCREF=k) the refinement steps of a centrality corrector's solve (10): a corrector only has to
     * lengthen the step; robust dose P11_01 (SOCP): 181 applications of the factor in 18 iterations, 105 of them solves */
    /* (13 November 2026, the SOCP developer: 0, unrefined, the default; one application of the factor serves a corrector, which is kept
     * only if it lengthens the step. The 39 development SOCPs 0.776, the 72 CBLIB SOCPs 0.79 with the small cones' eigenbasis for blocks
     * not definite as written (BRISK_CONEROTDEF=2), which removed the replaced pivots that made unrefined correctors fail on joint_HO_01
     * and joint_HO_05, SOCP; -1: refined only with replaced pivots, which lost sched_200_100_orig's first attempt, SOCP) */
    const int mcc_ref = getenv("BRISK_CONEMCCREF") ? atoi(getenv("BRISK_CONEMCCREF")) : 0, mcc_max = getenv("BRISK_CONEMCC") ? atoi(getenv("BRISK_CONEMCC")) : 3;
    const double mcc_da = getenv("BRISK_CONEMCCDA") ? atof(getenv("BRISK_CONEMCCDA")) : 0.2, mcc_bmin = getenv("BRISK_CONEMCCB") ? atof(getenv("BRISK_CONEMCCB")) : 0.1, mcc_bmax = 1.0 / mcc_bmin, mcc_acc = getenv("BRISK_CONEMCCACC") ? atof(getenv("BRISK_CONEMCCACC")) : 0.1, step_frac = opt->step > 0 && opt->step < 1 ? opt->step : (getenv("BRISK_CONESTEPF") ? atof(getenv("BRISK_CONESTEPF")) : 0.99);   /* (BRISK_CONESTEPF: a test, 29 October 2026) */
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
    double prev_alpha = 1.0;      /* (7 November 2026: the last iteration's step, for the end phase's trigger below) */
    int nfeas = 0, feas_pending = 0, nsmall = 0, nf_retry = 0, nf_restore = 0; double nf_sv[4] = { 0, 0, 0, 1 }; const int feas_on = getenv("BRISK_CONENOFEAS") == NULL;
    double feas_prev = 0;
    double *sv_x = (double *)xcalloc((size_t)n, sizeof(double)), *sv_z = (double *)xcalloc((size_t)n, sizeof(double)), *sv_y = (double *)xcalloc((size_t)m + 1, sizeof(double));
    double c_sw0 = 0;   /* (the modelled cost of the iteration that switched the small cones to their eigenbasis) */
    double t_chk = 0;   /* (the time of the last check of the time limit, since t0) */
    double tq[6] = {0, 0, 0, 0, 0, 0}, tq_last = -1; double td[6] = {0, 0, 0, 0, 0, 0}, td_last = 0;   /* (12 November 2026, debug: directions and steps by part) */   /* (debug: the iteration's time by part: residuals, errors, rays, scaling and factorization, the rest) */
    double tab_step = 0;   /* (3 November 2026: the default log's iteration table, brisk_itab_row in main.c: the last step's length) */
    double rr_np = -1, rr_nd = -1, rr_pred = 1, rr_mu = -1, rr_tau = -1, rr_g = -1;
    int sf_p = 0, sf_d = 0, ddst_cnt = 0, shst_cnt = 0, short_stop = 0;     /* (the residuals' shortfall against the direction's promise, and its run; BRISK_CONEDDSHORT) */
    double row_step = 0.0; int row_last = -1;                               /* (the default log: the step that led to the iterate of the row) */
    double stall_best = 1e300; int stall_it = 0;           /* (the stall rule: the best of the primal and dual errors, and the last iteration that cut it by a tenth or found it within the tolerance) */          /* (the homogeneous residuals' norms after the last step, and the reduction its direction promised) */
    /* (BRISK_CONEPOLISH: socp_polish) the polished point is judged by the loop's own errors: pol_state 1 while it is in S */
    int pol_state = 0, pol_try = 0, pol_np = 0, pol_skiprr = 0; double pol_mu = 1e300, pol_sv[8];
    double *pol_x = NULL, *pol_y = NULL, *pol_z = NULL, *pol_xs = NULL, *pol_ys = NULL, *pol_zs = NULL;
    double *pol_ax = NULL, *pol_az = NULL, *pol_bx = NULL, *pol_bz = NULL, pol_amu = 0, pol_bmu = 0;    /* (two earlier iterates / tau, A older than B) */
    static int polon = -1; if (polon < 0) { const char *e_ = getenv("BRISK_CONEPOLISH"); polon = e_ ? atoi(e_) : 0; }
    g_t_cstep = 0; g_t_ks = 0; { const double tp_ = S->t_prec, tm_ = S->t_mul; (void)tp_; (void)tm_; }
    for (it = 0; ; it++) {
        S->cur_it = it;
      pol_top:
        if (tq_last >= 0) tq[4] += sc_time() - tq_last;
        tq_last = sc_time();
        /* ---- residuals of the homogeneous model (scaled problem) */
        for (int i = 0; i < m; i++) rP[i] = S->b[i] * S->tau;
        A_mul(S, -1.0, S->x, rP);                                        /* b tau - A x */
        #pragma omp parallel for schedule(static) num_threads(g_cone_nt) if (cone_vpar(S))
        for (int j = 0; j < n; j++) rD[j] = S->c[j] * S->tau - S->z[j];
        At_mul(S, -1.0, S->y, rD);                                       /* c tau - A'y - z */
        const double cx = dotv(n, S->c, S->x), byv = dotv(m, S->b, S->y);
        const double rG = byv - cx - S->kap;
        double xz = 0; for (int j = S->nf; j < n; j++) xz += S->x[j] * S->z[j];
        const double mu = (xz + S->tau * S->kap) / nu;
        if (it == 0 && S->tau > 0) { R->start_err[0] = nrm2(m, rP) / (S->tau * (1.0 + nrm2(m, S->b))); R->start_err[1] = nrm2(n, rD) / (S->tau * (1.0 + nrm2(n, S->c))); }
        {   const double np_ = nrm2(m, rP), nd_ = nrm2(n, rD);
            if (dbg_on() && rr_np > 0) socp_printf("      [the homogeneous residuals fell by %.3e (primal) %.3e (dual) %.3e (gap row) against the direction's %.3e; mu by %.3e, tau by %.3e, kappa/tau %.2e]\n", np_ / rr_np, nd_ / fmax(rr_nd, 1e-300), fabs(rG) / fmax(rr_g, 1e-300), rr_pred, mu / rr_mu, S->tau / rr_tau, S->kap / S->tau);
            if (!pol_skiprr && !pol_state) {
            /* (8 November 2026, the SOCP developer, a test: BRISK_CONENESHORT=k) on the normal equations, a step after which the primal or
             * the dual residual of the homogeneous model fell by more than k times less than the direction promised (its ratio above k
             * times the promised one, which is below 1/2): the direction was inexact, and the full system takes over from the iterate (the
             * same affordability bound as the other exits). Heterogeneous cones (A05_00, SOCP): at iteration 7 the dual residual fell by
             * 6.2e-2 against 1.04e-2 (the homogeneous model's fixed column K^-1 [c; b] solved to 6e-6 in the large cone's rows, GMRES not
             * refining it), tau then grew 0.36 -> 500 and three iterations were lost; the full system ends at iteration 8, as MOSEK */
            { static double nes_k = -1; if (nes_k < 0) { const char *e_ = getenv("BRISK_CONENESHORT"); nes_k = e_ ? atof(e_) : 0.0; }
              if (nes_k > 0 && rr_np > 0 && rr_pred < 0.5 && best <= 1e-6 &&      /* (near the end only, as the other two exits) */ (np_ / rr_np > nes_k * rr_pred || nd_ / fmax(rr_nd, 1e-300) > nes_k * rr_pred) && S->ne && !S->ne_back && S->cflops > 0
                  && 4.0 * S->cflops <= (double)(it + 1) * fmax(schol_flops(S->ne), 1.0)) {
                  if (dbg_on()) socp_printf("      [the residuals fell by %.2e (primal) %.2e (dual) against the direction's %.2e on the normal equations: the full system from here]\n", np_ / rr_np, nd_ / fmax(rr_nd, 1e-300), rr_pred);
                  static int nb3_c = -2; const int nb3_ = cone_back("BRISK_CONENEBACK", &nb3_c);
                  if (nb3_ && S->nf == 0 && !S->ne_park) { S->ne_park = S->ne; S->ne = NULL; S->ne_ok = 0; S->ne_left = it; }
                  else { ne_free(S); S->ne_ok = 0; S->ne_left = it; }
              } }
            sf_p = rr_np > 0 && 1.0 - np_ / rr_np < 0.5 * (1.0 - rr_pred); sf_d = rr_nd > 0 && 1.0 - nd_ / fmax(rr_nd, 1e-300) < 0.5 * (1.0 - rr_pred);
            rr_np = np_; rr_nd = nd_; rr_mu = mu; rr_tau = S->tau; rr_g = fabs(rG); }
            pol_skiprr = 0; }
        tq[0] += sc_time() - tq_last; tq_last = sc_time();
        /* ---- errors of the problem as given */
        {
            const double f = 1.0 / S->tau;
            #pragma omp parallel for schedule(static) num_threads(g_cone_nt) if (cone_vpar(S))
            for (int j = 0; j < n; j++) { xo[j] = S->Ec[j] * S->x[j] * f * S->sb; zo[j] = S->z[j] * f * S->scl / S->Ec[j]; }
            cone_unboost(S, xo, zo);
            for (int i = 0; i < m; i++) yo[i] = S->Dr[i] * S->y[i] * f * S->scl;
            if (rgam) rot_ungamma(P->nq, S->nq, S->qs, rgam, rrho, rrow, xo, yo, zo);
            if (rgam && rotrow1) rot_rowdual(P->nq, S->nq, S->qs, rsd, rrow, rae, P->c, yo, zo);
            double *ra = S->w1, *rb = S->w2;
            /* (19 October 2026, LP/SOCP: on the threads with the row copy, each row's sum in the order of the columns
             * as by columns, so the same bits) */
            if (S->axc_part) { for (int i = 0; i < m; i++) ra[i] = -b0[i]; axc_mul(S, S->Ap, S->Ai, A0x, 1.0, xo, ra); }
            else if (S->R0x && cone_apar(S)) {
                #pragma omp parallel for schedule(static) num_threads(g_cone_nt)
                for (int ch = 0; ch < ENT_CH; ch++) { const int i1 = ent_lo(S->Rp, m, ch + 1);
                for (int i = ent_lo(S->Rp, m, ch); i < i1; i++) { double v = -b0[i]; for (int q = S->Rp[i]; q < S->Rp[i + 1]; q++) { const double xj = xo[S->Ri[q]]; if (xj != 0.0) v += S->R0x[q] * xj; } ra[i] = v; } }
            } else {
                for (int i = 0; i < m; i++) ra[i] = -b0[i];
                for (int j = 0; j < n; j++) { const double xj = xo[j]; if (xj != 0.0) for (int p = S->Ap[j]; p < S->Ap[j + 1]; p++) ra[S->Ai[p]] += A0x[p] * xj; }
            }
            #pragma omp parallel for schedule(static) num_threads(g_cone_nt) if (S->R0x && cone_apar(S))
            for (int ch = 0; ch < ENT_CH; ch++) { const int j1 = ent_lo(S->Ap, n, ch + 1);
            for (int j = ent_lo(S->Ap, n, ch); j < j1; j++) { double s = 0; for (int p = S->Ap[j]; p < S->Ap[j + 1]; p++) s += A0x[p] * yo[S->Ai[p]]; rb[j] = c0[j] - s - zo[j]; } }
            pobj = dotv(n, c0, xo); dobj = dotv(m, b0, yo);
            e_p = nrm2(m, ra) / (1.0 + nb0); e_d = nrm2(n, rb) / (1.0 + nc0);
            /* (3 November 2026, the SOCP developer, a test: BRISK_CONEDDSHORT=1) the linear solves judged by what the step
             * delivers: when for three iterations in a row a residual whose error is above the tolerance falls by less than
             * half of what the direction promised (1 - alpha (1 - sigma)), the solves are not accurate enough for the step
             * (db-joint-soerensen, SOCP: from iteration 40 the dual residual falls by 0.95-0.97 against 0.78-0.82, its free
             * columns' part not at all), and the factorization goes to double-double whatever its size: the double route
             * cannot converge there, so its cheaper factorization is no saving */
            { static int dds = -1; if (dds < 0) { const char *e_ = getenv("BRISK_CONEDDSHORT"); dds = e_ ? atoi(e_) : 0; }
              /* (3 November 2026, the SOCP developer, a test: BRISK_CONESHORTSTOP=1) the same evidence, three iterations in a row,
               * where double-double is not available (off, or above its size cap): the solves cannot deliver what the steps need
               * and no more precise factorization will come, so the solve stops with its best point, reduced accuracy, once the
               * primal and dual residuals and the gap are at most 1e-6. db-joint-soerensen (SOCP): from iteration 45 the dual
               * residual falls by 0.02 a step against 0.14 promised; the stall stop came 26 iterations (140 s) later */
              { static int shs = -1; if (shs < 0) { const char *e_ = getenv("BRISK_CONESHORTSTOP"); shs = e_ ? atoi(e_) : 0; }
                static double ddmx_ = -1; if (ddmx_ < 0) { const char *e_ = getenv("BRISK_CONEDDMAX"); ddmx_ = e_ ? atof(e_) : 5e8; }
                if (shs && !S->use_dd && (!S->dd_on || S->cflops > ddmx_) && !pol_state) {
                    if ((sf_d && e_d > opt->tol) || (sf_p && e_p > opt->tol)) shst_cnt++; else shst_cnt = 0;
                    if (shst_cnt >= 3) short_stop = 1;
                }
                /* (3 November 2026, the SOCP developer, a test: BRISK_CONEPRECSTOP=1) the precision limit: the double factor of the
                 * full system had to be refactored with a raised regularization because it broke the quasi-definite bound
                 * (BRISK_CONEQDCHECK) and double-double is not available: the iterations from here run on solves that the
                 * raised regularization makes inexact (they stall, ISSUES 801, 820), so the solve stops with its best point.
                 * db-joint-soerensen (SOCP): iteration 37, objective -3.3401284 (4e-5 from the optimum; COPT's 2.3e-3) */
                /* (4 November 2026, the SOCP developer, a test: BRISK_CONEDSTALL=K) once the double factor had to be refactored with a
                 * raised regularization (it broke the quasi-definite bound) and double-double is not available: the dual residual
                 * not halved in K iterations, while the primal and dual residuals and the gap are within 1e-6, stops the solve
                 * with its best point. What follows the raise decides, not the raise itself: on db-joint-soerensen (SOCP) the dual
                 * residual stops at 4e-8 (last halved at iteration 41); on joint_HO_01 (SOCP), the same state at the raise, it goes
                 * on to OPTIMAL (56 iterations, the precision-limit stop had cut it at iteration 34) */
                { static int dsk = -1; if (dsk < 0) { const char *e_ = getenv("BRISK_CONEDSTALL"); dsk = e_ ? atoi(e_) : 0; }
                  static _Thread_local double dsb; static _Thread_local int dsi, dson;
                  if (it == 0) { dsb = 1e300; dsi = 0; dson = 0; }
                  if (dsk > 0 && S->qd_ever && !S->use_dd && (!S->dd_on || S->cflops > ddmx_) && !pol_state) {
                      if (!dson) { dson = 1; dsb = e_d; dsi = it; }
                      else if (e_d < 0.5 * dsb) { dsb = e_d; dsi = it; }
                      if (it - dsi >= dsk && fmax(fmax(e_p, e_d), fabs(e_g)) <= 1e-6 && fmax(fmax(e_p, e_d), fmax(fabs(e_g), e_c)) > opt->tol) short_stop = 1;
                  } }
                { static int prs_c = -2; const int prs = cone_pkg("BRISK_CONEPRECSTOP", &prs_c);
                  if (prs && S->qd_raised && !S->use_dd && (!S->dd_on || S->cflops > ddmx_) && !pol_state && fmax(fmax(e_p, e_d), fmax(fabs(e_g), e_c)) > 1e-6) short_stop = 1; } }
              if (dds && S->dd_on && !S->use_dd) {
                  if ((sf_d && e_d > opt->tol) || (sf_p && e_p > opt->tol)) ddst_cnt++; else ddst_cnt = 0;
                  if (ddst_cnt >= 3 && !S->dd_stall) { S->dd_stall = 1; S->want_dd = 1;
                      if (dbg_on()) socp_printf("      [the residuals fell by less than half of the direction's promise for 3 iterations: double-double whatever its size]\n"); }
              } }
            if (getenv("BRISK_CONEDRES")) {     /* (2 November 2026, a test: where the dual residual of the problem as given sits, by kind of column, against the scaled one) */
                double fr = 0, li = 0, co = 0, sfr = 0, sli = 0, sco = 0, mx = 0; int jm = -1;
                for (int j = 0; j < n; j++) { const double v = rb[j] * rb[j], w_ = rD[j] * rD[j] / (S->tau * S->tau); if (j < S->nf) { fr += v; sfr += w_; } else if (j < S->nf + S->nl) { li += v; sli += w_; } else { co += v; sco += w_; } if (fabs(rb[j]) > mx) { mx = fabs(rb[j]); jm = j; } }
                socp_printf("      [dual residual as given: free %.2e linear %.2e cones %.2e; scaled (homogeneous / tau): free %.2e linear %.2e cones %.2e; largest %.2e at column %d (%s), x %.2e, column factor %.2e]\n", sqrt(fr), sqrt(li), sqrt(co), sqrt(sfr), sqrt(sli), sqrt(sco), mx, jm, jm < S->nf ? "free" : jm < S->nf + S->nl ? "linear" : "cone", jm >= 0 ? xo[jm] : 0.0, jm >= 0 ? S->Ec[jm] : 0.0);
            }
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
                if (S->R0x && cone_apar(S)) {      /* (2 November 2026: by the row copy on the threads; each row's sum in the order of its columns, as by columns: the same bits) */
                    #pragma omp parallel for schedule(static) num_threads(g_cone_nt)
                    for (int ch = 0; ch < ENT_CH; ch++) { const int i1 = ent_lo(S->Rp, m, ch + 1);
                    for (int i = ent_lo(S->Rp, m, ch); i < i1; i++) { double v = 0.0; for (int q = S->Rp[i]; q < S->Rp[i + 1]; q++) { const double xj = xo[S->Ri[q]]; if (xj != 0.0) v += fabs(S->R0x[q] * xj); } act[i] = v; } }
                } else {
                for (int i = 0; i < m; i++) act[i] = 0.0;
                for (int j = 0; j < n; j++) { const double xj = xo[j]; if (xj != 0.0) for (int p = S->Ap[j]; p < S->Ap[j + 1]; p++) act[S->Ai[p]] += fabs(A0x[p] * xj); }
                }
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
            if (getenv("BRISK_CONEDRES")) {     /* (2 November 2026, a test: x'z against the gap, x'z = c'x - b'y - y'r_p - x'r_d) */
                double xr = 0, yr = 0, xzl = 0, xzq = 0, xzr = 0; for (int j = 0; j < n; j++) xr += xo[j] * rb[j]; for (int i = 0; i < m; i++) yr += yo[i] * ra[i];
                for (int j = S->nf; j < S->nf + S->nl; j++) xzl += xo[j] * zo[j];
                for (int k = 0; k < S->nq; k++) { double a = 0; for (int i = 0; i < S->q[k]; i++) a += xo[S->qs[k] + i] * zo[S->qs[k] + i]; if (k < P->nq) xzq += a; else xzr += a; }
                socp_printf("      [x'z %.3e (linear %.3e, second-order cones %.3e, rotated cones %.3e); c'x - b'y %.3e; y'r_p %.3e; x'r_d %.3e; |x| %.2e |y| %.2e |z| %.2e]\n", xzo, xzl, xzq, xzr, pobj - dobj, yr, xr, nrm2(n, xo), nrm2(m, yo), nrm2(n, zo));
            }
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
            #pragma omp parallel for schedule(static) num_threads(g_cone_nt) if (cone_vpar(S))
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
        /* (2 November 2026, the SOCP developer, BRISK_CONETRACE=1) one line an iteration with the four errors the stopping test
         * reads and the time, so that other stopping rules can be judged on the trace afterwards; the solve is not changed */
        { static int trc = -1; if (trc < 0) trc = getenv("BRISK_CONETRACE") != NULL;
          if (trc) { printf("trace %d %.3f pfeas %.3e dfeas %.3e gap %.3e xz %.3e mu %.3e tau %.3e kappa %.3e\n", it, sc_time() - t0, e_p, e_d, e_g, e_c, mu, S->tau, S->kap); fflush(stdout); } }
        const double err = fmax(fmax(e_p, e_d), fmax(fabs(e_g), e_c));
        if (pol_state == 1) {          /* the polished point, judged */
            pol_state = 0;
            if (dbg_on() || S->verbose > 1) socp_printf("      [the polished point (%d pieces): primal %.2e dual %.2e gap %.2e x'z %.2e: %s]\n", pol_np, e_p, e_d, e_g, e_c, err <= opt->tol ? "accepted" : "not taken");
            if (!(err <= opt->tol)) {
                memcpy(S->x, pol_x, sizeof(double) * (size_t)n); memcpy(S->y, pol_y, sizeof(double) * (size_t)m); memcpy(S->z, pol_z, sizeof(double) * (size_t)n);
                S->tau = pol_sv[0]; S->kap = pol_sv[1]; pol_skiprr = 1;
                goto pol_top;
            }
            pol_try = 1000;      /* (accepted: the test below ends the solve) */
        } else if (polon && S->tau > 0 && !pol_state && (!pol_bx || mu < 0.1 * pol_bmu)) {     /* (the snapshots for the trend) */
            if (!pol_ax) { pol_ax = (double *)malloc(sizeof(double) * ((size_t)n + 1)); pol_az = (double *)malloc(sizeof(double) * ((size_t)n + 1)); pol_bx = (double *)malloc(sizeof(double) * ((size_t)n + 1)); pol_bz = (double *)malloc(sizeof(double) * ((size_t)n + 1)); pol_bmu = 0; }
            if (pol_bmu > 0) { double *t_ = pol_ax; pol_ax = pol_bx; pol_bx = t_; t_ = pol_az; pol_az = pol_bz; pol_bz = t_; pol_amu = pol_bmu; }
            for (int j = 0; j < n; j++) { pol_bx[j] = S->x[j] / S->tau; pol_bz[j] = S->z[j] / S->tau; } pol_bmu = mu / (S->tau * S->tau);
        }
        if (pol_state == 0 && polon && S->tau > 0 && pol_try < 4 && S->kap <= S->tau && fmax(fmax(e_p, e_d), fabs(e_g)) <= 1e-6 && err > opt->tol && mu < 0.1 * pol_mu) {
            if (!pol_x) { pol_x = (double *)malloc(sizeof(double) * ((size_t)n + 1)); pol_y = (double *)malloc(sizeof(double) * ((size_t)m + 1)); pol_z = (double *)malloc(sizeof(double) * ((size_t)n + 1));
                          pol_xs = (double *)malloc(sizeof(double) * ((size_t)n + 1)); pol_ys = (double *)malloc(sizeof(double) * ((size_t)m + 1)); pol_zs = (double *)malloc(sizeof(double) * ((size_t)n + 1)); }
            pol_try++; pol_mu = mu;
            const double mun_ = mu / (S->tau * S->tau);
            const double *hx_ = NULL, *hz_ = NULL;
            if (pol_bmu >= 10.0 * mun_) { hx_ = pol_bx; hz_ = pol_bz; } else if (pol_amu >= 10.0 * mun_) { hx_ = pol_ax; hz_ = pol_az; }
            if (socp_polish(S, pol_xs, pol_ys, pol_zs, &pol_np, hx_, hz_) == 0) {
                memcpy(pol_x, S->x, sizeof(double) * (size_t)n); memcpy(pol_y, S->y, sizeof(double) * (size_t)m); memcpy(pol_z, S->z, sizeof(double) * (size_t)n);
                pol_sv[0] = S->tau; pol_sv[1] = S->kap;
                memcpy(S->x, pol_xs, sizeof(double) * (size_t)n); memcpy(S->y, pol_ys, sizeof(double) * (size_t)m); memcpy(S->z, pol_zs, sizeof(double) * (size_t)n);
                S->tau = 1.0; S->kap = 0.0; pol_state = 1;
                goto pol_top;
            }
        }
        if (opt->wsave > 0 && !R->cx && S->tau > 0 && fabs(e_g) <= opt->wsave && fmax(e_p, e_d) <= 1e-2) {
            R->cx = (double *)malloc(sizeof(double) * ((size_t)n + 1)); R->cy = (double *)malloc(sizeof(double) * ((size_t)m + 1)); R->cz = (double *)malloc(sizeof(double) * ((size_t)n + 1));
            memcpy(R->cx, xo, sizeof(double) * (size_t)n); memcpy(R->cy, yo, sizeof(double) * (size_t)m); memcpy(R->cz, zo, sizeof(double) * (size_t)n);
        }
        { const double ef_ = fmax(e_p, e_d); if ((ef_ < 0.9 * stall_best || ef_ <= opt->tol) && S->tau > 0) { stall_best = ef_; stall_it = it; } }
        /* (2 November 2026, the SOCP developer, a test: BRISK_CONESTALL=K) where a stall rule would stop: the largest error (with the
         * complementarity x'z, and without it) not halved in K iterations while above the tolerance; printed once per variant,
         * with the time, the best error and kappa/tau, and the solve goes on unchanged (the data for judging the rule) */
        { static int stk = -1; static double sthr = 1e300; if (stk < 0) { const char *e_ = getenv("BRISK_CONESTALL"); stk = 0; if (e_) { double t_ = 0; if (sscanf(e_, "%d,%lf", &stk, &t_) == 2 && t_ > 0) sthr = t_; } }   /* (K[,threshold]: only when the best error is at most the threshold) */
          static _Thread_local double sb_[2]; static _Thread_local int si_[2], sd_[2];
          if (stk > 0) {
              if (it == 0) { sb_[0] = sb_[1] = 1e300; si_[0] = si_[1] = 0; sd_[0] = sd_[1] = 0; }
              const double ev_[2] = { fmax(fmax(e_p, e_d), fmax(fabs(e_g), e_c)), fmax(fmax(e_p, e_d), fabs(e_g)) };
              for (int v_ = 0; v_ < 2; v_++) {
                  if (S->tau > 0 && ev_[v_] < 0.5 * sb_[v_]) { sb_[v_] = ev_[v_]; si_[v_] = it; }
                  if (!sd_[v_] && it - si_[v_] >= stk && sb_[v_] > opt->tol && sb_[v_] <= sthr) {
                      sd_[v_] = 1;
                      printf("stallrule %s would stop at iteration %d, %.2fs: best error %.2e (iteration %d), now %.2e, kappa/tau %.2e\n",
                             v_ ? "without-xz" : "with-xz", it, sc_time() - t0, sb_[v_], si_[v_], ev_[v_], S->tau > 0 ? S->kap / S->tau : 1e300); fflush(stdout);
                  }
              }
          } }
        /* (3 November 2026, the SOCP developer; the owner, 3 November: keep the four-error stopping test and add the stall stop if
         * it makes no example worse) the stall stop: the largest of the four errors (x'z included) not halved in 20 iterations,
         * its best value at most 1e-6 and above the tolerance: the solve stops with reduced accuracy and returns its best point,
         * without the second attempt (which would repeat the stall). BRISK_CONESTALLSTOP=K: after K iterations; 0: off */
        { static int sst = -1; if (sst < 0) { const char *e_ = getenv("BRISK_CONESTALLSTOP"); sst = e_ ? atoi(e_) : 20; }
          static _Thread_local double ssb; static _Thread_local int ssi;
          if (it == 0) { ssb = 1e300; ssi = 0; }
          if (S->tau > 0 && err < 0.5 * ssb) { ssb = err; ssi = it; }
          const int shs_ok = short_stop && fmax(fmax(e_p, e_d), fabs(e_g)) <= 1e-6 && err > opt->tol && S->kap <= S->tau;
          if ((sst > 0 && it - ssi >= sst && ssb <= 1e-6 && ssb > opt->tol && S->kap <= S->tau) || shs_ok) {
              if (S->verbose > 1 || dbg_on()) socp_printf(shs_ok ? "      [the solves fell short of the steps' promise for 3 iterations, double-double not available: the solve stops with its best point]\n" : "      [the stall stop: the largest error not halved in %d iterations (best %.1e at iteration %d)]\n", it - ssi, ssb, ssi);
              R->stalled = 1; status = SOCP_REDUCED;
              if (err < best) { best = err; memcpy(bx, xo, sizeof(double) * (size_t)n); memcpy(by, yo, sizeof(double) * (size_t)m); memcpy(bz, zo, sizeof(double) * (size_t)n); best_e[0] = e_p; best_e[1] = e_d; best_e[2] = e_g; best_e[3] = e_c; R->pobj = pobj; R->dobj = dobj; }
              break;
          } }
        if (err < best && S->tau > 0) { best = err; memcpy(bx, xo, sizeof(double) * (size_t)n); memcpy(by, yo, sizeof(double) * (size_t)m); memcpy(bz, zo, sizeof(double) * (size_t)n); best_e[0] = e_p; best_e[1] = e_d; best_e[2] = e_g; best_e[3] = e_c; R->pobj = pobj; R->dobj = dobj; }
        if (S->verbose > 1 && it == 0) socp_printf("%4d %+.9e %+.9e %8.1e %8.1e %8.1e %8.1e %8.1e\n", it, pobj, dobj, e_p, e_d, e_g, mu, S->tau);
        brisk_itab_row(it, pobj, dobj, e_p, e_d, fabs(e_g), mu, tab_step);
        if (dbg_on() > 1) socp_printf("      [the stopping measure %.2e: primal %.2e, dual %.2e, gap %.2e, x'z %.2e]\n", err, e_p, e_d, e_g, e_c);
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
        /* (28 October 2026, the SOCP developer) the busy-machine rule each iteration, as the SDP solver's: the threads of the
         * parallel loops and of the factorization's dense panels follow the free cores (brisk_team_adapt; never above the
         * count of the start); every kernel gives the same bits for any count */
        if (it > 0) cone_threads_adapt(S->verbose);
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
        if (S->ne && S->it_maxres > 1e-6) S->ne_iron = 1;     /* (BRISK_CONENEIR: from the first iteration whose solves on the normal equations did not refine below 1e-6, for the rest of the solve) */
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
        if (dbg_on()) {      /* (debug, 13 November 2026: the first non-finite value of the iterate and of its scaling) */
            long bx = -1, bz = -1, by = -1, bh = -1, bw = -1; const int n_ = S->n, m_ = S->m;
            for (int j = 0; j < n_ && bx < 0; j++) if (!isfinite(S->x[j])) bx = j;
            for (int j = 0; j < n_ && bz < 0; j++) if (!isfinite(S->z[j])) bz = j;
            for (int i = 0; i < m_ && by < 0; i++) if (!isfinite(S->y[i])) by = i;
            for (int i = 0; i < S->nl && bh < 0; i++) if (!isfinite(S->h[i]) || S->h[i] <= 0) bh = i;
            for (int k = 0; k < S->nq && bw < 0; k++) { if (!isfinite(S->eta[k])) { bw = k; break; } for (int i = 0; i < S->q[k]; i++) if (!isfinite(S->wbar[S->qs[k] + i])) { bw = k; break; } }
            if (bx >= 0 || bz >= 0 || by >= 0 || bh >= 0 || bw >= 0 || !isfinite(S->tau) || !isfinite(S->kap))
                socp_printf("      [not finite at the iteration's start: x %ld, z %ld, y %ld, the linear weights %ld (h %g, x %g, z %g), the cones' scaling %ld; tau %g kappa %g]\n", bx, bz, by, bh, bh >= 0 ? S->h[bh] : 0.0, bh >= 0 ? S->x[S->nf + bh] : 0.0, bh >= 0 ? S->z[S->nf + bh] : 0.0, bw, S->tau, S->kap);
        }
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
            const int sup_ = S->sc || S->sc_lazy; const double lz = (double)(sup_ ? S->slnz : S->F.lnz), tsv = lz * (sup_ ? 3e-9 : 1.5e-9) + (double)S->Ap[n] * 3e-9 + (double)(n + m) * 2e-8;
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
        S->qd_raised = 0;
        kkt_factor_reg(S);
        tq[3] += sc_time() - tq_last; tq_last = sc_time(); td_last = tq_last;
        /* (BRISK_CONEPRECSTOP, below at the errors: the same evidence acted on at once, before a step on the raised factor) */
        { static int prs2_c = -2; const int prs2 = cone_pkg("BRISK_CONEPRECSTOP", &prs2_c); static double ddm2 = -1; if (ddm2 < 0) { const char *d_ = getenv("BRISK_CONEDDMAX"); ddm2 = d_ ? atof(d_) : 5e8; }
          /* (only while the largest of the four errors is still above 1e-6: then closing it, x'z above all, needs mu to fall by
           * orders on a factor at its precision limit; within 1e-6 a few more steps may still meet the tolerance (socp_svmsparse_50kx2000,
           * SOCP: stopped at 3.5e-7, the next step OPTIMAL)) */
          if (prs2 && S->qd_raised && !S->use_dd && (!S->dd_on || S->cflops > ddm2) && fmax(fmax(e_p, e_d), fabs(e_g)) <= 1e-6 && fmax(fmax(e_p, e_d), fmax(fabs(e_g), e_c)) > 1e-6 && S->kap <= S->tau) {
              if (S->verbose > 1 || dbg_on()) socp_printf("      [the double factor broke the quasi-definite bound and double-double is not available: the solve stops with its best point]\n");
              R->stalled = 1; status = SOCP_REDUCED; break;
          } }
        /* the objective is at the tolerance and a residual is not: Newton steps on the equations alone,
         * with the complementarity products kept (dz = -H dx) and tau fixed */
        const double e_obj = fmax(fabs(e_g), e_c), e_inf = fmax(e_p, e_d);
        /* (7 November 2026, the SOCP developer; the owner's SOCP benchmark of 6 November; a test, BRISK_CONEEQBLOCK=1) also when the last
         * step was blocked (below 1e-2) while the objective's errors are within 1e-6: the interior-point direction no longer moves and
         * only the residuals are left, which these steps close. A robust-dose SOCP (P11_00) on the full system: steps of 0.000-0.002
         * at iterations 16-19 with 11-27 refinement steps each (152 of the run's 243 solves), the gap 1.7e-9 .. 1.1e-10, the primal
         * error 3.5e-6 .. 2.4e-7, before these steps finished it at iteration 20 */
        static int eqb = -1; if (eqb < 0) { const char *e_ = getenv("BRISK_CONEEQBLOCK"); eqb = e_ && atoi(e_) == 1; }
        const int eq_blocked = eqb && prev_alpha < 1e-2 && e_obj <= 1e-6 && it > 2;
        if (feas_on && e_inf > opt->tol && (e_obj <= opt->tol || (e_obj <= 1e-6 && e_obj <= 1e-3 * e_inf) || eq_blocked) && nfeas < 12) {
            kkt_solve(S, rD, rP, dx, dy, S->refine_max);
            H_apply(S, dx, t2);
            #pragma omp parallel for schedule(static) num_threads(g_cone_nt) if (cone_vpar(S))
            for (int j = 0; j < n; j++) dz[j] = j < S->nf ? 0.0 : -t2[j];
            double al = 1e300;
            al = cone_step(S, S->x, dx, al); al = cone_step(S, S->z, dz, al);
            const double a = fmin(1.0, 0.99 * al);
            int okf = a > 1e-3;
            for (int j = 0; j < n && okf; j++) if (!isfinite(dx[j]) || !isfinite(dz[j])) okf = 0;
            if (okf) {
                memcpy(sv_x, S->x, sizeof(double) * (size_t)n); memcpy(sv_z, S->z, sizeof(double) * (size_t)n); memcpy(sv_y, S->y, sizeof(double) * (size_t)m);
                feas_pending = 1; feas_prev = e_inf;
                #pragma omp parallel for schedule(static) num_threads(g_cone_nt) if (cone_vpar(S))
                for (int j = 0; j < n; j++) { S->x[j] += a * dx[j]; S->z[j] += a * dz[j]; }
                for (int i = 0; i < m; i++) S->y[i] += a * dy[i];
                nfeas++;
                if (S->verbose > 1) socp_printf("%4d %+.9e %+.9e %8.1e %8.1e %8.1e %8.1e %8.1e %5.3f  equations\n", it + 1, pobj, dobj, e_p, e_d, e_g, mu, S->tau, a);
                continue;
            }
        }
        /* p = K^{-1} [c; b] */
        SOLVE_TAG(1); kkt_solve(S, S->c, S->b, px, py, S->refine_max);
        static int rdef3 = -1; if (rdef3 < 0) { const char *e_ = getenv("BRISK_CONEROTDEF"); rdef3 = e_ ? atoi(e_) >= 3 : 0; }
        if (((S->want_dd && !S->use_dd) || (S->use_dd && S->F.ndyn > 0) || (rdef3 && S->F.ndyn > 0)) && S->rot_def && !S->anyrot) {     /* (=3, a test: replaced pivots alone in double too) */     /* (BRISK_CONEROTDEF=2: also in double-double) */     /* (BRISK_CONEROTDEF: the small cones' blocks lost definiteness as written:
                                                                          * the eigenbasis now, double-double only if pivots are replaced again) */
            const double ts = sc_time();
            S->rot_switch = it + 1; S->rot_mode = 1; kkt_teardown(S); S->kkt_defer = 0;
            const int rc_ = kkt_setup(S);
            S->t_setup += sc_time() - ts; S->want_dd = 0; S->rot_def = 0; S->rot_want = 1;     /* (rot_want: kept for accuracy, not switched back for its cost) */
            if (S->verbose > 1) socp_printf("      [the small cones in the eigenbasis of their scaling from here: their blocks were not definite as written (%.3fs)]\n", sc_time() - ts);
            if (rc_) { status = SOCP_NUMERR; break; }
            kkt_fill(S); kkt_factor_reg(S);
            SOLVE_TAG(1); kkt_solve(S, S->c, S->b, px, py, S->refine_max);
        }
        if (S->want_dd && S->dd_on && !S->use_dd) {       /* replaced pivots and a solve that did not refine: this matrix again, in double-double */
            kkt_fill(S); kkt_factor_reg(S);
            SOLVE_TAG(1); kkt_solve(S, S->c, S->b, px, py, S->refine_max);
        }
        double denom0 = dotv(m, S->b, py) - dotv(n, S->c, px);       /* + kappa / tau */
        /* (9 November 2026, the SOCP developer, a test: BRISK_CONETAUEQ=1) tau's step from the equations as they are meant to hold, not
         * from the computed solutions' misses: with -H px + A'py = c, A px + dl py = b (dl the rows' term of the system solved), the
         * denominator b'py - c'px is (H px)'px + dl |py|^2, a sum of nonnegative terms instead of the difference of two numbers that
         * agree to seven digits at the end; and in the numerator c'qx = -(H px)'qx + py'(eta rP - dl qy), the miss of qx in the rows
         * left out. On heterogeneous cones (SOCP) that miss (5.5e-5, GMRES stops at its relative target) against terms of 3e-6 swung
         * dtau / tau to -20 and +12 and lost the last three iterations */
        static int taueq = -1; if (taueq < 0) { const char *e_ = getenv("BRISK_CONETAUEQ"); taueq = e_ ? atoi(e_) : 0; }
        double *hpx_ = NULL, dl_op = 0.0, denom_c = denom0;
        if (taueq) {
            dl_op = S->use_dd ? S->sreg : (S->ne && S->ne_ok && S->ne_dl > 0 ? fmin(S->sreg0, S->ne_dl) : S->sreg0);
            { static double dlx_t = -2; if (dlx_t < -1) { const char *e_ = getenv("BRISK_CONENEDLX"); dlx_t = e_ ? atof(e_) : -1.0; }
              if (dlx_t >= 0 && !S->use_dd && S->ne && S->ne_ok) dl_op = dlx_t; }
            { static _Thread_local double *tb_ = NULL; static _Thread_local size_t tbn_ = 0; if (tbn_ < (size_t)n + 1) { free(tb_); tbn_ = (size_t)n + 1; tb_ = (double *)xcalloc(tbn_, sizeof(double)); } hpx_ = tb_; }   /* (scratch only: its values never outlive the iteration) */
            H_apply(S, px, hpx_);
            double hpp = 0; for (int j = S->nf; j < n; j++) hpp += hpx_[j] * px[j];
            denom_c = denom0; denom0 = hpp + dl_op * dotv(m, py, py);
        }
        double sigma = 0.0, alpha = 0.0, aaff = 0.0, sfi = step_frac;      /* (sfi: this iteration's fraction of the step to the boundary) */
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
            #pragma omp parallel for schedule(static) num_threads(g_cone_nt) if (cone_vpar(S))
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
            #pragma omp parallel for schedule(static) num_threads(g_cone_nt) if (cone_vpar(S))
            for (int j = 0; j < n; j++) rhsx[j] = etaf * rD[j] - r3[j];
            for (int i = 0; i < m; i++) rhsy[i] = etaf * rP[i];
            static int refpred_t = -1; if (refpred_t < 0) { const char *e_ = getenv("BRISK_CONEREFPRED"); refpred_t = e_ ? atoi(e_) : 0; }   /* (11 November 2026, a test: the predictor refined as the corrector) */
            SOLVE_TAG(pass == 0 ? 2 : 3); kkt_solve(S, rhsx, rhsy, qx, qy, pass == 0 ? (refpred_t ? S->refine_max : S->ref_pred) : S->refine_max);
            const double r4 = -etaf * rG;
            /* (9 November 2026, the SOCP developer, a test: BRISK_CONEQREF=f) q refined against its own rows' equation before tau's step:
             * GMRES stops q at its relative target and leaves a miss w = eta rP - dl qy - A qx in the rows, which enters tau's step through
             * c'qx (heterogeneous cones, SOCP: 5.5e-5 against true terms of 3e-6, dtau / tau -20 and +12, the last iterations lost). When
             * |py'w| exceeds f of the numerator, the miss is solved for (K d = [0; w], one more solve) and added to q, so that the step's
             * formula sees a q that meets its equations; the formula itself is unchanged */
            { static double qrf = -2; if (qrf < -1) { const char *e_ = getenv("BRISK_CONEQREF"); qrf = e_ ? atof(e_) : 0.5; }   /* (12 November 2026: 0.5 the default; 0 off) */
              if (qrf > 0 && S->ne && S->ne_ok && !S->use_dd) {
                  const double dlq = S->ne_dl > 0 ? fmin(S->sreg0, S->ne_dl) : S->sreg0;
                  double *w = S->wdy, *ex = S->wdx, *ey = S->wdy2, *ez = S->wdz;
                  for (int i = 0; i < m; i++) w[i] = etaf * rP[i] - dlq * qy[i];
                  A_mul(S, -1.0, qx, w);
                  const double pw = fabs(dotv(m, py, w)), nu_ = fabs(r4 + r5 / S->tau - dotv(m, S->b, qy) + dotv(n, S->c, qx));
                  /* (11 November 2026, the SOCP developer, BRISK_CONEQREFMODE=2) the predictor solved again with the corrector's refinement
                   * instead of a separate solve for the rows' miss alone: one refined solve where the miss decides tau's step, the same
                   * system as the step's (multiscale rows, SOCP: the separate solve for a right-hand side of the miss alone took the
                   * refinement to its cap every iteration, twice the time) */
                  static int qmode = -1; if (qmode < 0) { const char *e_ = getenv("BRISK_CONEQREFMODE"); qmode = e_ ? atoi(e_) : 2; }   /* (12 November 2026: 2 the default) */
                  if (pw > qrf * nu_ && pw > 0 && qmode == 2 && pass == 0) {
                      SOLVE_TAG(2); kkt_solve(S, rhsx, rhsy, qx, qy, S->refine_max);
                      if (dbg_on()) socp_printf("      [the predictor solved again with refinement before tau's step: py'(miss) %.2e against the numerator %.2e]\n", pw, nu_);
                  } else
                  if (pw > qrf * nu_ && pw > 0 && qmode != 2) {      /* (mode 2: the corrector is refined already) */
                      for (int j = 0; j < n; j++) ez[j] = 0.0;
                      SOLVE_TAG(6); kkt_solve(S, ez, w, ex, ey, S->refine_max);
                      for (int j = 0; j < n; j++) qx[j] += ex[j];
                      for (int i = 0; i < m; i++) qy[i] += ey[i];
                      if (dbg_on()) socp_printf("      [q refined against its rows before tau's step: py'(miss) %.2e against the numerator %.2e]\n", pw, nu_);
                  }
              } }
            if (taueq) {
                double hq = 0; for (int j = S->nf; j < n; j++) hq += hpx_[j] * qx[j];
                double prp = 0, pqy = 0; for (int i = 0; i < m; i++) { prp += py[i] * rP[i]; pqy += py[i] * qy[i]; }
                const double cqx = -hq + etaf * prp - dl_op * pqy, cqx0 = dotv(n, S->c, qx);
                const double nu_eq = r4 + r5 / S->tau - dotv(m, S->b, qy) + cqx, nu_0 = r4 + r5 / S->tau - dotv(m, S->b, qy) + cqx0;
                /* (BRISK_CONETAUEQ=2) the equations' form only where the computed product is dominated by the solve's miss (the
                 * difference of the two numerators, against the equations' numerator): elsewhere the computed products are kept */
                if (taueq == 2 && !(fabs(cqx0 - cqx) > 0.5 * fabs(nu_eq))) dtau = nu_0 / (denom_c + S->kap / S->tau);
                else if (taueq == 3) {     /* (both products from the equations: b'qy = (A px + dl py)'qy = px'(eta rD - r3 + H qx) + dl py'qy,
                                            * with A'qy = eta rD - r3 + H qx (the dual rows, as solved); the misses of both rows left out) */
                    double prd = 0; for (int j = S->nf; j < n; j++) prd += px[j] * rhsx[j];
                    for (int j = 0; j < S->nf; j++) prd += px[j] * rhsx[j];
                    const double bqy = prd + hq + dl_op * pqy;
                    dtau = (r4 + r5 / S->tau - bqy + cqx) / (denom0 + S->kap / S->tau);
                }
                else if (taueq == 4) dtau = nu_eq / (denom_c + S->kap / S->tau);     /* (the numerator from the equations, the denominator as computed) */
                else if (taueq == 5) {
                    /* (9 November 2026, the SOCP developer, a test: BRISK_CONETAUEQ=5) tau held where its step is not determined. Each part of
                     * tau's step is computed twice, from the computed solutions and from the equations they are meant to satisfy; where the
                     * two disagree by more than half their size (the numerator or the denominator), the step's tau component is noise: near
                     * the end of a feasible problem (kappa below tau) the model has an almost free direction, the scaling of the whole point,
                     * and no form computes it reliably. There dtau = 0, the infeasible-start Newton step: the residuals and the products fall
                     * as before, only the drift along the ray is gone. Where the two agree (the robust families, SOCP, to four digits at every
                     * iteration) the ordinary step is taken unchanged */
                    const double dn = fabs(nu_0 - nu_eq), dd = fabs(denom_c - denom0);
                    const int undet = S->kap < S->tau && (dn > 0.5 * fmax(fabs(nu_0), fabs(nu_eq)) || dd > 0.5 * fmax(fabs(denom_c), fabs(denom0)));
                    dtau = undet ? 0.0 : nu_0 / (denom_c + S->kap / S->tau);
                    if (undet && dbg_on()) socp_printf("      [tau held (pass %d): its step not determined (numerators %.2e and %.2e, denominators %.2e and %.2e)]\n", pass, nu_0, nu_eq, denom_c, denom0);
                }
                else dtau = nu_eq / (denom0 + S->kap / S->tau);
            } else
            dtau = (r4 + r5 / S->tau - dotv(m, S->b, qy) + dotv(n, S->c, qx)) / (denom0 + S->kap / S->tau);
            /* (9 November 2026, the SOCP developer, a test: BRISK_CONEDLTAU=f) the rows' regularization of the system the direction is
             * refined against (A dx + dl dy = eta rP on the normal equations' route) enters tau's step through c'qx = -(H px)'qx + py'(A qx),
             * as -dl py'qy; at the end |qy| is large and the denominator b'py - c'px small, and that term alone swung dtau / tau between +8
             * and -8 (heterogeneous cones, SOCP: tau grew 55 times in one step, the last iterations lost). From the iteration where the
             * term exceeds f of the numerator, the solves are refined against the exact system (dl 0 in the operator; the factorization
             * keeps its regularization, the preconditioner). Measured, not tuned: elsewhere the regularization stays, which the exact
             * system everywhere cost (multiscale rows A02_07, SOCP, 14 -> 20 iterations) */
            { static double dlt = -2; if (dlt < -1) { const char *e_ = getenv("BRISK_CONEDLTAU"); dlt = e_ ? atof(e_) : 0.0; }
              if (dlt > 0 && pass == 0 && !S->dl_exact && S->ne && S->ne_ok && !S->use_dd) {
                  const double dl_ = S->ne_dl > 0 ? fmin(S->sreg0, S->ne_dl) : S->sreg0;
                  const double tr_ = dl_ * fabs(dotv(m, py, qy)), nu_ = fabs(r4 + r5 / S->tau - dotv(m, S->b, qy) + dotv(n, S->c, qx));
                  if (tr_ > dlt * nu_ && tr_ > 0) { S->dl_exact = 1; if (dbg_on()) socp_printf("      [tau's step: the rows' regularization's term %.2e against the numerator %.2e: the exact system from here]\n", tr_, nu_); }
              } }
            if (dbg_on() > 1 && pass == 0) {     /* (9 November 2026, debug: tau's step and its parts; b'py and c'px apart: the denominator's cancellation) */
                const double bpy = dotv(m, S->b, py), cpx = dotv(n, S->c, px), bqy = dotv(m, S->b, qy), cqx = dotv(n, S->c, qx);
                socp_printf("      [tau's step: dtau/tau %.3e; numerator %.3e (r4 %.3e, r5/tau %.3e, -b'qy %.3e, c'qx %.3e); denominator %.3e (b'py %.6e, c'px %.6e, kappa/tau %.3e)]\n",
                            dtau / S->tau, r4 + r5 / S->tau - bqy + cqx, r4, r5 / S->tau, -bqy, cqx, denom0 + S->kap / S->tau, bpy, cpx, S->kap / S->tau);
                /* the same products from the system's own equations: with -H px + A'py = c and A px = b (the regularization aside),
                 * c'qx = -(H px)'qx + py'(A qx), b'qy = (A px)'qy, and b'py - c'px = (H px)'px */
                double *hp = (double *)xcalloc((size_t)n + 1, sizeof(double)), *aq = (double *)xcalloc((size_t)m + 1, sizeof(double)), *ap = (double *)xcalloc((size_t)m + 1, sizeof(double));
                H_apply(S, px, hp); A_mul(S, 1.0, qx, aq); A_mul(S, 1.0, px, ap);
                double hpq = 0, hpp = 0; for (int j = S->nf; j < n; j++) { hpq += hp[j] * qx[j]; hpp += hp[j] * px[j]; }
                const double cqx2 = -hpq + dotv(m, py, aq), bqy2 = dotv(m, ap, qy), bpy2 = dotv(m, ap, py);
                socp_printf("      [tau's step from the equations: -b'qy %.3e, c'qx %.3e (-(H px)'qx %.3e, py'A qx %.3e); (H px)'px %.3e, b'py %.6e; the numerator %.3e, dtau/tau %.3e]\n",
                            -bqy2, cqx2, -hpq, dotv(m, py, aq), hpp, bpy2, r4 + r5 / S->tau - bqy2 + cqx2, (r4 + r5 / S->tau - bqy2 + cqx2) / (hpp + S->kap / S->tau) / S->tau);
                { const double dlq = (S->ne_dl > 0 ? fmin(S->sreg0, S->ne_dl) : S->sreg0); double prp = 0; for (int i = 0; i < m; i++) prp += py[i] * rP[i];
                  socp_printf("      [tau's step: py'(A qx) %.3e = eta py'rP %.3e - dl py'qy %.3e (dl %.1e, eta %.3f) + the solve's miss %.3e]\n", dotv(m, py, aq), (1.0 - sigma) * prp, dlq * dotv(m, py, qy), dlq, 1.0 - sigma, dotv(m, py, aq) - (1.0 - sigma) * prp + dlq * dotv(m, py, qy)); }
                free(hp); free(aq); free(ap);
            }
            #pragma omp parallel for schedule(static) num_threads(g_cone_nt) if (cone_vpar(S))
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
            #pragma omp parallel for schedule(static) num_threads(g_cone_nt) if (cone_vpar(S))
            for (int j = 0; j < n; j++) dz[j] = j < S->nf ? 0.0 : r3[j] - t2[j];
            dual_refine(S, opt->tol, etaf, rD, dtau, byv, cx, ncs, dx, dy, dz, t2);
            primal_refine(S, opt->tol, etaf, rP, dtau, dx, dy, dz, t2);
            /* step length */
            double al = 1e300;
            al = cone_step(S, S->x, dx, al); al = cone_step(S, S->z, dz, al);
            if (dtau < 0) al = fmin(al, -S->tau / dtau);
            if (dkap < 0) al = fmin(al, -S->kap / dkap);
            if (pass == 0) {
                const double aa = fmin(1.0, al);
                aaff = aa;
                if (dbg_on() >= 2) {     /* (12 November 2026, debug: the cones that limit the predictor's own step: the side, the step, lambda's
                                          * eigenvalues squared over mu, and the relative miss of the cone's equation lambda o (W dx + W^-1 dz) = -lambda o lambda) */
                    int kb[5]; double ab[5]; char sb[5]; for (int t = 0; t < 5; t++) { kb[t] = -1; ab[t] = 1e300; sb[t] = ' '; }
                    double *u1 = (double *)malloc(sizeof(double) * ((size_t)n + 1)), *u2 = (double *)malloc(sizeof(double) * ((size_t)n + 1));
                    for (int k = 0; k < S->nq; k++) {
                        const int d = S->q[k], s0 = S->qs[k];
                        const double ax_ = soc_step(d, S->x + s0, dx + s0, 1e300), az_ = soc_step(d, S->z + s0, dz + s0, 1e300);
                        const double a_ = fmin(ax_, az_); const char sd = ax_ <= az_ ? 'x' : 'z';
                        for (int t = 0; t < 5; t++) if (a_ < ab[t]) { for (int r = 4; r > t; r--) { ab[r] = ab[r - 1]; kb[r] = kb[r - 1]; sb[r] = sb[r - 1]; } ab[t] = a_; kb[t] = k; sb[t] = sd; break; }
                    }
                    for (int t = 0; t < 5 && kb[t] >= 0; t++) {
                        const int k = kb[t], d = S->q[k], s0 = S->qs[k]; const double *l = S->lam + s0;
                        W_apply(S, k, 0, dx + s0, u1 + s0); W_apply(S, k, 1, dz + s0, u2 + s0);
                        double ln = 0; for (int i = 1; i < d; i++) ln += l[i] * l[i]; ln = sqrt(ln);
                        double r0 = 0, rn = 0, l2n = 0, w0 = 0;     /* (lambda o (u1 + u2) + lambda o lambda) */
                        for (int i = 0; i < d; i++) w0 += l[i] * (u1[s0 + i] + u2[s0 + i] + l[i]);
                        r0 = w0; double ll0 = 0; for (int i = 0; i < d; i++) ll0 += l[i] * l[i]; l2n = ll0 * ll0;
                        for (int i = 1; i < d; i++) { const double v = l[0] * (u1[s0 + i] + u2[s0 + i] + l[i]) + (u1[s0] + u2[s0] + l[0]) * l[i]; rn += v * v; l2n += 4.0 * l[0] * l[0] * l[i] * l[i]; }
                        socp_printf("      [the predictor's step limited by cone %d (dimension %d), %c: %.3f; lambda eigenvalues^2 / mu %.2e, %.2e; the cone equation's miss %.1e]\n",
                                    k, d, sb[t], ab[t], (l[0] + ln) * (l[0] + ln) / mu, (l[0] - ln) * (l[0] - ln) / mu, sqrt(r0 * r0 + rn) / sqrt(l2n));
                    }
                    free(u1); free(u2);
                }
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
                /* (12 November 2026, the SOCP developer, a test: BRISK_CONESIG0=s,k) sigma at least s in the first k iterations: on multiscale
                 * rows A02_00 (SOCP) the first full step cut mu by 6,400 and the later steps by 7-45, where MOSEK's first cut it by 30
                 * and its end was superlinear */
                { static double sg0 = -1; static int sgk = 1; if (sg0 < 0) { const char *e_ = getenv("BRISK_CONESIG0"); sg0 = 0; if (e_) sscanf(e_, "%lf,%d", &sg0, &sgk); }
                  if (sg0 > 0 && it < sgk && sigma < sg0) sigma = sg0; }
                if (sigma < 1e-12) sigma = 1e-12;
                /* (29 October 2026, the SOCP developer; BRISK_CONESTEPA=f[,s], 0: off) the step to 0.9999 of the way to the
                 * boundary, not 0.99, in an iteration whose centring parameter is at most 1e-6: the predictor's own step then went
                 * nearly all the way (sigma = (1 - alpha_aff)^3), the iterate needs no centring, and a fraction 0.99 caps the
                 * reduction of every residual at a factor 100 an iteration. The scaling family of the owner's SOCP benchmark
                 * (an exact least-squares fit, the optimum 0, the residuals cut by 100 each iteration): 7 -> 5 iterations, as
                 * MOSEK; the 77 held-out SOCPs 696 -> 666 iterations, the 39 mezo SOCPs 948 -> 938, every status unchanged.
                 * A fraction 0.999 or 0.9999 in every iteration costs the classification SOCPs 13 -> 17 (TRICKS 8). Only when the
                 * step to the boundary is itself at least 0.99: a step blocked well before the full step (sched_50_50_orig,
                 * SOCP: the corrected step blocked at 0.72 with sigma 1e-12) went 0.9999 of the way to that cone's boundary and
                 * the next steps were 0.004 (21 -> 53 iterations, a second attempt) */
                { static double sa = -1, ss = 1e-6; if (sa < 0) { const char *e_ = getenv("BRISK_CONESTEPA"); sa = e_ ? atof(e_) : 0.9999; if (e_ && strchr(e_, ',')) ss = atof(strchr(e_, ',') + 1); }
                  if (sa > 0 && !(opt->step > 0) && sigma <= ss) sfi = sa; }
                if (arcq) { memcpy(axa, dx, sizeof(double) * (size_t)n); memcpy(aza, dz, sizeof(double) * (size_t)n); atau = dtau; akap = dkap; }
            } else alpha = fmin(1.0, (al >= 0.99 ? sfi : step_frac) * al);
        }
        int arc_taken = 0;
        if (arcq == 2) { memcpy(alx, dx, sizeof(double) * (size_t)n); memcpy(alz, dz, sizeof(double) * (size_t)n); memcpy(aly, dy, sizeof(double) * (size_t)m); altau = dtau; alkap = dkap; }
        if (arcq == 1 && alpha < arct && !S->use_dd) {
            double *tv = ctv;
            for (int i = 0; i < S->nf; i++) tv[i] = 0.0;
            for (int i = S->nf; i < S->nf + S->nl; i++) tv[i] = -2.0 * axa[i] * aza[i];
            #pragma omp parallel for schedule(static) num_threads(g_cone_nt) if (cone_vpar(S))
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
            #pragma omp parallel for schedule(static) num_threads(g_cone_nt) if (cone_vpar(S))
            for (int j = 0; j < n; j++) { dxs[j] = dx[j] - 0.5 * dxa[j]; dzs[j] = dz[j] - 0.5 * dza[j]; }   /* d1 = d_line - d2 / 2 */
            for (int i = 0; i < m; i++) ay1[i] = dy[i] - 0.5 * dya[i];
            const double dt1 = dtau - 0.5 * dt2, dk1 = dkap - 0.5 * dk2;
            int fin = 1; for (int j = 0; j < n && fin; j++) if (!isfinite(dxa[j]) || !isfinite(dza[j])) fin = 0;
            if (fin) {
                double lo = 0, hi = M_PI / 2;
                if (arc_inside(S, hi, dxs, dzs, dt1, dk1, dxa, dza, dt2, dk2)) lo = hi;
                else for (int b = 0; b < 40; b++) { const double mid = 0.5 * (lo + hi); if (arc_inside(S, mid, dxs, dzs, dt1, dk1, dxa, dza, dt2, dk2)) lo = mid; else hi = mid; }
                const double aa = (lo >= 0.99 ? sfi : step_frac) * lo;
                if (sin(aa) > alpha) {
                    const double sa = sin(aa), ca = 1.0 - cos(aa);
                    #pragma omp parallel for schedule(static) num_threads(g_cone_nt) if (cone_vpar(S))
                    for (int j = 0; j < n; j++) { dx[j] = sa * dxs[j] + ca * dxa[j]; dz[j] = sa * dzs[j] + ca * dza[j]; }
                    for (int i = 0; i < m; i++) dy[i] = sa * ay1[i] + ca * dya[i];
                    dtau = sa * dt1 + ca * dt2; dkap = sa * dk1 + ca * dk2;
                    if (dbg_on()) socp_printf("      [arc: the line's step %.3f, the arc's angle %.3f (sin %.3f)]\n", alpha, aa, sa);
                    alpha = 1.0; arc_taken = 1; narc++;
                }
            }
        }
        td[0] += sc_time() - td_last; td_last = sc_time();     /* (the predictor, tau's step, the corrector) */
        /* centrality correctors (Gondzio): at a longer trial step the complementarity products (the
         * eigenvalues of the scaled Jordan products for the cones) are moved into [bmin, bmax] x the
         * target; the correction is kept when it lengthens the step */
        int ncor = 0;
        /* (13 November 2026, the SOCP developer, a test: BRISK_CONEMCCGAIN=g) a further centrality corrector only where the previous one
         * paid: its relative step gain (alpha' - alpha) / alpha at least g (the first always tried). A corrector costs its solves and a pass
         * of cone arithmetic; with a cheap factorization (db-joint-soerensen, SOCP, METIS's order: a factorization about ten solves) the
         * second and third cost more than the iterations they save: to mu 1e-10, 63.0 s with at most 3, 56.9 with 2, 54.6 with 1 */
        static double mgain = -1; if (mgain < 0) { const char *e_ = getenv("BRISK_CONEMCCGAIN"); mgain = e_ ? atof(e_) : 0.0; }
        double mprev = 1e300;
        for (int kc_ = 0; kc_ < mcc_now && alpha < 0.999; kc_++) {
            if (kc_ > 0 && mgain > 0 && mprev < mgain) break;
            const double tcc0 = sc_time();     /* (debug: the corrector's parts, g_ctime) */
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
            #pragma omp parallel for schedule(static) num_threads(g_cone_nt) if (cone_vpar(S))
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
            /* (12 November 2026, the SOCP developer, a test: BRISK_CONEMCCMEAN=1) the correction made mean-free, as Andersen, Roos and Terlaky
             * do (Math. Program. 95, 2003, section 4.3, eq. (45): mu_c = gamma mu - (e'f + f_tk) / (k + 1)): it only moves complementarity
             * between the pairs, so the step's own reduction of x'z + tau kappa stays the predictor-corrector's. Without it the lifted small
             * products (many) outweigh the clipped large ones and the correction raises the step's target */
            { static int mmean = -1; if (mmean < 0) { const char *e_ = getenv("BRISK_CONEMCCMEAN"); mmean = e_ ? atoi(e_) : 0; }
              if (mmean) {
                double sum = ttk;
                for (int i = S->nf; i < S->nf + S->nl; i++) sum += tv[i];
                for (int k = 0; k < S->nq; k++) sum += tv[S->qs[k]];
                const double dl_ = -sum / nu;
                for (int i = S->nf; i < S->nf + S->nl; i++) tv[i] += dl_;
                for (int k = 0; k < S->nq; k++) tv[S->qs[k]] += dl_;
                ttk += dl_;
              } }
            double tc_, kc2;
            const double tcc1 = sc_time(); g_ctime[0] += tcc1 - tcc0;
            /* (13 November 2026, the SOCP developer, a test: BRISK_CONEMCCTOL=t) the corrector's solve refined to its own relative tolerance t,
             * looser than the direction's: a corrector only has to lengthen the step, and it is kept only if it does. Early one application
             * of the factor meets t and nothing is refined; late the refinement keeps the corrector's equations satisfied enough. Unrefined
             * (BRISK_CONEMCCREF=0): db-joint-soerensen (SOCP) to mu 1e-10 47.9 -> 37.7 s, the 39 development SOCPs 0.913, but
             * socp_socopf_3012wp_k (SOCP) OPTIMAL -> numerical difficulties */
            static double mctol = -1; if (mctol < 0) { const char *e_ = getenv("BRISK_CONEMCCTOL"); mctol = e_ ? atof(e_) : 0.0; }
            const double gtol_sv = S->gm_tol; if (mctol > 0) S->gm_tol = fmax(S->gm_tol, mctol);
            /* (13 November 2026, the SOCP developer, a test: BRISK_CONEMCCREF=-1) the corrector's solve unrefined while the factor is a
             * factor of the matrix (no pivot replaced): one application of it then solves the corrector's equations to the factor's
             * accuracy; with replaced pivots it is a factor of another matrix, and the refinement (up to 10 steps) makes the corrector's
             * direction one of the system (joint_HO_01, SOCP, its endgame in double-double with replaced pivots: unrefined, steps of
             * 0.000 to 0.03 from iteration 53 and REDUCED at 72 iterations; refined, OPTIMAL at 56) */
            const int mref_ = mcc_ref >= 0 ? mcc_ref : S->F.ndyn > 0 ? 10 : 0;
            corr_solve(&C, tv, ttk, dxa, dya, dza, &tc_, &kc2, mref_);
            S->gm_tol = gtol_sv;
            if (dbg_on()) { long nf_ = 0, nc_ = 0, ny_ = 0;     /* (debug: a corrector's direction not finite, by part) */
                for (int j = 0; j < S->nf; j++) if (!isfinite(dxa[j])) nf_++;
                for (int j = S->nf; j < n; j++) if (!isfinite(dxa[j]) || !isfinite(dza[j])) nc_++;
                for (int i = 0; i < m; i++) if (!isfinite(dya[i])) ny_++;
                if (nf_ || nc_ || ny_ || !isfinite(tc_)) socp_printf("      [centrality corrector %d: its direction not finite: %ld free, %ld cone or linear, %ld y entries, dtau %g (use_dd %d)]\n", kc_ + 1, nf_, nc_, ny_, tc_, S->use_dd); }
            const double tcc2 = sc_time(); g_ctime[1] += tcc2 - tcc1;
            #pragma omp parallel for schedule(static) num_threads(g_cone_nt) if (cone_vpar(S))
            for (int j = 0; j < n; j++) { dxa[j] += dx[j]; dza[j] += dz[j]; }
            const double dt2 = dtau + tc_, dk2 = dkap + kc2;
            double a2 = 1e300;
            a2 = cone_step(S, S->x, dxa, a2); a2 = cone_step(S, S->z, dza, a2);
            if (dt2 < 0) a2 = fmin(a2, -S->tau / dt2);
            if (dk2 < 0) a2 = fmin(a2, -S->kap / dk2);
            const double al2 = fmin(1.0, (a2 >= 0.99 ? sfi : step_frac) * a2);
            const double tcc3 = sc_time(); g_ctime[2] += tcc3 - tcc2;
            if (!(al2 >= alpha + mcc_acc * mcc_da)) { g_ctime[4] += 1; break; }
            memcpy(dx, dxa, sizeof(double) * (size_t)n); memcpy(dz, dza, sizeof(double) * (size_t)n);
            for (int i = 0; i < m; i++) dy[i] += dya[i];
            g_ctime[3] += sc_time() - tcc3; g_ctime[5] += 1;
            mprev = alpha > 0 ? (al2 - alpha) / alpha : 1e300;
            dtau = dt2; dkap = dk2; alpha = al2; ncor++;
        }
        S->ncor_total += ncor;
        td[1] += sc_time() - td_last; td_last = sc_time();     /* (the extra correctors) */
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
            #pragma omp parallel for schedule(static) num_threads(g_cone_nt) if (cone_vpar(S))
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
            #pragma omp parallel for schedule(static) num_threads(g_cone_nt) if (cone_vpar(S))
            for (int j = 0; j < n; j++) { dxs[j] = alx[j] - 0.5 * dxa[j]; dzs[j] = alz[j] - 0.5 * dza[j]; }
            for (int i = 0; i < m; i++) ay1[i] = aly[i] - 0.5 * dya[i];
            const double dt1 = altau - 0.5 * dt2, dk1 = alkap - 0.5 * dk2;
            int fin = 1; for (int j = 0; j < n && fin; j++) if (!isfinite(dxa[j]) || !isfinite(dza[j])) fin = 0;
            if (fin) {
                double lo = 0, hi = M_PI / 2;
                if (arc_inside(S, hi, dxs, dzs, dt1, dk1, dxa, dza, dt2, dk2)) lo = hi;
                else for (int b = 0; b < 40; b++) { const double mid = 0.5 * (lo + hi); if (arc_inside(S, mid, dxs, dzs, dt1, dk1, dxa, dza, dt2, dk2)) lo = mid; else hi = mid; }
                const double aa = (lo >= 0.99 ? sfi : step_frac) * lo;
                if (sin(aa) > alpha) {
                    const double sa = sin(aa), ca = 1.0 - cos(aa);
                    #pragma omp parallel for schedule(static) num_threads(g_cone_nt) if (cone_vpar(S))
                    for (int j = 0; j < n; j++) { dx[j] = sa * dxs[j] + ca * dxa[j]; dz[j] = sa * dzs[j] + ca * dza[j]; }
                    for (int i = 0; i < m; i++) dy[i] = sa * ay1[i] + ca * dya[i];
                    dtau = sa * dt1 + ca * dt2; dkap = sa * dk1 + ca * dk2;
                    if (dbg_on()) socp_printf("      [arc: the corrected line's step %.3f, the arc's angle %.3f (sin %.3f)]\n", alpha, aa, sa);
                    alpha = 1.0; arc_taken = 1; narc++;
                }
            }
        }
        /* the correctors' directions were added: the same refinement of the sum, and its step length */
        if (!arc_taken && (dual_refine(S, opt->tol, 1.0 - sigma, rD, dtau, byv, cx, ncs, dx, dy, dz, t2) + primal_refine(S, opt->tol, 1.0 - sigma, rP, dtau, dx, dy, dz, t2))) {
            double a2 = 1e300;
            a2 = cone_step(S, S->x, dx, a2); a2 = cone_step(S, S->z, dz, a2);
            if (dtau < 0) a2 = fmin(a2, -S->tau / dtau);
            if (dkap < 0) a2 = fmin(a2, -S->kap / dkap);
            alpha = fmin(alpha, fmin(1.0, (a2 >= 0.99 ? sfi : step_frac) * a2));
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
            socp_printf("      [steps to the boundary: lin x %.3g (%d: x %.1e z %.1e) lin z %.3g (%d) cone x %.3f cone z %.3f tau %.3f kap %.3f; %d linear products below mu/100]\n",
                        axl, ixl, ixl >= 0 ? S->x[ixl] : 0, ixl >= 0 ? S->z[ixl] : 0, azl, izl, axq, azq, at, ak, nsmall);
        }
        {   /* (18 October 2026, LP/SOCP: a direction that is not finite is not taken: the point stays, the iteration
             * ends as a numerical failure; a branch-flow relaxation's last double-double step had tau = NaN, and the
             * attempt then ended on an earlier point) */
            int fin = isfinite(alpha) && isfinite(dtau) && isfinite(dkap);
            for (int j = 0; j < n && fin; j++) if (!isfinite(dx[j]) || !isfinite(dz[j])) fin = 0;
            for (int i = 0; i < m && fin; i++) if (!isfinite(dy[i])) fin = 0;
            /* (13 November 2026, BRISK_CONEPARTNE) on the partial normal equations a direction whose last solve missed by more than a tenth
             * of its right-hand side is not taken either (db-joint-soerensen, SOCP: 6.2e5 at iteration 49, the step taken, the scaling's
             * update failed and the solve ended): the cones are chosen again at this mu and the system set up again, or, chosen at nearly
             * this mu already, the full system from here */
            /* (BRISK_CONEPARTNEGUARD=0, a test: no such check; the step's own check of the residuals' fall judges the direction) */
            static int pguard = -1; if (pguard < 0) { const char *e_ = getenv("BRISK_CONEPARTNEGUARD"); pguard = e_ ? atoi(e_) : 1; }
            if (dbg_on() && fin && S->ne_part && S->ne && S->last_res > 0.1 && !pguard) socp_printf("      [the partial normal equations' last solve missed by %.1e: the direction taken (no check)]\n", S->last_res);
            if (pguard && fin && S->ne_part && S->ne && S->ne_ok && S->last_res > 0.1) {
                static double pne2 = -2; if (pne2 < -1) { const char *e_ = getenv("BRISK_CONEPARTNE"); pne2 = e_ ? atof(e_) : -1.0; }
                const double mu_ = S->tau > 0 ? mu : 0;
                if (pne2 >= 0 && mu_ > 0 && mu_ <= 0.5 * S->ne_part_mu) {
                    const double ts_ = sc_time(); const int nk_ = partne_build(S, mu_, pne2); S->t_setup += sc_time() - ts_;
                    if (dbg_on() || S->verbose > 1) socp_printf("      [the partial normal equations' solve missed by %.1e: the cones chosen again at mu %.1e, %d kept; %s]\n", S->last_res, mu_, nk_, S->ne ? "the iteration done again" : "the full system from here");
                } else {
                    if (dbg_on() || S->verbose > 1) socp_printf("      [the partial normal equations' solve missed by %.1e: the full system from here]\n", S->last_res);
                    ne_free(S); S->ne_ok = 0; S->ne_left = it;
                }
                continue;
            }
            if (!fin) {
                /* (4 November 2026, the SOCP developer, with BRISK_CONENEBACK) after the return to the normal equations a
                 * direction that is not finite comes from their factor at the limit of double precision (db-joint-soerensen,
                 * SOCP: 2,452 tiny pivots at iteration 51): neither double-double nor a raised regularization exists on that
                 * route, and the three attempts that follow repeat it (5 s); the solve ends with its best point */
                if (S->ne_back && S->ne && S->ne_ok) {
                    /* (12 November 2026, the SOCP developer) not the end while the full system remains: the normal equations square the
                     * scaling's range (cones of order 3 near their boundary: H^-1's eigenvalues from 1/mu to mu), and past about 1e16
                     * their matrix cannot hold the small directions in double however it is formed; the full system holds them, its
                     * double factor kept by the raised regularization (db-joint-soerensen, SOCP: on the full system no direction is
                     * non-finite to the end). The normal equations are left for good (no return) and the iteration done again.
                     * BRISK_CONENENANEND=1: the end, as before */
                    static int nanend = -1; if (nanend < 0) nanend = getenv("BRISK_CONENENANEND") != NULL;
                    if (nanend) {
                        if (dbg_on()) socp_printf("      [the direction is not finite on the normal equations after the return: the end]\n");
                        status = SOCP_NUMERR; break;
                    }
                    /* (13 November 2026, the SOCP developer, a test: BRISK_CONEPARTNE with BRISK_CONEPARTNELATE=1) here, where the normal
                     * equations kept past tiny pivots give out (db-joint-soerensen, SOCP: mu 2.2e-11, iteration 53; from there the full
                     * system in double-double, 8-9 s an iteration), the partial normal equations instead, chosen at this mu */
                    { static double pne3 = -2; if (pne3 < -1) { const char *e_ = getenv("BRISK_CONEPARTNE"); pne3 = e_ ? atof(e_) : -1.0; }
                      const double mu_ = S->tau > 0 ? mu : 0;
                      if (pne3 >= 0 && S->nf == 0 && mu_ > 0 && (!S->ne_part || mu_ <= 0.5 * S->ne_part_mu)) {
                          if (S->ne_park) { schol_free(S->ne_park); S->ne_park = NULL; }
                          const double ts_ = sc_time(); const int nk_ = partne_build(S, mu_, pne3); S->t_setup += sc_time() - ts_;
                          if (dbg_on() || S->verbose > 1) socp_printf("      [the direction is not finite on the normal equations: the partial normal equations at mu %.1e, %d cones kept; %s]\n", mu_, nk_, S->ne ? "the iteration done again" : "refused: the full system");
                          if (S->ne) continue;
                      } }
                    if (dbg_on() || S->verbose > 1) socp_printf("      [the direction is not finite on the normal equations after the return: the full system from here, for good]\n");
                    if (S->ne_park) { schol_free(S->ne_park); S->ne_park = NULL; }
                    ne_free(S); S->ne_ok = 0; S->ne_back = 0; S->ne_left = it;
                    continue;
                }
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
                    nf_retry++; S->bump *= 100.0; S->nbump++; bump_raised(S);
                    S->sreg = S->sreg0 * S->bump; S->sregx = S->sreg0 * S->bump; S->sregf = fmax(1e-6, 1e-8 * S->bump);
                    continue;
                }
                status = SOCP_NUMERR; break;
            }
        }
        td[2] += sc_time() - td_last; td_last = sc_time();     /* (the arc, the neighbourhood, the checks before the step) */
        if (nf_restore) { S->sreg = nf_sv[0]; S->sregx = nf_sv[1]; S->sregf = nf_sv[2]; S->bump = nf_sv[3]; nf_restore = 0; }   /* (the raise for that iteration only) */
        if (dbg_on() >= 2) {     /* (12 November 2026, debug: the two cones with the largest x'z over mu, their complementarity's change split:
                                  * the linear part alpha (x'dz + z'dx), the second-order part alpha^2 dx'dz, against the centring target sigma mu) */
            int k1 = -1, k2 = -1; double v1 = -1, v2 = -1;
            for (int k = 0; k < S->nq; k++) { const int d = S->q[k], s0 = S->qs[k]; double v = 0; for (int i = 0; i < d; i++) v += S->x[s0 + i] * S->z[s0 + i];
                if (v > v1) { v2 = v1; k2 = k1; v1 = v; k1 = k; } else if (v > v2) { v2 = v; k2 = k; } }
            for (int t = 0; t < 2; t++) { const int k = t ? k2 : k1; if (k < 0) continue; const int d = S->q[k], s0 = S->qs[k];
                double xz = 0, lin = 0, sec = 0; for (int i = 0; i < d; i++) { xz += S->x[s0 + i] * S->z[s0 + i]; lin += S->x[s0 + i] * dz[s0 + i] + S->z[s0 + i] * dx[s0 + i]; sec += dx[s0 + i] * dz[s0 + i]; }
                socp_printf("      [cone %d (dimension %d): x'z / mu %.3e; the step's linear part %.3e, second-order %.3e (over mu); after it %.3e against the target sigma mu %.3e; dz0 %.2e of z0 %.2e]\n",
                            k, d, xz / mu, alpha * lin / mu, alpha * alpha * sec / mu, (xz + alpha * lin + alpha * alpha * sec) / mu, sigma, dz[s0], S->z[s0]); }
        }
        if (dbg_on() >= 2) {     /* (12 November 2026, debug: the final direction's primal equation A dx - b dtau = (1 - sigma) rP, its miss
                                  * against the parts' sizes; multiscale rows A02_00, SOCP: the first step from a primal feasible start left 1.1e-3) */
            double *q_ = (double *)malloc(sizeof(double) * ((size_t)m + 1));
            if (q_) { for (int i = 0; i < m; i++) q_[i] = 0.0; A_mul(S, 1.0, dx, q_);
                double nax = 0, nbt = 0, nmiss = 0, nr = 0;
                for (int i = 0; i < m; i++) { nax += q_[i] * q_[i]; nbt += (S->b[i] * dtau) * (S->b[i] * dtau); const double t_ = q_[i] - S->b[i] * dtau - (1.0 - sigma) * rP[i]; nmiss += t_ * t_; nr += rP[i] * rP[i]; }
                socp_printf("      [the direction's primal equation: |A dx - b dtau - (1 - sigma) rP| %.3e; |A dx| %.3e, |b dtau| %.3e, |rP| %.3e, dtau %.3e, tau %.3e]\n", sqrt(nmiss), sqrt(nax), sqrt(nbt), sqrt(nr), dtau, S->tau);
                free(q_); } }
        /* (12 November 2026, the SOCP developer, a test: BRISK_CONEPFIX=1) the direction's primal equation corrected by one more solve:
         * dx is a difference of large terms where H^-1 is large (the normal equations' A H^-1 rx), so A dx - b dtau misses (1 - sigma) rP
         * by the rounding of those terms, whatever the refinement does with the whole right-hand side (multiscale rows A02_00, SOCP: the
         * first step from a primal feasible start missed by 2.6e-8 against |A dx| 1.4e-7, 1.1e-3 on the rows as given). The miss q
         * alone is solved for, [-H A'; A 0] [ddx; ddy] = [0; q]: the solve's relative accuracy then applies to q's size; ddz = -H ddx
         * keeps the dual equation and the linearized complementarity; dtau unchanged. Only when the miss is above a tenth of the
         * primal equation's target */
        { static int pfx = -1; if (pfx < 0) { const char *e_ = getenv("BRISK_CONEPFIX"); pfx = e_ ? atoi(e_) : 0; }
          if (pfx && !S->use_dd) {
            double *q_ = (double *)calloc((size_t)m + 1, sizeof(double)), *zx_ = (double *)calloc((size_t)n + 1, sizeof(double));
            double *ddx = (double *)calloc((size_t)n + 1, sizeof(double)), *ddy = (double *)calloc((size_t)m + 1, sizeof(double)), *hd = (double *)calloc((size_t)n + 1, sizeof(double));
            if (q_ && zx_ && ddx && ddy && hd) {
                A_mul(S, -1.0, dx, q_);
                double nq = 0, nt = 0;
                for (int i = 0; i < m; i++) { const double tg = (1.0 - sigma) * rP[i]; q_[i] += S->b[i] * dtau + tg; nq += q_[i] * q_[i]; nt += tg * tg; }
                nq = sqrt(nq); nt = sqrt(nt);
                if (nq > 0.1 * nt && nq > 0) {
                    kkt_solve(S, zx_, q_, ddx, ddy, S->refine_max);
                    /* (tau's share as in corr_solve, so that the gap row keeps its equation: ddtau from c'ddx - b'ddy, the tau
                     * direction p added with it, kappa moved by -kappa ddtau / tau) */
                    const double ddt = (dotv(n, S->c, ddx) - dotv(m, S->b, ddy)) / (denom0 + S->kap / S->tau);
                    if (isfinite(ddt)) {
                        for (int j = 0; j < n; j++) ddx[j] += ddt * px[j];
                        for (int i = 0; i < m; i++) ddy[i] += ddt * py[i];
                        dtau += ddt; dkap -= S->kap * ddt / S->tau;
                    }
                    H_apply(S, ddx, hd);
                    for (int j = 0; j < n; j++) { dx[j] += ddx[j]; if (j >= S->nf) dz[j] -= hd[j]; }
                    for (int i = 0; i < m; i++) dy[i] += ddy[i];
                    if (dbg_on()) socp_printf("      [the direction's primal equation corrected: its miss %.3e against the target %.3e]\n", nq, nt);
                }
            }
            free(q_); free(zx_); free(ddx); free(ddy); free(hd);
          } }
        #pragma omp parallel for schedule(static) num_threads(g_cone_nt) if (cone_vpar(S))
        for (int j = 0; j < n; j++) { S->x[j] += alpha * dx[j]; S->z[j] += alpha * dz[j]; }
        if (S->fpr) { if (!S->fp_dxp) S->fp_dxp = (double *)calloc((size_t)S->nf + 1, sizeof(double)); for (int j = 0; j < S->nf; j++) S->fp_dxp[j] = alpha * dx[j]; }
        for (int i = 0; i < m; i++) S->y[i] += alpha * dy[i];
        S->tau += alpha * dtau; S->kap += alpha * dkap;
        /* the step length comes from a quadratic per cone: in a large cone its rounding can leave the
         * new point outside; the step is then shortened */
        int scu_in = 0;
        for (int bt = 0; bt < 20; bt++) {
            if (scu_on() ? scu_update(S, alpha, dx, dz) == 0 : cone_interior(S)) { scu_in = scu_on(); break; }
            const double back = 0.2 * alpha;
            if (dbg_on()) {     /* (12 November 2026, debug: which cone the step left, its margin x0 - |x1| against x0, its size) */
                int kb = -1; double xm = 0, zm = 0, x0 = 0, z0 = 0;
                for (int k = 0; k < S->nq && kb < 0; k++) { const int d = S->q[k], s0 = S->qs[k]; double xn = 0, zn = 0;
                    for (int i = 1; i < d; i++) { xn += S->x[s0 + i] * S->x[s0 + i]; zn += S->z[s0 + i] * S->z[s0 + i]; }
                    xn = sqrt(xn); zn = sqrt(zn);
                    if (!((S->x[s0] - xn) * (S->x[s0] + xn) > 0) || !((S->z[s0] - zn) * (S->z[s0] + zn) > 0)) { kb = k; xm = S->x[s0] - xn; zm = S->z[s0] - zn; x0 = S->x[s0]; z0 = S->z[s0]; } }
                socp_printf("      [the new point outside the cones by rounding: the step %.3f taken back by a fifth; cone %d (dimension %d): x0 - |x1| %.2e of x0 %.2e, z0 - |z1| %.2e of z0 %.2e]\n", alpha, kb, kb >= 0 ? S->q[kb] : 0, xm, x0, zm, z0);
            }
            #pragma omp parallel for schedule(static) num_threads(g_cone_nt) if (cone_vpar(S))
            for (int j = 0; j < n; j++) { S->x[j] -= back * dx[j]; S->z[j] -= back * dz[j]; }
            for (int i = 0; i < m; i++) S->y[i] -= back * dy[i];
            S->tau -= back * dtau; S->kap -= back * dkap;
            alpha -= back;
        }
        if (scu_in) S->scu_ok = 1;      /* (the next nt_update takes the scaling scu_update formed at this step) */
        td[3] += sc_time() - td_last; td_last = sc_time();     /* (the step taken, its membership test) */
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
          if ((necost || S->ne_tinykept || S->ne_depused) && S->ne) {      /* (by cost also once the normal equations are kept past tiny pivots: ne_factor) */
              const double ex = (double)(S->refine_total - ref0) - (double)(S->solves - sol0);
              const double cstep = 2.0 * (double)S->ne_lnz * 1.5e-9 + 2.0 * (double)S->Ap[n] * 3e-9 + (double)(n + m) * 2e-8;
              /* (3 November 2026, the SOCP developer) once a correction on a row dependent to working precision is in use
               * (ne_factor), the full system has the same cancellation: its factor in double replaces pivots and the solve goes
               * to double-double, at the column code's rate (about 3e8 flops a second), so that is the cost of leaving.
               * db-plate-yield-line (SOCP): left at iteration 31 by the count rule (15 steps for 4 solves, 0.2 s of extra steps),
               * then 8 iterations of 1-2 s in double-double; kept, the solve ends on the normal equations */
              const double tfull = S->ne_depused ? S->cflops / 3e8 : S->cflops / 6e9 * (S->kkt_defer ? 2.0 : 1.0);
              leave_ne = ex * cstep > tfull;
          } else leave_ne = S->ne && S->refine_total - ref0 > ne_k * (double)(S->solves - sol0);
          /* (7 November 2026, the SOCP developer; the owner's SOCP benchmark of 6 November; the default with K = 2, BRISK_CONENESTALL=K, 0: off) the normal equations
           * left also when their directions stop making progress near the end: the largest of the four errors not halved in K
           * iterations while its best is at most 1e-6 and above the tolerance. A heterogeneous-cones SOCP (A05_10) on the normal
           * equations: the complementarity error stalled at 4e-8 from iteration 9 with full steps and 3-4 refinement steps a solve
           * (the count rule silent), tau growing to 1e33, 24 iterations against the full system's 8 */
          { static int nst = -1; if (nst < 0) { const char *e_ = getenv("BRISK_CONENESTALL"); nst = e_ ? atoi(e_) : 2; }
            static _Thread_local double nsb; static _Thread_local int nsi;
            if (it == 0 || !S->ne) { nsb = 1e300; nsi = it; }
            static double nsf = -1; if (nsf < 0) { const char *e_ = getenv("BRISK_CONENESTALLF"); nsf = e_ ? atof(e_) : 0.5; }   /* (11 November 2026, a test: the progress asked of K iterations, a factor; 0.5 as before) */
            if (S->ne && S->tau > 0 && err < nsf * nsb) { nsb = err; nsi = it; }
            /* (the full system affordable: finishing on it, about four factorizations of its work, against the second attempt it
             * replaces, about it + 1 of the normal equations'; 2013_firL2Linfalph, SOCP: 2.7e9 flops against 2.4e11, the full system
             * beyond the memory: not left) */
            const int afford_ = S->ne && S->cflops > 0 && 4.0 * S->cflops <= (double)(it + 1) * fmax(schol_flops(S->ne), 1.0);
            /* (11 November 2026, the SOCP developer, a test: BRISK_CONENEAFF=a) near the end (the best error at most 1e-6) a predictor step
             * below a on the normal equations: their directions no longer lead to the solution (the end of a convergent solve takes predictor
             * steps near 1); the full system from the next iteration, as after a blocked step. Robust families (SOCP) with the predictor
             * refined: P06_01 affine steps 0.60, 0.39 in the large cone of dimension 4,813, 20 iterations; the full system 16 */
            { static double naf = -1; if (naf < 0) { const char *e_ = getenv("BRISK_CONENEAFF"); naf = e_ ? atof(e_) : 0.0; }
              if (naf > 0 && S->ne && !leave_ne && afford_ && nsb <= 1e-6 && nsb > opt->tol && aaff < naf) {
                  if (dbg_on()) socp_printf("      [a short predictor step (%.3f) near the end on the normal equations: the full system from here]\n", aaff);
                  leave_ne = 1;
              } }
            if (nst > 0 && S->ne && !leave_ne && afford_ && it - nsi >= nst && nsb <= 1e-6 && nsb > opt->tol) {
                if (dbg_on()) socp_printf("      [the normal equations' directions stalled: the error not reduced enough in %d iterations (best %.1e at iteration %d)]\n", it - nsi, nsb, nsi);
                leave_ne = 1;
            } }
          /* (13 November 2026, the SOCP developer, a test: BRISK_CONEPARTNE=t) where the normal equations would be left for their accuracy,
           * once: the cones whose own scaling spread (w0 + |w1|)^4 exceeds (1/mu)^t keep their unknowns (the partial normal equations,
           * ne_setup_fix), and the solve stays on that system. db-joint-soerensen (SOCP), iteration 47, a study on its scalings: GMRES
           * preconditioned by the normal equations made no progress in 20 steps, by the partial form with the 8,565 cones of the largest
           * spread kept 4e-13 in 26 steps */
          { static double pne = -2; if (pne < -1) { const char *e_ = getenv("BRISK_CONEPARTNE"); pne = e_ ? atof(e_) : -1.0; }
            /* (BRISK_CONEPARTNEMU=x, a test: the partial normal equations set up as soon as mu falls below x, before the normal
             * equations lose their accuracy: db-joint-soerensen (SOCP) lost iteration 38 to them, a step of 0.008, 9 s) */
            static double pmu = -1; if (pmu < 0) { const char *e_ = getenv("BRISK_CONEPARTNEMU"); pmu = e_ ? atof(e_) : 0.0; }
            const int pearly = pmu > 0 && !S->ne_part && mu < pmu;
            /* (BRISK_CONEPARTNERS=r, default 0.1: on the partial system the cones are chosen again as soon as mu falls by r since the last
             * choice, before a solve fails: the wrong-sign pivots in the kept nodes grew 49 -> 509 from mu 2.2e-9 to 2.2e-10, and the
             * factor broke at mu 6.8e-11, 30 times below the choice; 0: only after a failure or a stall) */
            static double prs = -1; if (prs < 0) { const char *e_ = getenv("BRISK_CONEPARTNERS"); prs = e_ ? atof(e_) : 0.1; }
            const int pre_ = S->ne_part && prs > 0 && mu <= prs * S->ne_part_mu;
            /* (BRISK_CONEPARTNELATE=1: the partial normal equations first set up only where the normal equations kept past tiny pivots
             * give out, a direction not finite, below) */
            static int plate = -1; if (plate < 0) { const char *e_ = getenv("BRISK_CONEPARTNELATE"); plate = e_ ? atoi(e_) : 0; }
            if (pne >= 0 && S->ne && (!S->ne_back || S->ne_part) && ((!S->ne_part && !plate && (leave_ne || pearly)) || pre_ || (S->ne_part && leave_ne && mu <= 0.1 * S->ne_part_mu)) && S->nf == 0 && mu > 0 && mu < 1) {
                const double ts_ = sc_time(); const int nk_ = partne_build(S, mu, pne); S->t_setup += sc_time() - ts_;
                if (dbg_on() || S->verbose > 1) socp_printf("      [the partial normal equations at iteration %d: %d cones kept (spread above (1/mu)^%.2f, mu %.1e); %s; %.3fs]\n", it + 1, nk_, pne, mu, S->ne ? "the solve stays on them" : "refused: the full system", sc_time() - ts_);
                if (S->ne) leave_ne = 0;
            } }
          if (S->ne && !S->ne_back && (leave_ne || (dropit > 0 && it + 1 == dropit))) {
              if (dbg_on()) socp_printf("      [the normal equations left at iteration %d: %ld refinement steps for %ld solves]\n", it + 1, S->refine_total - ref0, S->solves - sol0);
              /* (4 November 2026, a test: BRISK_CONENEBACK=1) with no free column left the normal equations' factor is parked, not
               * freed: kkt_factor comes back to it when the full system's double factor breaks the quasi-definite bound */
              static int nb_c = -2; const int nb_ = cone_back("BRISK_CONENEBACK", &nb_c);
              if (nb_ && S->nf == 0 && !S->ne_park) { S->ne_park = S->ne; S->ne = NULL; S->ne_ok = 0; S->ne_left = it + 1; }
              else { ne_free(S); S->ne_ok = 0; S->ne_left = it + 1; } } }
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
        if (S->rot_switch == it && S->rot_mode == 1 && !S->use_dd && !S->rot_want && c_this > c_sw0) {     /* (not when the switch answered a factor below the bound: there it is for accuracy) */
            const double ts = sc_time();
            kkt_teardown(S); S->rot_mode = 0; S->kkt_defer = 0; S->rot_back = it + 1;
            const int rc_ = kkt_setup(S);
            S->t_setup += sc_time() - ts;
            if (S->verbose > 1) socp_printf("      [the small cones back in their original basis (the iteration %.3g against %.3g before the eigenbasis, in the cost model; %.3fs)]\n", c_this, c_sw0, sc_time() - ts);
            if (rc_) { status = SOCP_NUMERR; break; }
        }
        if (S->rot_want && getenv("BRISK_ROTDBG")) socp_printf("      [the eigenbasis asked for: rot_mode %d, rot_switch %d, anyrot %d, ne %d, use_dd %d]\n", S->rot_mode, S->rot_switch, S->anyrot, S->ne != NULL, S->use_dd);
        if (((S->rot_mode == 2 && !S->rot_switch) || (S->rot_want && S->rot_mode == 0 && S->rot_back)) && !S->anyrot && (!S->ne || S->rot_want) && !S->use_dd
            && (S->rot_want || (S->solves - sol0 > 0 && S->refine_total - ref0 >= 2 * (S->solves - sol0) && S->it_rs2 > 0.5 * S->it_rt2))) {
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
        tab_step = alpha;
        rr_pred = 1.0 - alpha * (1.0 - sigma);
        row_step = alpha;
        if (S->verbose > 1) socp_printf("%4d %+.9e %+.9e %8.1e %8.1e %8.1e %8.1e %8.1e %5.3f %6.0e %3ld%s\n", it + 1, pobj, dobj, e_p, e_d, e_g, mu, S->tau, alpha, sigma, S->refine_total - ref0, S->F.ndyn ? (S->F.dd ? " dd*" : " *") : S->F.dd ? " dd" : "");
        if (dbg_on()) { static double tprev = -1; const double tn = sc_time(); socp_printf("      [iteration %d: %.3fs (factorizations %.3fs, solves %.3fs, products %.3fs so far; %ld solves, %ld with the factor)]\n", it + 1, tprev < 0 ? 0.0 : tn - tprev, S->t_factor, S->t_solve, S->t_mul, S->solves, S->nback); tprev = tn; }
        if (dbg_on()) socp_printf("      [applications of the factor so far: tau direction %ld, predictor %ld, corrector %ld, extra correctors %ld, refinements %ld (%ld solves)]\n", g_tag_apps[1], g_tag_apps[2], g_tag_apps[3], g_tag_apps[4], g_tag_apps[5], g_tag_solves[5]);
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
        /* (4 November 2026, the SOCP developer, with BRISK_CONEQDCHECK; BRISK_CONEQDEND=0 turns it off) the same end phase once
         * the double factor had to be refactored with a raised regularization (it broke the quasi-definite bound): the raised
         * regularization makes GMRES's solves "succeed" to its target, so the rule above (a solve failed outright) no longer
         * fires, and the iterate stalls just above the tolerance. joint_HO_05 (SOCP): 3.1e-8 for 20 iterations, REDUCED in 221 s;
         * by default (no bound check) the rule above takes it to double-double, OPTIMAL in 291 s */
        { static int qde = -1; if (qde < 0) { const char *e_ = getenv("BRISK_CONEQDEND"); qde = e_ ? atoi(e_) : 1; }
          if (qde && S->qd_ever && !S->use_dd && S->dd_on && !S->want_dd && !S->dd_polish && best <= 10.0 * opt->tol && S->cflops <= 5e9 && !getenv("BRISK_CONENODDEND")) {
              S->dd_polish = 1; S->want_dd = 1;
              if (S->verbose > 1) socp_printf("      [the regularization was raised for the bound of the double factor and the error is within 10x the tolerance (%.1e): double-double from here]\n", best);
          } }
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
        /* (31 October 2026, the SOCP developer, a test: BRISK_CONEDDSTALL=k, after k iterations; off; ISSUES 810) the primal and dual
         * errors (the larger of the two; the gap is not a measure of progress in the first iterations, where it stays near 1)
         * have not fallen by a tenth in ten iterations, above the tolerance, while the iterate is not on its way to an infeasibility certificate (kappa <=
         * tau): the double factorization, refined by GMRES, no longer gives the directions the step needs, although no solve
         * fails outright. A user's truss compliance SOCP: the iterate's entries spread over many orders, GMRES stagnates at a
         * relative residual of 1e-7 .. 1e-10 against the double factor whatever its target, the linearized equations hold in
         * norm but the steps stay at 0.05 .. 0.6, mu falls faster than the residuals and tau decays: 500 iterations at an error
         * of 1e-2. In double-double the solves reach 1e-16, the steps 0.7 .. 0.99, tau stays at 1.4e-2: OPTIMAL in 44
         * iterations. Double-double from here, under the work limit of the other double-double rules; once per solve */
        { static int ddstall = -1; if (ddstall < 0) { const char *e_ = getenv("BRISK_CONEDDSTALL"); ddstall = e_ ? atoi(e_) : 0; }
          if (ddstall > 0 && it - stall_it >= ddstall && S->kap <= S->tau && !S->use_dd && S->dd_on && !S->want_dd && !S->dd_polish && S->cflops <= 5e9) {
              S->dd_polish = 1; S->want_dd = 1;
              if (S->verbose > 1) socp_printf("      [the primal and dual errors have not fallen by a tenth in %d iterations (best %.1e): double-double from here]\n", it - stall_it, stall_best);
          } }
        /* (7 November 2026, the SOCP developer; with BRISK_CONENESTALL) a blocked step (below 1e-3) on the normal equations leaves them
         * for the full system, and the solve goes on from the iterate, instead of a second attempt from the start: their directions
         * are the inaccurate ones (the large cones' rank-two terms and Theta (A'dy - r) for the linear steps), the full system's are
         * not. A near-tangent SOCP (A03_01): steps 0.001 and 0.000 at iterations 7-8 on the normal equations, then a second
         * attempt of 14 iterations; robust-inventory SOCPs (P03_01): step 0 at iteration 15, a second attempt of 23 */
        { static int nss = -1; if (nss < 0) { const char *e_ = getenv("BRISK_CONENESTALL"); nss = !e_ || atoi(e_) > 0; }
          /* (near the end only: the best error at most 1e-6, as the stall rule; early in a solve one short step is common and the rule
           * below tolerates it; the full system then cost more than the whole solve: 2013_firL2Linfalph, SOCP, a blocked step at
           * iteration 1 left normal equations of 2,002 rows (2.7e9 flops) for a full system of 3.7e11) */
          /* (8 November 2026, the SOCP developer; the default since that day, BRISK_CONENEEND=0 turns it off) also at any error when the step is the second blocked one
           * in a row, which ends the attempt (below): the choice is then the full system from the iterate or a second attempt from the
           * start, which costs a whole solve. Robust dose (P11_09, SOCP): steps 0.001 twice at an error of 2.1e-6, NUMERICAL
           * DIFFICULTIES and a second attempt of 25 iterations, where the same exit one iteration later ends OPTIMAL at the next */
          static int nee = -1; if (nee < 0) { const char *e_ = getenv("BRISK_CONENEEND"); nee = e_ ? atoi(e_) : 1; }
          static double nbe = -1; if (nbe < 0) { const char *e_ = getenv("BRISK_CONENEBLKERR"); nbe = e_ ? atof(e_) : 1.0; }
          /* (8 November 2026, the default since that day; BRISK_CONENEBLKERR=1e-6 is the earlier rule) the first blocked step already moves
           * the solve to the full system at any relative error up to 1, not only near the end: the guard against an early exit is the
           * affordability bound (the full system's factorization at most a quarter of the normal equations' work so far); the 2013 case
           * above stays on the normal equations by that bound. Distributionally robust (SOCP) 0.90 of the time, robust dose (SOCP) 0.97,
           * no iteration count higher on the families, their fresh and transformed copies or the development mezo set */
          if (nss && alpha < 1e-3 && (best <= nbe || (nee && nsmall >= 1)) && S->ne && !S->ne_back && S->tau > 0 && S->tau == S->tau && S->cflops > 0 && 4.0 * S->cflops <= (double)(it + 1) * fmax(schol_flops(S->ne), 1.0)) {
              if (dbg_on()) socp_printf("      [a blocked step (%.1e) on the normal equations: the full system from here]\n", alpha);
              static int nb2_c = -2; const int nb2_ = cone_back("BRISK_CONENEBACK", &nb2_c);
              if (nb2_ && S->nf == 0 && !S->ne_park) { S->ne_park = S->ne; S->ne = NULL; S->ne_ok = 0; S->ne_left = it + 1; }
              else { ne_free(S); S->ne_ok = 0; S->ne_left = it + 1; }
              nsmall = 0; continue;
          } }
        /* (14 November 2026, the SOCP developer; BRISK_CONENELEFTGO=0 turns it off) the normal equations left in this very iteration
         * by the rules above their step (their refinement's cost, their stall): the blocked step was theirs, and the full system takes
         * the next iteration from the iterate, as the blocked-step rule just above does when the normal equations are still there.
         * Without it the two exits disagreed: the first freed the normal equations, so the second did not see them, and the step of
         * 0 ended the attempt. joint_FC_7 (SOCP) with the free variables in one cone: the normal equations left at iteration 35 (70
         * refinement steps for 9 solves, a step of 5e-12), then a second attempt from the start: 101 iterations, 64.7 s */
        { static int nlg = -1; if (nlg < 0) { const char *e_ = getenv("BRISK_CONENELEFTGO"); nlg = e_ ? atoi(e_) : 1; }
          if (nlg && !S->ne && S->ne_left == it + 1 && alpha < 1e-3 && nsmall == 0 && S->tau > 0 && S->tau == S->tau) {
              if (dbg_on()) socp_printf("      [a blocked step (%.1e) in the iteration that left the normal equations: the full system from here]\n", alpha);
              prev_alpha = alpha; continue;
          } }
        prev_alpha = alpha;
        nsmall = alpha < 1e-3 ? nsmall + 1 : 0;
        if (!(alpha > 1e-10) || nsmall >= 2 || !(S->tau > 0) || !(S->tau == S->tau)) { status = SOCP_NUMERR; break; }
    }
    /* (4 November 2026, with BRISK_CONENEBACK) a failure after the solve came back to the normal equations (the full system broke
     * the quasi-definite bound, double-double refused) is the limit of double precision, not of the step length: the second
     * attempt with shorter steps would repeat it (db-joint-soerensen, SOCP: 156 s more for 7.9e-7 -> 7.3e-7) */
    if (status == SOCP_NUMERR && S->ne_back) R->stalled = 1;
    if (tq_last >= 0) tq[4] += sc_time() - tq_last;
    brisk_itab_end();
    if (dbg_on()) socp_printf("  [directions and steps: the linear solves %.3f s (the factor's applications %.3f, the products %.3f, the rest of the solves %.3f), the steps to the boundary %.3f s]\n", g_t_ks, S->t_prec, S->t_mul, g_t_ks - S->t_prec - S->t_mul, g_t_cstep);
    if (dbg_on() && scu_on()) socp_printf("  [the scaling updated in the scaled coordinates: %ld cone scalings formed from the point again (W x and W^-1 z apart)]\n", S->scu_nredo);
    if (dbg_on()) socp_printf("  [directions and steps by part: the predictor, tau and the corrector %.3f, the extra correctors %.3f, the checks before the step %.3f, the step and its membership test %.3f s]\n", td[0], td[1], td[2], td[3]);
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
    if (getenv("BRISK_CONEXZDBG") && R->x && R->y && R->z) {
        /* (11 November 2026, the SOCP developer; ISSUES 865, debug) the returned point on the problem as given: c'x - b'y = x'z + x'r_d - y'r_p
         * (r_p = b - A x, r_d = c - A'y - z), the sizes, and which blocks carry x'z: their share and whether x and z are both away from the
         * boundary there (a pair not complementary) or the product is spread */
        const int m_ = P->m, n_ = P->n, nf_ = P->nf, nl_ = P->nl, nk_ = P->nq + P->nr;
        double *rp_ = (double *)malloc(sizeof(double) * ((size_t)m_ + 1)); for (int i = 0; i < m_; i++) rp_[i] = P->b[i];
        double cx = 0, by = 0, xz = 0, xrd = 0, yrp = 0, nx = 0, ny = 0, nrp = 0, nrd = 0;
        for (int j = 0; j < n_; j++) { double aty = 0; for (int p = P->Ap[j]; p < P->Ap[j + 1]; p++) { rp_[P->Ai[p]] -= P->Ax[p] * R->x[j]; aty += P->Ax[p] * R->y[P->Ai[p]]; }
            const double rd = P->c[j] - aty - (j < nf_ ? 0.0 : R->z[j]); cx += P->c[j] * R->x[j]; if (j >= nf_) xz += R->x[j] * R->z[j]; xrd += R->x[j] * rd; nx = fmax(nx, fabs(R->x[j])); nrd = fmax(nrd, fabs(rd)); }
        for (int i = 0; i < m_; i++) { by += P->b[i] * R->y[i]; yrp += R->y[i] * rp_[i]; ny = fmax(ny, fabs(R->y[i])); nrp = fmax(nrp, fabs(rp_[i])); }
        free(rp_);
        socp_printf("  [x'z debug: c'x %.10e  b'y %.10e  c'x-b'y %.3e = x'z %.3e + x'r_d %.3e - y'r_p %.3e; |x| %.2e |y| %.2e |r_p| %.2e |r_d| %.2e]\n", cx, by, cx - by, xz, xrd, yrp, nx, ny, nrp, nrd);
        /* the blocks' products */
        const int nb_ = nl_ + nk_; double *pr = (double *)malloc(sizeof(double) * ((size_t)nb_ + 1)); int *ix = (int *)malloc(sizeof(int) * ((size_t)nb_ + 1));
        double *lx = (double *)malloc(sizeof(double) * ((size_t)nb_ + 1)), *lz = (double *)malloc(sizeof(double) * ((size_t)nb_ + 1));
        for (int b = 0; b < nl_; b++) { const int j = nf_ + b; pr[b] = R->x[j] * R->z[j]; lx[b] = R->x[j]; lz[b] = R->z[j]; ix[b] = b; }
        { int j0 = nf_ + nl_; for (int k = 0; k < nk_; k++) { const int d = k < P->nq ? P->q[k] : P->r[k - P->nq]; double p_ = 0, tx = 0, tz = 0; for (int t = 0; t < d; t++) p_ += R->x[j0 + t] * R->z[j0 + t];
              double x0, z0; if (k < P->nq) { x0 = R->x[j0]; z0 = R->z[j0]; for (int t = 1; t < d; t++) { tx += R->x[j0 + t] * R->x[j0 + t]; tz += R->z[j0 + t] * R->z[j0 + t]; } }
              else { x0 = (R->x[j0] + R->x[j0 + 1]) * sqrt(0.5); z0 = (R->z[j0] + R->z[j0 + 1]) * sqrt(0.5); tx = 0.5 * (R->x[j0] - R->x[j0 + 1]) * (R->x[j0] - R->x[j0 + 1]); tz = 0.5 * (R->z[j0] - R->z[j0 + 1]) * (R->z[j0] - R->z[j0 + 1]); for (int t = 2; t < d; t++) { tx += R->x[j0 + t] * R->x[j0 + t]; tz += R->z[j0 + t] * R->z[j0 + t]; } }
              pr[nl_ + k] = p_; lx[nl_ + k] = x0 - sqrt(tx); lz[nl_ + k] = z0 - sqrt(tz); ix[nl_ + k] = nl_ + k; j0 += d; } }
        /* the largest products first (a partial selection by repeated passes: few blocks are printed) */
        double tot = 0; for (int b = 0; b < nb_; b++) tot += fabs(pr[b]);
        double acc = 0; int n50 = -1, n90 = -1;
        double *srt = (double *)malloc(sizeof(double) * ((size_t)nb_ + 1)); for (int b = 0; b < nb_; b++) srt[b] = fabs(pr[b]);
        qsort(srt, (size_t)nb_, sizeof(double), xzdbg_cmpd);
        for (int b = 0; b < nb_; b++) { acc += srt[b]; if (n50 < 0 && acc >= 0.5 * tot) n50 = b + 1; if (n90 < 0 && acc >= 0.9 * tot) { n90 = b + 1; break; } }
        socp_printf("  [x'z debug: %d blocks (%d nonnegative, %d cones); half of sum |x_b'z_b| in the largest %d, 90 %% in %d; the largest %.3e, the median %.3e]\n", nb_, nl_, nk_, n50, n90, nb_ ? srt[0] : 0.0, nb_ ? srt[nb_ / 2] : 0.0);
        for (int r = 0; r < 6 && r < nb_; r++) { int bb = -1; double bv = -1; for (int b = 0; b < nb_; b++) if (fabs(pr[b]) > bv) { bv = fabs(pr[b]); bb = b; }
            socp_printf("  [x'z debug: block %d (%s): x'z %.3e, x's smallest eigenvalue %.3e, z's %.3e]\n", bb, bb < nl_ ? "nonnegative" : "cone", pr[bb], lx[bb], lz[bb]); pr[bb] = 0; }
        free(pr); free(ix); free(lx); free(lz); free(srt);
    }
    g_fr_tau = S->tau;
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
        if (dbg_on()) socp_printf("  [centrality correctors: %.0f accepted, %.0f rejected; the correction term %.3fs, the solves %.3fs, the step %.3fs, the copies %.3fs]\n", g_ctime[5], g_ctime[4], g_ctime[0], g_ctime[1], g_ctime[2], g_ctime[3]);
        if (dbg_on()) socp_printf("  [factorization / solve: estimated %.1f, measured %.1f]\n", S->rr_est, (S->t_factor / fmax(S->nfact, 1)) / fmax((S->t_solve + S->t_mul) / fmax(S->nback, 1), 1e-12));
        if (dbg_on()) { static const char *nm_[7] = { "other", "tau direction", "predictor", "corrector", "extra correctors", "refinements", "q against its rows" };
            socp_printf("  [solves by purpose (solves / applications of the factor):");
            for (int t_ = 0; t_ < 7; t_++) if (g_tag_solves[t_]) socp_printf(" %s %ld / %ld;", nm_[t_], g_tag_solves[t_], g_tag_apps[t_]);
            socp_printf("]\n"); socp_printf("  [the normal equations' partitioned solves: %ld tried, %ld done]\n", g_part_try, g_part_ok); }
        if (S->verbose > 1 && S->ncor_total) socp_printf("  %ld centrality correctors\n", S->ncor_total);
        if (S->verbose > 1 && narc) socp_printf("  %ld steps on the arc\n", narc);
        if (S->verbose > 1 && S->ne_nfact) socp_printf("  %ld factorizations of the normal equations (nnz(L) = %ld, %d tiny pivots%s)\n", S->ne_nfact, S->ne_lnz, S->ne_tiny, S->ne_dense_msg);
        if (S->verbose > 1 && S->nbump) socp_printf("  regularization of the factored matrix raised to %.0e\n", 1e-8 * S->bump);
        if (S->nbump) SNOTED("the regularization of the factored matrix was raised to %.0e", 1e-8 * S->bump);
        if (S->rot_back) SNOTED("the small cones factored in the eigenbasis of their scaling in iteration %d only (its larger factor cost more than the steps it saved)", S->rot_switch);
        else if (S->rot_switch) SNOTED("the small cones factored in the eigenbasis of their scaling from iteration %d", S->rot_switch);
        if (S->verbose > 1 && S->ndd) socp_printf("  %d factorizations in double-double\n", S->ndd);
        if (S->ndd) SNOTED("%d of %d factorizations in double-double arithmetic", S->ndd, S->nfact);
        if (S->verbose > 1 && S->dz_count) socp_printf("  %ld refinements of the direction against the dual equation\n", S->dz_count);
        if (S->verbose > 1 && S->rc_used) socp_printf("  %ld solves started from the previous Krylov space (mean gain %.1f digits)\n", S->rc_used, S->rc_gain / (double)S->rc_used / log(10.0));
        if (!att_line && !by_elim && status != SOCP_PINF && status != SOCP_DINF) socp_printf("  primal obj c'x = %+.12e   dual obj b'y = %+.12e\n  errors: pinf %.1e  dinf %.1e  gap %.1e  x'z %.1e\n", R->pobj, R->dobj, R->err[0], R->err[1], R->err[2], R->err[3]);
    }
    free(px); free(py); free(qx); free(qy); free(dx); free(dy); free(dz); free(rP); free(rD); free(pol_x); free(pol_y); free(pol_z); free(pol_xs); free(pol_ys); free(pol_zs); free(pol_ax); free(pol_az); free(pol_bx); free(pol_bz); free(r3); free(rhsx); free(rhsy); free(ds); free(t1); free(t2);
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
    unsigned **rhk; int **rhp; unsigned *rhm; int *rhc;     /* the index (row, column) -> position in the row: a table a row */
    unsigned char *rh;                                      /* the row is in the index (else: as read, sorted) */
    double *esr, *esc;                                      /* the row and column factors of the pivot tests */
} Elim;
/* 18 October 2026 (LP/SOCP): a row that a substitution updates enters an open-addressing index (linear probing,
 * deletion by backward shift) that gives the position of an entry, and is updated in place (unsorted from then on),
 * so that a substitution costs O(len(pivot row)) per row it updates, after the row's first update. Rows never
 * updated stay as read (sorted; binary search). Before, each update merged the pivot row into the whole sorted row: with long rows holding many free
 * variables (one cone of dimension 50,001 tied to its free copy, with 50 dense rows) every substitution rewrote
 * 50 rows of length n, O(n^2) in all (426 s). The entries, their values and the reduced problem are the same. */
/* (3 November 2026, the SOCP developer) one small table a row (open addressing, linear probing, deletion by backward shift;
 * a key is the column + 1, 0 empty), allocated when the row enters the index and freed with the row, in place of one table
 * for the whole matrix: the global table had four slots an entry (CBLIB's beam30, SOCP: 28 million entries, 128 million
 * slots of 12 bytes) and every access missed the caches; a row's table holds only its own entries. The positions it gives
 * are the same, so the substitutions and the reduced problem are the same bits (beam30: the index 4.4 -> ... s) */
static inline unsigned eh_slot(unsigned key, unsigned mask) { return (unsigned)(((unsigned long long)key * 0x9E3779B97F4A7C15ULL) >> 40) & mask; }
static int row_find(const Elim *E, int i, int col) {
    if (!E->rh[i]) {
        int lo = 0, hi = E->rlen[i] - 1; const int *rc = E->rcol[i];
        while (lo <= hi) { const int mid = (lo + hi) >> 1; if (rc[mid] == col) return mid; if (rc[mid] < col) lo = mid + 1; else hi = mid - 1; }
        return -1;
    }
    const unsigned key = (unsigned)col + 1u, mask = E->rhm[i]; const unsigned *hk = E->rhk[i];
    for (unsigned h = eh_slot(key, mask); ; h = (h + 1) & mask) { if (hk[h] == key) return E->rhp[i][h]; if (hk[h] == 0) return -1; }
}
static _Thread_local double g_eh_tgrow, g_eh_thash;      /* (debug timers of the substitution's index) */
static void eh_alloc(Elim *E, int i, int want) {       /* row i's table for at least want entries, its entries moved over */
    const double tg_ = sc_time();
    unsigned c = 16; while (c < 2u * (unsigned)want + 2u) c <<= 1;
    unsigned *ok = E->rhk[i]; int *op = E->rhp[i]; const unsigned om = ok ? E->rhm[i] + 1 : 0;
    E->rhk[i] = (unsigned *)calloc(c, sizeof(unsigned)); E->rhp[i] = (int *)malloc(sizeof(int) * c); E->rhm[i] = c - 1; E->rhc[i] = 0;
    for (unsigned h = 0; h < om; h++) if (ok[h]) {
        unsigned g = eh_slot(ok[h], c - 1); while (E->rhk[i][g]) g = (g + 1) & (c - 1);
        E->rhk[i][g] = ok[h]; E->rhp[i][g] = op[h]; E->rhc[i]++; }
    free(ok); free(op);
    g_eh_tgrow += sc_time() - tg_;
}
static void eh_set(Elim *E, int i, int col, int pos) {          /* insert or update */
    if (2u * ((unsigned)E->rhc[i] + 1u) > E->rhm[i] + 1u) eh_alloc(E, i, 2 * E->rhc[i] + 2);
    const unsigned key = (unsigned)col + 1u, mask = E->rhm[i]; unsigned *hk = E->rhk[i];
    unsigned h = eh_slot(key, mask);
    while (hk[h] != 0 && hk[h] != key) h = (h + 1) & mask;
    if (hk[h] == 0) { hk[h] = key; E->rhc[i]++; }
    E->rhp[i][h] = pos;
}
static void eh_del(Elim *E, int i, int col) {
    const unsigned key = (unsigned)col + 1u, mask = E->rhm[i]; unsigned *hk = E->rhk[i]; int *hp = E->rhp[i];
    unsigned h = eh_slot(key, mask);
    while (hk[h] != key) { if (hk[h] == 0) return; h = (h + 1) & mask; }
    for (unsigned g = (h + 1) & mask; hk[g] != 0; g = (g + 1) & mask) {   /* backward shift */
        const unsigned want = eh_slot(hk[g], mask);
        if (((g - want) & mask) >= ((g - h) & mask)) { hk[h] = hk[g]; hp[h] = hp[g]; h = g; }
    }
    hk[h] = 0; E->rhc[i]--;
}
static void eh_drop(Elim *E, int i) { free(E->rhk[i]); free(E->rhp[i]); E->rhk[i] = NULL; E->rhp[i] = NULL; E->rhm[i] = 0; E->rhc[i] = 0; }
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
static _Thread_local int g_elim_wide = 0;     /* (per thread: the two substitutions may run side by side, socp_solve_elim) */
static int elim_run(Elim *E, const SocpProb *P) {
    const double tel0_ = sc_time();
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
    E->rh = (unsigned char *)calloc((size_t)m + 1, 1);
    E->rhk = (unsigned **)calloc((size_t)m + 1, sizeof(unsigned *)); E->rhp = (int **)calloc((size_t)m + 1, sizeof(int *));
    E->rhm = (unsigned *)calloc((size_t)m + 1, sizeof(unsigned)); E->rhc = (int *)calloc((size_t)m + 1, sizeof(int));
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
    if (g_elim_wide) { el_rowmax = 8; el_fill = 4.0; el_pass = 5; el_col = 0.001; }     /* (the wide rules of the cost-based choice, socp_solve_elim) */
    static int elsing = -1; if (elsing < 0) elsing = getenv("BRISK_CONEELIMSING") ? atoi(getenv("BRISK_CONEELIMSING")) : 1;
    { const char *e_ = getenv("BRISK_CONEELIM"); if (e_) sscanf(e_, "%d,%lf,%d,%lf", &el_rowmax, &el_fill, &el_pass, &el_col); }   /* (test: row length, fill factor, passes, the pivot's share of its column's largest entry) */
    if (dbg_on()) socp_printf("  [substitution: the column route and the rows %.3fs]\n", sc_time() - tel0_);
    g_eh_tgrow = g_eh_thash = 0;
    for (int pass = 0; pass < el_pass; pass++) {
        const double tp_ = sc_time();
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
            if (E->rh[pi]) { eh_drop(E, pi); E->rh[pi] = 0; }      /* (the pivot row leaves: its table with it) */
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
                if (!E->rh[k]) { const double th_ = sc_time(); E->rh[k] = 1; eh_alloc(E, k, lk + li); for (int t = 0; t < lk; t++) eh_set(E, k, E->rcol[k][t], t); g_eh_thash += sc_time() - th_; }
                if (lk + li > E->rcap[k]) { E->rcap[k] = lk + li + (lk + li) / 2 + 4; E->rcol[k] = (int *)realloc(E->rcol[k], sizeof(int) * (size_t)E->rcap[k]); E->rval[k] = (double *)realloc(E->rval[k], sizeof(double) * (size_t)E->rcap[k]); }
                for (int t = 0; t < li; t++) {
                    const int col = ic[t], pos = row_find(E, k, col);
                    if (col == u) { if (pos >= 0) row_del(E, k, pos); continue; }
                    if (pos < 0) {
                        const int w = E->rlen[k]++;
                        E->rcol[k][w] = col; E->rval[k][w] = -mk * iv[t]; eh_set(E, k, col, w);
                        if (col < nf) { E->fcnt[col]++; flist_add(E, col, k); }
                    } else {
                        /* (4 November 2026, the SOCP developer, a test: BRISK_CONEELIMCANCEL=1) a matrix entry that cancels to
                         * rounding is zero, by the rule b and c already follow (elim_sub, 1e-12 of the larger term); kept, such
                         * residues (4.4e-32 after the equilibration) made rows of the wide substitution's problem nearly empty, and
                         * in double-double their pivots overflowed L (joint_HO_24, SOCP; ISSUES 823) */
                        static int ecanc = -1; if (ecanc < 0) ecanc = getenv("BRISK_CONEELIMCANCEL") ? atoi(getenv("BRISK_CONEELIMCANCEL")) : 0;
                        const double v = ecanc ? elim_sub(E->rval[k][pos], mk * iv[t]) : E->rval[k][pos] - mk * iv[t];
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
        if (dbg_on()) socp_printf("  [substitution pass %d: %d done, %.3fs (the index: rows entered %.3fs, tables made %.3fs)]\n", pass, done, sc_time() - tp_, g_eh_thash, g_eh_tgrow);
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
    if (E->rhk) for (int i = 0; i < E->m; i++) if (E->rhk[i]) eh_drop(E, i);
    free(E->rhk); free(E->rhp); free(E->rhm); free(E->rhc); free(E->rh); free(E->esr); free(E->esc);
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
/* (14 November 2026, the SOCP developer, a test: BRISK_CONEFREECONE=2) each free variable x_j in its own cone (t_j, x_j) of order 2,
 * t_j a new variable with no rows and no cost: the same problem (t_j >= |x_j| is no restriction), no free variables, and no single
 * cone of the order of all the free variables together. joint_FC_7 (SOCP) with the one cone of mode 1: the cone of order 104,097
 * lags behind the others (its products / mu up to 4.9e+02 against at most 10) and its part of the direction loses the primal digits
 * at iteration 44 (the primal residual grew 64 times in one step). The cones go after the second-order ones, before the rotated. */
static int core_freecone2(const SocpProb *Q, const SocpOpts *o, SocpRes *R) {
    const int nf = Q->nf, n = Q->n, nnz = Q->Ap[n], n2 = n + nf;
    int nrot = 0; for (int k = 0; k < Q->nr; k++) nrot += Q->r[k];
    const int cut = n - nrot;                                   /* (the original columns before the rotated cones) */
    int *Ap = (int *)malloc(sizeof(int) * ((size_t)n2 + 1)), *Ai = (int *)malloc(sizeof(int) * ((size_t)nnz + 1));
    double *Ax = (double *)malloc(sizeof(double) * ((size_t)nnz + 1)), *c2 = (double *)malloc(sizeof(double) * ((size_t)n2 + 1));
    int *q2 = (int *)malloc(sizeof(int) * ((size_t)Q->nq + nf + 1)), *perm = (int *)malloc(sizeof(int) * ((size_t)n2 + 1));
    int w = 0, j2 = 0; Ap[0] = 0;
    #define PUTCOL(src) do { for (int p = Q->Ap[src]; p < Q->Ap[(src) + 1]; p++) { Ai[w] = Q->Ai[p]; Ax[w++] = Q->Ax[p]; } c2[j2] = Q->c[src]; perm[j2] = (src); Ap[++j2] = w; } while (0)
    for (int j = nf; j < cut; j++) PUTCOL(j);                    /* the linear columns and the second-order cones */
    for (int j = 0; j < nf; j++) { c2[j2] = 0.0; perm[j2] = -1; Ap[j2 + 1] = w; j2++; PUTCOL(j); }   /* (t_j, x_j) */
    for (int j = cut; j < n; j++) PUTCOL(j);                     /* the rotated cones */
    #undef PUTCOL
    for (int k = 0; k < Q->nq; k++) q2[k] = Q->q[k];
    for (int k = 0; k < nf; k++) q2[Q->nq + k] = 2;
    SocpProb Q2 = *Q; Q2.n = n2; Q2.nf = 0; Q2.nq = Q->nq + nf; Q2.q = q2; Q2.Ap = Ap; Q2.Ai = Ai; Q2.Ax = Ax; Q2.c = c2;
    if (o && o->verbose > 0) socp_printf("presolve: %d free variables each in a cone of order 2 (test)\n", nf);
    const int st = socp_solve_core(&Q2, o, R);
    if (R->x && R->z) {
        double *x = (double *)malloc(sizeof(double) * ((size_t)n + 1)), *z = (double *)malloc(sizeof(double) * ((size_t)n + 1));
        for (int j = 0; j < n2; j++) if (perm[j] >= 0) { x[perm[j]] = R->x[j]; z[perm[j]] = R->z[j]; }
        for (int j = 0; j < nf; j++) z[j] = 0.0;
        free(R->x); free(R->z); R->x = x; R->z = z;
    }
    free(Ap); free(Ai); free(Ax); free(c2); free(q2); free(perm);
    return st;
}
static int core_freecone(const SocpProb *Q, const SocpOpts *o, SocpRes *R) {
    const int nf = Q->nf, n = Q->n, m = Q->m;
    static int on = -1; if (on < 0) { const char *e_ = getenv("BRISK_CONEFREECONE"); on = e_ ? atoi(e_) : 0; }
    if (nf == 0 || !on) return core_split(Q, o, R);
    if (on == 2) return core_freecone2(Q, o, R);
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
/* (3 November 2026, the SOCP developer) the cost of the system a substitution leaves: AMD's multiply-subtract pairs for the
 * cheaper of the full system (rows and the columns left, a cone's columns a clique) and the normal equations (each linear
 * column and each cone joins its rows into a clique) with the free columns left as a border of unknowns joined to their
 * rows. The two counts the solver's own route choice rests on; the pairs of rows of the normal equations are capped (above
 * 5e7 the form is not counted) */
/* (3 November 2026) the chosen reduction's counted full system, for the solver that follows (BRISK_CONEMEASDEFER) */
static _Thread_local int g_meas_ok; static _Thread_local double g_meas_pairs, g_meas_lnz;
static double elim_cost(const Elim *E, const SocpProb *P, double *fpair, double *flnz) {
    const int m = P->m, n = P->n, nf = P->nf, nl = P->nl; const double tq0_ = sc_time();
    int *cmap = (int *)malloc(sizeof(int) * ((size_t)n + 1)), *rmap = (int *)malloc(sizeof(int) * ((size_t)m + 1)), *grp = (int *)malloc(sizeof(int) * ((size_t)n + 1));
    int n2 = 0, m2 = 0;
    for (int j = 0; j < n; j++) cmap[j] = E->cgone[j] ? -1 : n2++;
    for (int i = 0; i < m; i++) rmap[i] = E->ralive[i] ? m2++ : -1;
    for (int j = 0; j < n; j++) grp[j] = j < nf ? -1 : j < nf + nl ? j - nf : -2;
    int *qs = (int *)malloc(sizeof(int) * ((size_t)P->nq + 1)); qs[0] = nf + nl; for (int k = 0; k < P->nq; k++) qs[k + 1] = qs[k] + P->q[k];     /* (the cones' columns follow the free and linear ones) */
    /* (the normal equations' groups: a linear column alone, a cone of at most NE_MAXCONE entries its columns together; a larger
     * cone's columns each alone, its low-rank part the solver's separate update: as the solver forms them) */
    int ngr = nl + P->nq;
    for (int k = 0; k < P->nq; k++) for (int t = 0; t < P->q[k]; t++) grp[qs[k] + t] = P->q[k] <= NE_MAXCONE ? nl + k : ngr++;
    /* the reduced matrix by columns */
    int *cp = (int *)calloc((size_t)n + 2, sizeof(int)); long nz = 0;
    for (int i = 0; i < m; i++) if (E->ralive[i]) for (int t = 0; t < E->rlen[i]; t++) { cp[E->rcol[i][t] + 1]++; nz++; }
    for (int j = 0; j < n; j++) cp[j + 1] += cp[j];
    int *ci = (int *)malloc(sizeof(int) * ((size_t)nz + 1)), *nx = (int *)malloc(sizeof(int) * ((size_t)n + 1)); memcpy(nx, cp, sizeof(int) * (size_t)n);
    for (int i = 0; i < m; i++) if (E->ralive[i]) for (int t = 0; t < E->rlen[i]; t++) ci[nx[E->rcol[i][t]]++] = rmap[i];
    /* the full system */
    /* (a cone of at most SOCP_DENSE_MAX entries a clique of its columns, a larger one two auxiliary unknowns joined to its
     * columns and to each other: the solver's own forms of the full system) */
    long ncl = 0; int naux = 0; for (int k = 0; k < P->nq; k++) { if (P->q[k] <= SOCP_DENSE_MAX) ncl += (long)P->q[k] * (P->q[k] - 1) / 2; else { ncl += 2L * P->q[k] + 1; naux += 2; } }
    int *pa = (int *)malloc(sizeof(int) * ((size_t)(nz + ncl) + 1)), *pb = (int *)malloc(sizeof(int) * ((size_t)(nz + ncl) + 1)); long e = 0;
    for (int j = 0; j < n; j++) if (cmap[j] >= 0) for (int p = cp[j]; p < cp[j + 1]; p++) { pa[e] = cmap[j]; pb[e] = n2 + ci[p]; e++; }
    { int au = n2 + m2;
      for (int k = 0; k < P->nq; k++) {
        if (P->q[k] <= SOCP_DENSE_MAX) { for (int a = 0; a < P->q[k]; a++) for (int b = a + 1; b < P->q[k]; b++) {
            const int ja = cmap[qs[k] + a], jb = cmap[qs[k] + b]; if (ja >= 0 && jb >= 0) { pa[e] = ja; pb[e] = jb; e++; } } }
        else { for (int a = 0; a < P->q[k]; a++) { const int ja = cmap[qs[k] + a]; if (ja < 0) continue; pa[e] = ja; pb[e] = au; e++; pa[e] = ja; pb[e] = au + 1; e++; }
               pa[e] = au; pb[e] = au + 1; e++; au += 2; } } }
    const double tq1_ = sc_time();
    const double full = nem_amd(dbg_on() ? "the full system after the substitution" : NULL, n2 + m2 + naux, e, pa, pb);
    if (fpair) *fpair = full; if (flnz) *flnz = g_nem_lnz;
    if (full >= 1e299) { free(cmap); free(rmap); free(grp); free(cp); free(ci); free(nx); free(qs); return 1e300; }     /* (stopped at the cap: it loses) */
    free(pa); free(pb);
    /* the normal equations with the free columns as a border */
    double ne = 1e300;
    {
        const int ng = ngr; int *gs = (int *)calloc((size_t)ng + 2, sizeof(int));
        for (int j = nf; j < n; j++) if (cmap[j] >= 0 && grp[j] >= 0) gs[grp[j] + 1] += cp[j + 1] - cp[j];
        for (int g = 0; g < ng; g++) gs[g + 1] += gs[g];
        int *gr = (int *)malloc(sizeof(int) * ((size_t)gs[ng] + 1)); int *gx = (int *)malloc(sizeof(int) * ((size_t)ng + 1)); memcpy(gx, gs, sizeof(int) * (size_t)ng);
        for (int j = nf; j < n; j++) if (cmap[j] >= 0 && grp[j] >= 0) for (int p = cp[j]; p < cp[j + 1]; p++) gr[gx[grp[j]]++] = ci[p];
        int *mk = (int *)malloc(sizeof(int) * ((size_t)m2 + 1)); for (int i = 0; i < m2; i++) mk[i] = -1;
        /* (dense rows (more than 10 sqrt(m) entries, AMD's own threshold) left out of the pairs: AMD and the solver set them
         * apart; with them a single row made tens of millions of duplicate pairs) */
        { int *rl_ = (int *)calloc((size_t)m2 + 1, sizeof(int)); for (long q = 0; q < nz; q++) rl_[ci[q]]++;
          const int dth = (int)fmax(16.0, 10.0 * sqrt((double)m2)); for (int i = 0; i < m2; i++) if (rl_[i] > dth) mk[i] = -2; free(rl_); }
        const double npcap = fmin(5e7, full);      /* (the normal equations' factor costs at least as many pairs as their matrix has entries: above the full system's count they cannot be cheaper) */
        /* the graph row by row without duplicates: each row's neighbours are the rows of the groups that contain it (a mark per
         * row), and the free columns left as border vertices; no list of pairs, no sort (a row in many groups gave tens of
         * millions of duplicate pairs: socp_rocket_4000, SOCP, 2.4 s) */
        int *gu = (int *)malloc(sizeof(int) * ((size_t)gs[ng] + 1)), *gup = (int *)malloc(sizeof(int) * ((size_t)ng + 1));
        { long o = 0; for (int g = 0; g < ng; g++) { gup[g] = (int)o; for (int p = gs[g]; p < gs[g + 1]; p++) { const int r_ = gr[p]; if (mk[r_] != g && mk[r_] != -2) { mk[r_] = g; gu[o++] = r_; } } } gup[ng] = (int)o; }
        int *rgp = (int *)calloc((size_t)m2 + 2, sizeof(int));
        for (int g = 0; g < ng; g++) for (int p = gup[g]; p < gup[g + 1]; p++) rgp[gu[p] + 1]++;
        for (int i = 0; i < m2; i++) rgp[i + 1] += rgp[i];
        int *rg = (int *)malloc(sizeof(int) * ((size_t)rgp[m2] + 1)); int *rx = (int *)malloc(sizeof(int) * ((size_t)m2 + 1)); memcpy(rx, rgp, sizeof(int) * (size_t)m2);
        for (int g = 0; g < ng; g++) for (int p = gup[g]; p < gup[g + 1]; p++) rg[rx[gu[p]]++] = g;
        int nb = 0; for (int j = 0; j < nf; j++) if (cmap[j] >= 0) nb++;
        int *fb = (int *)malloc(sizeof(int) * ((size_t)nf + 1)); { int t = 0; for (int j = 0; j < nf; j++) fb[j] = cmap[j] >= 0 ? t++ : -1; }
        int *rfp = (int *)calloc((size_t)m2 + 2, sizeof(int));          /* (each row's free columns) */
        for (int j = 0; j < nf; j++) if (fb[j] >= 0) for (int p = cp[j]; p < cp[j + 1]; p++) rfp[ci[p] + 1]++;
        for (int i = 0; i < m2; i++) rfp[i + 1] += rfp[i];
        int *rf = (int *)malloc(sizeof(int) * ((size_t)rfp[m2] + 1)); memcpy(rx, rfp, sizeof(int) * (size_t)m2);
        for (int j = 0; j < nf; j++) if (fb[j] >= 0) for (int p = cp[j]; p < cp[j + 1]; p++) rf[rx[ci[p]]++] = fb[j];
        const int nv = m2 + nb; int *stp = (int *)malloc(sizeof(int) * ((size_t)nv + 1)); for (int v = 0; v < nv; v++) stp[v] = -1;
        int *Ap_ = (int *)malloc(sizeof(int) * ((size_t)nv + 1)); double tot = 0; int ok = 1;
        Ap_[0] = 0;
        for (int i = 0; i < m2 && ok; i++) { long d_ = 0;
            for (int q = rgp[i]; q < rgp[i + 1]; q++) { const int g = rg[q]; for (int p = gup[g]; p < gup[g + 1]; p++) { const int k = gu[p]; if (k != i && stp[k] != i) { stp[k] = i; d_++; } } }
            d_ += rfp[i + 1] - rfp[i]; tot += (double)d_; if (tot > npcap + 2.0 * (double)rfp[m2]) ok = 0; Ap_[i + 1] = (int)tot; }
        if (ok) {
            for (int t = 0; t < nb; t++) Ap_[m2 + t + 1] = 0;
            for (int i = 0; i < m2; i++) for (int q = rfp[i]; q < rfp[i + 1]; q++) Ap_[m2 + rf[q] + 1]++;
            for (int t = 0; t < nb; t++) Ap_[m2 + t + 1] += Ap_[m2 + t];
            int *Ai_ = (int *)malloc(sizeof(int) * ((size_t)Ap_[nv] + 1)); int *wp = (int *)malloc(sizeof(int) * ((size_t)nv + 1)); memcpy(wp, Ap_, sizeof(int) * (size_t)nv);
            for (int v = 0; v < nv; v++) stp[v] = -1;
            for (int i = 0; i < m2; i++) {
                for (int q = rgp[i]; q < rgp[i + 1]; q++) { const int g = rg[q]; for (int p = gup[g]; p < gup[g + 1]; p++) { const int k = gu[p]; if (k != i && stp[k] != i) { stp[k] = i; Ai_[wp[i]++] = k; } } }
                for (int q = rfp[i]; q < rfp[i + 1]; q++) { Ai_[wp[i]++] = m2 + rf[q]; Ai_[wp[m2 + rf[q]]++] = i; }
            }
            int *Pm = (int *)malloc(sizeof(int) * (size_t)nv); double ctl[AMD_CONTROL], info[AMD_INFO]; amd_defaults(ctl);
            const double t_ = sc_time(); const int st = amd_order(nv, Ap_, Ai_, Pm, ctl, info);
            ne = st < 0 || brisk_amd_capped ? 1e300 : info[AMD_NMULTSUBS_LDL]; brisk_amd_capped = 0;
            if (dbg_on()) socp_printf("  [the measure, the normal equations after the substitution (free columns as a border): %d unknowns, %d entries of the graph; AMD in %.2fs: nnz(L) %.4g, multiply-subtract pairs %.4g]\n", nv, Ap_[nv], sc_time() - t_, info[AMD_LNZ], ne);
            free(Ai_); free(wp); free(Pm);
        }
        free(gu); free(gup); free(rgp); free(rg); free(rx); free(fb); free(rfp); free(rf); free(stp); free(Ap_);
        free(gs); free(gr); free(gx); free(mk);
    }
    if (dbg_on()) socp_printf("  [the count's stages: the matrix and the full system's graph %.2fs, the rest %.2fs]\n", tq1_ - tq0_, sc_time() - tq1_);
    free(cmap); free(rmap); free(grp); free(cp); free(ci); free(nx); free(qs);
    return fmin(full, ne);
}
static int socp_solve_elim(const SocpProb *P, const SocpOpts *opt, SocpRes *R) {
    if (P->nf == 0 || getenv("BRISK_CONENOELIM")) return socp_solve_core(P, opt, R);
    const double t0 = sc_time();
    const int m = P->m, n = P->n, nf = P->nf;
    Elim E_; Elim *E = &E_; memset(E, 0, sizeof *E);
    /* (3 November 2026, the SOCP developer, a test: BRISK_CONEELIMCOST=1) two substitutions, by the default rules and by
     * wider ones (rows up to 8 entries, fill up to 4 times the entries removed, 5 passes, a pivot down to 0.001 of its
     * column's largest entry), and the one whose system costs less to factor (elim_cost) is kept. db-joint-soerensen (SOCP):
     * the default rules keep 289,728 free variables (full system 1.45e9 pairs), the wide ones none (the normal equations,
     * 9e6 pairs), and the free variables' rows were where its solve stalled; nql90 (SOCP): the wide rules remove 5,396 more
     * but the matrix grows from 170k to 219k entries and a solve takes 6.9 s instead of 2.6 */
    static int ecost_c = -2; const int ecost = cone_on("BRISK_CONEELIMCOST", &ecost_c);
    if (ecost && !getenv("BRISK_CONEELIM")) {
        const double tc0 = sc_time();
        /* (4 November 2026, the SOCP developer) the wide rules' result is a candidate only when it leaves no free variable: their
         * purpose is the normal equations, whose matrix is positive definite once every free variable is gone (db-joint-soerensen,
         * SOCP). Where free variables stay, the full system is factored either way and the wide result lost: joint_HO_24 and
         * joint_HO_26 (SOCP) ended with reduced accuracy (the last iterations moved the objective by 1.6e-6 and 4.4e-6, with the
         * wide rules' pivots capped at 0.01 as well), joint_HO_23 (SOCP) 4.2e-7 off the optimum while OPTIMAL, mpc_300,
         * socp_snl_20000 and strictmin_2D_43_dual (SOCP) slower; the counts of both systems are skipped then (about 4 s on a
         * joint_HO file). At -acc low (cone_pkg's tolerance) the wide result stays a candidate with free variables left: that
         * stop comes long before mu reaches 1e-12, where the end broke (ISSUES 823), and the joint_HO files gain there (the same
         * bits as before). BRISK_CONEELIMWIDEPART=1 or 0: always or never (a test) */
        static int ewpart_ = -2; if (ewpart_ == -2) ewpart_ = getenv("BRISK_CONEELIMWIDEPART") ? atoi(getenv("BRISK_CONEELIMWIDEPART")) : -1;
        const int ewpart = ewpart_ >= 0 ? ewpart_ : g_cone_lowtol;
        Elim W_; Elim *W = &W_; memset(W, 0, sizeof *W);
        /* (3 November 2026) on a large problem the two substitutions and their counts side by side on two threads: each works on
         * its own Elim and only reads P, so the result is the same as one after the other (db-joint-soerensen, SOCP: 6.5 -> about
         * 3.7 s); a small problem as before, the wide rules only when free variables are left */
        int r0 = 0, r1 = 0, left = 0; double c0 = 1e300, c1 = 1e300, tc1 = 0, tc2 = 0, tc3 = 0, fp0 = 0, fl0 = 0, fp1 = 0, fl1 = 0; int both = 0;
        if ((long)P->Ap[P->n] >= 1000000L && g_cone_nt > 1 && !omp_in_parallel()) {
            #pragma omp parallel sections num_threads(2)
            {
                #pragma omp section
                { g_elim_wide = 0; r0 = elim_run(E, P); }
                #pragma omp section
                { g_elim_wide = 1; r1 = elim_run(W, P); g_elim_wide = 0; }
            }
            tc1 = tc2 = sc_time(); left = 1; both = 1;
            if (r1 && !ewpart) { int lw = 0; for (int j = 0; j < nf; j++) if (!W->cgone[j]) lw++; if (lw > 0) r1 = 0; }
            if (!r1) { elim_free(W); memset(W, 0, sizeof *W); }
            if (r1 && W->ne > (r0 ? E->ne : 0)) {      /* (the counts only when the wide rules removed more: beam7, SOCP, 3.5 s of counts for nothing) */
                /* (the default's count, once known, caps the wide one's orderings: AMD stops when the wide count passes 2.2 times
                 * it, where the wide reduction loses anyway; the choice is the same, only the time differs. qssp180old (SOCP):
                 * the wide reduction's orderings 3.7 s for 1.8e10 against 1.7e9) */
                /* (5 November 2026, the SOCP developer: the cap removed. Whether it acted depended on which count finished first, and
                 * a capped full system skipped the wide reduction's normal-equations count, which can be the cheaper one: the
                 * choice changed from run to run at 2 threads or more (socp_snl_20000, SOCP, -acc low: 135,623 or 120,955 variables
                 * substituted; at one thread, where the counts run one after the other without a cap, always 135,623). Now both
                 * counts are complete, as at one thread: the same choice at any thread count) */
                #pragma omp parallel sections num_threads(2)
                {
                    #pragma omp section
                    { if (r0) c0 = elim_cost(E, P, &fp0, &fl0); }
                    #pragma omp section
                    { c1 = elim_cost(W, P, &fp1, &fl1); }
                }
            }
            tc3 = sc_time();
        } else {
            g_elim_wide = 0; r0 = elim_run(E, P);
            tc1 = sc_time();
            if (r0) { for (int j = 0; j < nf; j++) if (!E->cgone[j]) left++; } else left = nf;
            if (left > 0) { g_elim_wide = 1; r1 = elim_run(W, P); g_elim_wide = 0; }     /* (no free variable left: nothing for the wide rules) */
            else memset(W, 0, sizeof *W);
            if (r1 && !ewpart) { int lw = 0; for (int j = 0; j < nf; j++) if (!W->cgone[j]) lw++; if (lw > 0) { elim_free(W); memset(W, 0, sizeof *W); r1 = 0; left = 0; } }
        }
        if (r1 && W->ne > (r0 ? E->ne : 0)) {
            if (!both) { tc2 = sc_time(); c0 = r0 ? elim_cost(E, P, &fp0, &fl0) : 1e300; tc3 = sc_time(); c1 = elim_cost(W, P, &fp1, &fl1); }
            if (opt && opt->verbose > 0 && dbg_on()) socp_printf("  [substitution by cost: the default rules %d variables, %.3g pairs; the wide rules %d, %.3g pairs; %.2fs (the default rules %.2f, the wide %.2f, their costs %.2f and %.2f)]\n", r0 ? E->ne : 0, c0, W->ne, c1, sc_time() - tc0, tc1 - tc0, tc2 - tc1, tc3 - tc2, sc_time() - tc3);
            if (c1 < c0) { elim_free(E); *E = *W; g_meas_pairs = fp1; g_meas_lnz = fl1; } else { elim_free(W); g_meas_pairs = fp0; g_meas_lnz = fl0; }
            g_meas_ok = g_meas_pairs > 0 && g_meas_lnz > 0;
        } else if (left > 0) elim_free(W);
        if (!r0 && E->ne == 0) { elim_free(E); return socp_solve_core(P, opt, R); }
    } else
    if (elim_run(E, P) == 0) { elim_free(E); return socp_solve_core(P, opt, R); }
    if (opt && opt->verbose > 0 && dbg_on()) socp_printf("  [substitution: elim_run %.3fs]\n", sc_time() - t0);
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
    if (opt && opt->verbose > 0) socp_user_presolve("%d free variable%s eliminated by substitution (rows %d -> %d)", E->ne, E->ne == 1 ? "" : "s", m, m2);
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
    } else {
        /* (the default log's rows: c'x = c2'x2 + K and b'y = b2'y2 + K, K = sum over the substitutions of c_u b_i / a_iu) */
        double K = 0.0; for (int e = 0; e < E->ne; e++) K += E->ecu[e] * E->ebi[e] / E->epiv[e];
        const double ra0 = g_row_a, rk0 = g_row_k; g_row_k = rk0 + ra0 * K;
        st = socp_drop_empty(&Q, &o2, R, core_freecone);      /* (cones tied to free copies are empty only now) */
        g_row_a = ra0; g_row_k = rk0;
    }
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
    if (st == SOCP_OPTIMAL && (g_fr_face_err >= 0 ? fmax(R->err[0], g_fr_face_err) : R->maxerr) > 10.0 * o2.tol) st2 = SOCP_REDUCED;     /* (a face's result: judged on the primal here and the dual on the face, whose multipliers exist) */
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
    if (opt && opt->verbose > 0) socp_user_presolve("%d pairs of linear variables (a, -a) treated as free variables", np);
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

/* (8 November 2026, the SOCP developer) the mean complementarity product of the start's pairs, each linear pair and each cone (x_k'z_k)
 * one pair, without the cones marked in skip (those whose start is being set from it); dflt when there is none. The starts of the
 * fixed cones and of the epigraph cones take their central point at it (BRISK_CONEFIXMU=1): the linear part's mean alone is 1 when
 * there is no linear part, whatever the shared shifts made the other products */
static double start_mu_rest(const Socp *S, const unsigned char *skip, double dflt) {
    double sxz = 0; long np_ = 0;
    for (int i = S->nf; i < S->nf + S->nl; i++) { sxz += S->x[i] * S->z[i]; np_++; }
    for (int k = 0; k < S->nq; k++) {
        if (skip && skip[k]) continue;
        const int j = S->qs[k], d = S->q[k]; double v_ = 0; for (int i = 0; i < d; i++) v_ += S->x[j + i] * S->z[j + i];
        sxz += v_; np_++;
    }
    if (dbg_on()) {     /* (the products' spread: their count, smallest, median and largest) */
        double *pv = (double *)malloc(sizeof(double) * ((size_t)np_ + 1)); long w = 0;
        for (int i = S->nf; i < S->nf + S->nl; i++) pv[w++] = S->x[i] * S->z[i];
        for (int k = 0; k < S->nq; k++) { if (skip && skip[k]) continue; const int j = S->qs[k], d = S->q[k]; double v_ = 0; for (int i = 0; i < d; i++) v_ += S->x[j + i] * S->z[j + i]; pv[w++] = v_; }
        if (w > 0) { qsort(pv, (size_t)w, sizeof(double), cmp_dbl); socp_printf("  [start: the rest's %ld pairs (%d linear): products %.3e, median %.3e, largest %.3e]\n", w, S->nl, pv[0], pv[w / 2], pv[w - 1]); }
        free(pv);
    }
    if (start_fixmu() >= 2) {     /* (2, 3: the start's mean product with the cones being set counted at dflt: (S_rest + n_skip dflt) / (n_rest + n_skip);
                                    * friction contacts, 51,824 other pairs at 58 and 51,823 fixed cones: 29.5; group sparse, one other pair at 34 and
                                    * 1,408 fixed cones: 1.02) */
        long ns_ = 0; if (skip) for (int k = 0; k < S->nq; k++) ns_ += skip[k] != 0;
        const double m2 = (sxz + (double)ns_ * dflt) / fmax(1.0, (double)(np_ + ns_));
        /* (the start is left as it is where the two agree within a factor of 2: the rule repairs a gross inconsistency, the
         * fixed cones at 1 against a start at 30-60, and a perturbation of a consistent start only moves its end: P00_00,
         * SOCP, 125,000 fixed cones and one other pair, mu 1.0002 instead of 1, 7 -> 8 iterations) */
        if (start_fixmu() == 3 && m2 > 0 && m2 <= 2.0 * dflt && m2 >= 0.5 * dflt) return dflt;
        return m2 > 0 && isfinite(m2) ? m2 : dflt;
    }
    return np_ > 0 && sxz > 0 && isfinite(sxz) ? sxz / (double)np_ : dflt;
}
static int start_fixmu(void) { static int v = -1; if (v < 0) { const char *e_ = getenv("BRISK_CONEFIXMU"); v = e_ ? atoi(e_) : 3; } return v; }     /* (8 November 2026: 3 the default, ISSUES 852; 0 off; 1, 2 the earlier tests) */
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
void schol_set_cone_tiles(int on);
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
        SNOTE("presolve: a block in no row of A has its cost outside the dual cone: dual infeasible, the certificate is that ray"); socp_user_presolve("a block in no row of A has its cost outside the dual cone: the problem is dual infeasible");
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
    if (opt && opt->verbose > 0) socp_user_presolve("%d variables and cones in no row of A set to 0 (columns %d -> %d)", ngone, n, n2);
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
#define SOCP_DUAL_UNUSED (-77)     /* (socp_solve_dualform with g_dual_only: the dual form not taken, nothing solved) */
static int g_dual_only = 0;
/* (2 November 2026, the SOCP developer; BRISK_CONENULL=0: off) the null-space form of a dense problem with few more columns than
 * rows. With A of full row rank m and k = n - m, the feasible x are x = x0 - N t, t in R^k, N a basis of the null space of A
 * (n x k) and x0 a particular solution. Since c - z is in the range of A' exactly when N'(c - z) = 0, and b'y = x0'(c - z), the
 * dual problem is, in its slack z alone,
 *     min x0'z  subject to  N'z = N'c,  z in K   (k rows, the cones of the problem),
 * a problem of the same form: its multipliers t give x = x0 - N t (its own dual slack), and y solves A'y = c - z. One LU
 * factorization of A' with partial pivoting, A' = P L U (P a permutation, L = [L1; L2] unit lower trapezoidal, U upper of
 * order m), gives both: with u = P'x, A x = 0 exactly when L1'u1 + L2'u2 = 0, so N = P [-W; I_k] with W = L1^-T L2' (m x k),
 * x0 = P [L1^-T U^-T b; 0], and y = U^-1 L1^-1 (P'(c - z))_{1:m}. An infeasibility certificate of one form is one of the
 * other: a ray z (N'z = 0, x0'z < 0) gives y = -U^-1 L1^-1 (P'z)_{1:m} with -A'y = z and b'y > 0; a certificate t of the
 * reduced problem's infeasibility gives the ray x = -N t. The LU costs about two thirds of m^2 (n - m/3) flops once, while the
 * interior-point method on a dense A factors a dense matrix of order m (about m^3 / 3 flops) at every iteration; the reduced
 * problem's iterations cost about n k^2. CBLIB's 2013_firL2a (SOCP, one cone of 10,002 entries over 10,001 rows of half-dense
 * data: the feasible set is a line, k = 1), which MOSEK solves in its presolve. Taken when A is at least a quarter dense,
 * m >= 1000, 16 k <= m, no free variables, the LU's array within the memory caps, and every pivot of U above 1e-12 times the
 * largest (A of full row rank) */
void SC_BL(dgetrf_)(const int *, const int *, double *, const int *, int *, int *);
void SC_BL(dtrsm_)(const char *, const char *, const char *, const char *, const int *, const int *, const double *, const double *, const int *, double *, const int *);
void SC_BL(dtrsv_)(const char *, const char *, const char *, const int *, const double *, const int *, double *, const int *);
#define SOCP_NULL_UNUSED (-78)
/* the LU factorization with partial pivoting of G (r x c, r >= c, column-major, ld r) by blocks of NS_NB columns: each panel by
 * dgetrf, its interchanges on the other columns and the triangular solve and product of the trailing columns by tiles of
 * NS_TB columns on the threads, each tile by one call of one thread (the same arithmetic at any thread count). ipiv 0-based,
 * global. Returns 0, or 1 when a panel is singular */
#define NS_NB 192
#define NS_TB 384
static int ns_lu(double *G, int r, int c, int *ipiv, int nt) {
    const double one = 1.0, mone = -1.0;
    for (int j0 = 0; j0 < c; j0 += NS_NB) {
        const int jb = c - j0 < NS_NB ? c - j0 : NS_NB, rr = r - j0, ld = r;
        int info = 0;
        SC_BL(dgetrf_)(&rr, &jb, G + j0 + (size_t)j0 * ld, &ld, ipiv + j0, &info);
        if (info != 0) return 1;
        for (int i = j0; i < j0 + jb; i++) ipiv[i] += j0 - 1;          /* (1-based within the panel -> 0-based global) */
        const int c1 = j0 + jb, nrest = c - c1, ntl = (j0 + NS_TB - 1) / NS_TB, ntr = (nrest + NS_TB - 1) / NS_TB;
        /* the interchanges on the columns left of the panel (L's earlier columns) and right of it */
        #pragma omp parallel for schedule(static) num_threads(nt) if (nt > 1)
        for (int t = 0; t < ntl + ntr; t++) {
            const int a = t < ntl ? t * NS_TB : c1 + (t - ntl) * NS_TB, b = t < ntl ? (a + NS_TB < j0 ? a + NS_TB : j0) : (a + NS_TB < c ? a + NS_TB : c);
            for (int col = a; col < b; col++) { double *g = G + (size_t)col * ld;
                for (int i = j0; i < j0 + jb; i++) if (ipiv[i] != i) { const double s = g[i]; g[i] = g[ipiv[i]]; g[ipiv[i]] = s; } }
            if (t >= ntl) {         /* U12 = L11^-1 G12, then G22 -= L21 U12, for this tile's columns */
                const int w = b - a, rb = r - c1;
                SC_BL(dtrsm_)("L", "L", "N", "U", &jb, &w, &one, G + j0 + (size_t)j0 * ld, &ld, G + j0 + (size_t)a * ld, &ld);
                if (rb > 0) SC_BL(dgemm_)("N", "N", &rb, &w, &jb, &mone, G + c1 + (size_t)j0 * ld, &ld, G + j0 + (size_t)a * ld, &ld, &one, G + c1 + (size_t)a * ld, &ld);
            }
        }
    }
    return 0;
}
static int socp_solve_nullspace(const SocpProb *P, const SocpOpts *opt, SocpRes *R) {
    static int on = -1; if (on < 0) { const char *e_ = getenv("BRISK_CONENULL"); on = !(e_ && atoi(e_) == 0); }
    const int m = P->m, n = P->n, k = n - m;
    if (!on || g_dual_only || P->nf > 0 || m < 1000 || k < 1 || 16 * k > m) return SOCP_NULL_UNUSED;
    if (opt && opt->cutoff < INFINITY) return SOCP_NULL_UNUSED;     /* (a cutoff is on the problem's objective; the reduced problem's is c'x0 - b'y) */
    if ((double)P->Ap[n] < 0.25 * (double)m * (double)n) return SOCP_NULL_UNUSED;
    if ((double)m * (double)n * 8.0 + (double)m * k * 8.0 > 2e9 * brisk_mem_scale()) return SOCP_NULL_UNUSED;
    const double t0 = sc_time();
    const int nt = g_cone_nt > 1 && !omp_in_parallel() ? g_cone_nt : 1;
    double *G = (double *)malloc(sizeof(double) * (size_t)n * m); int *ipiv = (int *)malloc(sizeof(int) * ((size_t)m + 1));
    if (!G || !ipiv) { free(G); free(ipiv); return SOCP_NULL_UNUSED; }
    /* G = A' (n x m): row j of G is column j of A */
    #pragma omp parallel for schedule(static) num_threads(nt) if (nt > 1)
    for (int i = 0; i < m; i++) memset(G + (size_t)i * n, 0, sizeof(double) * (size_t)n);
    for (int j = 0; j < n; j++) for (int p = P->Ap[j]; p < P->Ap[j + 1]; p++) G[j + (size_t)P->Ai[p] * n] = P->Ax[p];
    int bad = ns_lu(G, n, m, ipiv, nt);
    double umax = 0, umin = INFINITY;
    if (!bad) for (int i = 0; i < m; i++) { const double d = fabs(G[i + (size_t)i * n]); if (d > umax) umax = d; if (d < umin) umin = d; }
    if (bad || !(umin > 1e-12 * umax)) {
        if (dbg_on()) socp_printf("  [the null-space form: not taken (A not of full row rank: the smallest pivot %.1e of the largest %.1e; %.3fs)]\n", umin, umax, sc_time() - t0);
        free(G); free(ipiv); return SOCP_NULL_UNUSED;
    }
    /* perm[i]: the column of A in place i after the interchanges */
    int *perm = (int *)malloc(sizeof(int) * (size_t)n); for (int j = 0; j < n; j++) perm[j] = j;
    for (int i = 0; i < m; i++) if (ipiv[i] != i) { const int s = perm[i]; perm[i] = perm[ipiv[i]]; perm[ipiv[i]] = s; }
    /* W = L1^-T L2' (m x k) */
    double *W = (double *)malloc(sizeof(double) * (size_t)m * k);
    for (int t = 0; t < k; t++) for (int i = 0; i < m; i++) W[i + (size_t)t * m] = G[m + t + (size_t)i * n];
    { const double one = 1.0; SC_BL(dtrsm_)("L", "L", "T", "U", &m, &k, &one, G, &n, W, &m); }
    /* x0 = P [L1^-T U^-T b; 0] */
    double *x0 = (double *)calloc((size_t)n, sizeof(double)), *v = (double *)malloc(sizeof(double) * ((size_t)(m > n ? m : n) + 1));
    { const int inc = 1; for (int i = 0; i < m; i++) v[i] = P->b[i];
      SC_BL(dtrsv_)("U", "T", "N", &m, G, &n, v, &inc); SC_BL(dtrsv_)("L", "T", "U", &m, G, &n, v, &inc);
      for (int i = 0; i < m; i++) x0[perm[i]] = v[i]; }
    /* the reduced problem: N' by columns (column perm[i], i < m: -W(i, :); column perm[m + t]: 1 in row t), b2 = N'c, c2 = x0 */
    int *Ap2 = (int *)calloc((size_t)n + 1, sizeof(int)); long nz2 = 0;
    for (int i = 0; i < m; i++) { int cnt = 0; for (int t = 0; t < k; t++) cnt += W[i + (size_t)t * m] != 0.0; Ap2[perm[i] + 1] = cnt; nz2 += cnt; }
    for (int t = 0; t < k; t++) { Ap2[perm[m + t] + 1] = 1; nz2++; }
    for (int j = 0; j < n; j++) Ap2[j + 1] += Ap2[j];
    int *Ai2 = (int *)malloc(sizeof(int) * ((size_t)nz2 + 1)); double *Ax2 = (double *)malloc(sizeof(double) * ((size_t)nz2 + 1));
    double *b2 = (double *)calloc((size_t)k + 1, sizeof(double));
    for (int i = 0; i < m; i++) { int w = Ap2[perm[i]]; const double cj = P->c[perm[i]];
        for (int t = 0; t < k; t++) { const double a = W[i + (size_t)t * m]; if (a != 0.0) { Ai2[w] = t; Ax2[w++] = -a; b2[t] -= a * cj; } } }
    for (int t = 0; t < k; t++) { const int j = perm[m + t]; Ai2[Ap2[j]] = t; Ax2[Ap2[j]] = 1.0; b2[t] += P->c[j]; }
    SocpProb Q = *P; Q.m = k; Q.Ap = Ap2; Q.Ai = Ai2; Q.Ax = Ax2; Q.b = b2; Q.c = x0;
    SocpOpts o2; if (opt) o2 = *opt; else socp_default_opts(&o2);
    const double tred = sc_time() - t0;
    if (o2.timelimit > 0) o2.timelimit = fmax(o2.timelimit - tred, 1e-3);     /* (the LU's time counts against the limit) */
    if (o2.verbose > 0) socp_printf("the null-space form solved: %d row%s instead of %d (A' factored in %.2fs)\n", k, k == 1 ? "" : "s", m, tred);
    SNOTED("the null-space form solved by the SOCP solver: %d row%s instead of %d", k, k == 1 ? "" : "s", m);
    const char *pfx = g_stat_prefix; g_stat_prefix = "the null-space form: ";
    /* (the default log's rows: the reduced problem's objective x0'z is c'x0 - b'y, its dual bound (N'c)'t is c'x0 - c'x: the two
     * objectives exchanged about K = c'x0 and negated) */
    double Kc = 0.0; for (int j = 0; j < n; j++) Kc += P->c[j] * x0[j];
    const double ra0 = g_row_a, rk0 = g_row_k; const int rs0 = g_row_swap;
    g_row_k = rk0 + ra0 * Kc; g_row_a = -ra0; g_row_swap = !rs0;
    SocpRes R2; memset(&R2, 0, sizeof R2);
    const int st2 = socp_solve_seq_empty(&Q, &o2, &R2);
    g_row_a = ra0; g_row_k = rk0; g_row_swap = rs0; g_stat_prefix = pfx;
    int st = st2 == SOCP_PINF ? SOCP_DINF : st2 == SOCP_DINF ? SOCP_PINF : st2;
    memset(R, 0, sizeof *R);
    R->x = (double *)calloc((size_t)n + 1, sizeof(double)); R->y = (double *)calloc((size_t)m + 1, sizeof(double)); R->z = (double *)calloc((size_t)n + 1, sizeof(double));
    R->iters = R2.iters; R->lnz = R2.lnz;
    const int inc = 1;
    if (R2.x && R2.y && R2.z) {
        const int cert = st2 == SOCP_PINF || st2 == SOCP_DINF;
        if (st2 != SOCP_DINF) {         /* x = x0 - N t (a ray -N t when the reduced problem is infeasible) */
            for (int i = 0; i < m; i++) { double s = 0; for (int t = 0; t < k; t++) s -= W[i + (size_t)t * m] * R2.y[t]; R->x[perm[i]] = (cert ? 0.0 : x0[perm[i]]) - s; }
            for (int t = 0; t < k; t++) R->x[perm[m + t]] = (cert ? 0.0 : x0[perm[m + t]]) - R2.y[t];
        }
        if (st2 != SOCP_PINF) {         /* z from the reduced point, y from A'y = c - z (a certificate y = -U^-1 L1^-1 (P'z)_{1:m} for a ray) */
            for (int j = 0; j < n; j++) R->z[j] = R2.x[j];
            for (int i = 0; i < m; i++) v[i] = cert ? -R->z[perm[i]] : P->c[perm[i]] - R->z[perm[i]];
            SC_BL(dtrsv_)("L", "N", "U", &m, G, &n, v, &inc); SC_BL(dtrsv_)("U", "N", "N", &m, G, &n, v, &inc);
            for (int i = 0; i < m; i++) R->y[i] = v[i];
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
    if (dbg_on() && (st == SOCP_PINF || st == SOCP_DINF)) {     /* (the certificate checked on the problem as given) */
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
    free(G); free(ipiv); free(perm); free(W); free(x0); free(v); free(Ap2); free(Ai2); free(Ax2); free(b2);
    return st;
}
static int socp_solve_dualform(const SocpProb *P, const SocpOpts *opt, SocpRes *R);
static int fr_inner_dualform(const SocpProb *P, const SocpOpts *opt, SocpRes *R, double t0) { (void)t0; return socp_solve_dualform(P, opt, R); }
static int socp_solve_dualform(const SocpProb *P, const SocpOpts *opt, SocpRes *R) {
    /* (10 November 2026, the SOCP developer; a test with BRISK_CONEFR: the faces single rows prove, removed from the problem as given
     * before the choice of the primal or dual form; in the dual form a face of the primal is one of the dual, which the row search
     * does not see: socp_frzero_6000x2000x4, SOCP, dualized, its 400 forced cones missed) */
    { const int on_ = fr_mode(opt);
      if (on_ && on_ != 3 && !g_fr1_outer && !(opt && opt->wx)) {
          g_fr1_outer = 1; int d_ = 0; const int s_ = socp_solve_core_fr1(P, opt, R, sc_time(), &d_, fr_inner_dualform); g_fr1_outer = 0;
          if (d_) return s_; } }
    { const int s_ = socp_solve_nullspace(P, opt, R); if (s_ != SOCP_NULL_UNUSED) return s_; }
    const int m = P->m, n = P->n, nf = P->nf, nl = P->nl, nbk = nl + P->nq + P->nr;
    const char *e = getenv("BRISK_CONEDUAL"); const int mode = e ? atoi(e) : 2;
    if (mode == 0 || m < 1 || n - nf < 1) return g_dual_only ? SOCP_DUAL_UNUSED : socp_solve_seq_empty(P, opt, R);
    unsigned char *slk = (unsigned char *)malloc((size_t)nbk + 1); int nro = 0;
    const int md = dual_shape(P, slk, &nro);
    /* (the dual's free variables are the rows without a slack: free variables are the SOCP solver's weak point, and a
     * dual with many of them was much harder: a branch-flow relaxation 22 -> 47 iterations with 60,001, an AC-OPF
     * relaxation 29 -> 70, a powered-descent problem reduced accuracy after 400. At most NE_MAXFREE, as for the normal
     * equations) */
    static int dmf = -1; if (dmf < 0) { const char *e_ = getenv("BRISK_CONEDUALMAXFREE"); dmf = e_ ? atoi(e_) : ne_maxfree(); }   /* (a test: the cap on the dual's free variables) */
    /* (2 November 2026, the SOCP developer) the cap on the dual's free variables compared with the primal's: a dual with no more
     * free variables than the primal is not worse on that account, and the estimate of the factor below decides. CBLIB
     * db-joint-soerensen (SOCP, a plasticity limit analysis): the primal 1,040,496 free variables, the dual 979,219; the dual
     * form refused by the cap, the primal's dual residual stalled at 2e-7 for over 1,500 s; the dual form forced reached
     * residuals of 7e-9 in 119 s (MOSEK too solves the dual). A test, off (BRISK_CONEDUALFREE=1: on): the micro bench of 15
     * SOCPs it moves to the dual form (bench/mittelmann_socp_1102/dualfree/ab.txt) lost on the class it was meant for
     * (db-shear-wall 62 -> 170 iterations, db-plane-strain-prism 37 -> 75, joint_FC_12 OPTIMAL -> reduced accuracy,
     * socp_rocket_4000 OPTIMAL in 30 -> reduced accuracy after 400) and db-joint-soerensen still at the time limit; it gained
     * on the SOC optimal power flows (pegase 2869: 143 -> 65 iterations) */
    /* (15 November 2026, the SOCP developer; the default since that day, BRISK_CONEDUALFREE=0: the earlier rule) the dual form with free
     * variables chosen by its estimated factor and its start's residuals on the equilibrated problem (mode 4, below): the 239
     * development SOCPs no status changed, the 12 files where it switches 48.2 -> 26.8 s, the others unchanged; a held-out set of 33
     * (power flows and nql SOCPs with row-rescaled and permuted copies) 161.8 -> 88.9 s, no status lost */
    static int dfr = -1; if (dfr < 0) { const char *e_ = getenv("BRISK_CONEDUALFREE"); dfr = e_ ? atoi(e_) : 4; }
    const int use = md >= 1 && md < m && (mode == 1 || (m >= 1000 && (nro <= dmf || (dfr && nro <= nf))));
    if (!use) { free(slk); return g_dual_only ? SOCP_DUAL_UNUSED : socp_solve_seq_empty(P, opt, R); }
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
            /* (7 November 2026, the SOCP developer; the owner's SOCP benchmark of 6 November; the default, BRISK_CONEDUALNE=0: off) the
             * estimate above is the full system's; a form with at most NE_MAXFREE free variables also has the normal equations,
             * whose factor holds at most r (r + 1) / 2 entries over its r rows, and the route takes the cheaper of the two
             * (ne_setup, by work). An array design SOCP (P01_03: 2,956 cones of three over two dense rows each, 1,480 weights):
             * the dual's full system estimated at 1.31e7 entries against the primal's 8.78e6, while the dual's normal equations
             * of 1,480 rows hold 1.1e6; the primal was solved, its free weights substituted onto the penalty cone, 1.2e11 flops a
             * factorization (MOSEK: 1.4e10) */
            { static int dne = -1; if (dne < 0) { const char *e_ = getenv("BRISK_CONEDUALNE"); dne = !(e_ && atoi(e_) == 0); }
              if (dne) {
                  int nren_ = 0, nfree_ = pass ? nfr2 : nf;
                  if (!pass && ren) for (int j = nf; j < n; j++) if (ren[j] >= 0) { nren_++; nfree_--; }
                  const double r_ = (double)(mm - nren_);
                  if (nfree_ <= ne_maxfree() && r_ > 0) est = fmin(est, 0.5 * r_ * (r_ + 1.0));
              } }
            if (pass) est_d = est; else est_p = est;
        }
        free(mk); free(gone);
    }
    free(ren); free(renrow);
    if (getenv("BRISK_CONEDBG")) socp_printf("  [the dual form: factor estimated %.3g against the primal's %.3g; nnz %d against %d; rows %d against %d, free %d against %d]\n", est_d, est_p, Ap2[n2], nz, nrow2, m, nfr2, nf);
    /* (the factor's size decides beyond the rows: FIR filter design has 152 rows in the dual against 42,301, but its
     * cones of three each meet most of them, and the dual's factor had 6.4e6 entries against 4.5e6: 6.1 -> 7.4 s) */
    if (getenv("BRISK_CONEDUALSCAN")) { printf("dualscan %s est_d %.4g est_p %.4g rows %d %d\n", est_d > 1.2 * est_p ? "primal" : "dual", est_d, est_p, nrow2, m); fflush(stdout); exit(0); }   /* (experiments: the choice alone) */
    /* (8 November 2026, the SOCP developer, a test: BRISK_CONEFORMCOST=1) when the two estimates are within a factor 2, the form is chosen by
     * the work of the system each will factor: the default substitution of its free variables, then the minimum-degree count of the
     * cheaper of its full system and its normal equations (elim_cost, the substitution's own measure). The estimates count each column's
     * own fill, not the cascade of a sparse row structure: heterogeneous cones (A05_10, SOCP) estimated at 3.51e5 entries (dual) against
     * 4.2e5 (primal), factored at 9.9e7 against 7.3e7; the dual's normal equations 6.6e11 flops, the primal's full system 5.7e11, and the
     * primal 8 iterations against 12 (37.5 against 53.6 s); robust inventory, distributionally robust and robust dose the other way */
    int form_by_cost = -1;
    { static int fcost = -1; if (fcost < 0) { const char *e_ = getenv("BRISK_CONEFORMCOST"); fcost = e_ ? atoi(e_) : 0; }
      if (fcost && mode == 2 && est_d <= 2.0 * est_p && est_p <= 2.0 * est_d) {
          const double tfc_ = sc_time();
          SocpProb Qc = *P; Qc.m = nrow2; Qc.n = n2; Qc.Ap = Ap2; Qc.Ai = Ai2; Qc.Ax = Ax2; Qc.b = b2; Qc.c = c2; Qc.nf = nfr2;
          double cst[2] = { 1e300, 1e300 };
          for (int f_ = 0; f_ < 2; f_++) {
              const SocpProb *Pf = f_ ? &Qc : P; Elim Ef; memset(&Ef, 0, sizeof Ef);
              const int ge_ = g_elim_wide; g_elim_wide = 0; elim_run(&Ef, Pf); g_elim_wide = ge_;
              double fp_ = 0, fl_ = 0; cst[f_] = elim_cost(&Ef, Pf, &fp_, &fl_); elim_free(&Ef);
          }
          form_by_cost = cst[1] <= cst[0];
          if (getenv("BRISK_CONEDBG")) socp_printf("  [the forms by their systems' work: the primal %.3g, the dual %.3g pairs: the %s (%.2fs)]\n", cst[0], cst[1], form_by_cost ? "dual" : "primal", sc_time() - tfc_);
      } }
    /* (14 November 2026, the SOCP developer, a test: BRISK_CONEDUALFREE=2) a dual with more free variables than the cap of 8 (and no
     * more than the primal's) taken only when its estimated factor is at most 0.8 of the primal's: free variables cost the factor's
     * accuracy (the double factor fails in their directions late in a solve and the rest runs in double-double), so the other form
     * must save clearly. nql30-180 (SOCP, DIMACS) 0.73: 12.8 -> 4.4 s on nql180; the optimal power flows 0.49-0.50; the limit-analysis
     * and friction-contact SOCPs, where the dual lost (joint_HO_01 48 -> 71 iterations), 0.94-0.98 */
    const int dual_free_case = mode == 2 && nro > dmf;
    /* (14 November 2026, the SOCP developer, a test: BRISK_CONEDUALFREE=3) as 2, and the dual's start must also be much nearer to
     * feasibility than the primal's: both forms solved to their starts (no iteration), the larger of the primal and dual residuals
     * compared; the dual kept when its start's is at most a tenth of the primal's (bench/socp_1113/dualfree_start.txt: the nql files
     * 1e13 apart, the power flows 25-1,000, the masonry, branch-flow and opf_50k SOCPs, where the dual lost, within 1.3) */
    int start_refused = 0;
    if ((dfr == 3 || dfr == 4) && dual_free_case && est_d <= 0.8 * est_p) {
        SocpOpts oz; if (opt) oz = *opt; else socp_default_opts(&oz);
        oz.maxit = 0; oz.verbose = 0; oz.noretry = 1;
        SocpProb Qz = *P; Qz.m = nrow2; Qz.n = n2; Qz.Ap = Ap2; Qz.Ai = Ai2; Qz.Ax = Ax2; Qz.b = b2; Qz.c = c2; Qz.nf = nfr2;
        SocpRes Rp, Rd; memset(&Rp, 0, sizeof Rp); memset(&Rd, 0, sizeof Rd);
        const int gdo = g_dual_only; g_dual_only = 0;
        socp_solve_seq_empty(P, &oz, &Rp); socp_solve_seq_empty(&Qz, &oz, &Rd);
        g_dual_only = gdo;
        /* (15 November 2026, BRISK_CONEDUALFREE=4) the starts' residuals compared on the equilibrated problems (start_err), which a
         * rescaling of the rows does not change; =3 compares them on the problems as given, and a row-rescaled copy of a power flow
         * (socopf_3012wp_k) then took the other form and lost its status (bench/socp_1115/heldout_df3.txt) */
        const double ep = dfr == 4 ? fmax(Rp.start_err[0], Rp.start_err[1]) : fmax(Rp.err[0], Rp.err[1]);
        const double ed = dfr == 4 ? fmax(Rd.start_err[0], Rd.start_err[1]) : fmax(Rd.err[0], Rd.err[1]);
        free(Rp.x); free(Rp.y); free(Rp.z); free(Rp.cx); free(Rp.cy); free(Rp.cz); free(Rd.x); free(Rd.y); free(Rd.z); free(Rd.cx); free(Rd.cy); free(Rd.cz);
        start_refused = !(ed <= 0.1 * ep);
        if (getenv("BRISK_CONEDBG")) socp_printf("  [the forms' starts: the primal's residuals %.2e, the dual's %.2e: the %s]\n", ep, ed, start_refused ? "primal" : "dual");
    }
    if (((dfr == 2 || dfr == 3 || dfr == 4) && dual_free_case && est_d > 0.8 * est_p) || start_refused) {
        if (getenv("BRISK_CONEDBG")) socp_printf("  [the dual form refused: %d free variables and a factor estimated at %.2f of the primal's]\n", nro, est_d / fmax(est_p, 1e-300));
        free(slk); free(srow); free(prow); free(rs_); free(rfree); free(rp); free(rc); free(rv); free(Ap2); free(Ai2); free(Ax2); free(b2); free(c2);
        return g_dual_only ? SOCP_DUAL_UNUSED : socp_solve_seq_empty(P, opt, R);
    }
    if (form_by_cost == 0 || (form_by_cost < 0 && mode == 2 && est_d > 1.2 * est_p)) {
        free(slk); free(srow); free(prow); free(rs_); free(rfree); free(rp); free(rc); free(rv); free(Ap2); free(Ai2); free(Ax2); free(b2); free(c2);
        return g_dual_only ? SOCP_DUAL_UNUSED : socp_solve_seq_empty(P, opt, R);
    }
    g_dual_only = 0;      /* (the dual is solved: nothing below asks again) */
    SocpProb Q = *P;
    Q.m = nrow2; Q.n = n2; Q.Ap = Ap2; Q.Ai = Ai2; Q.Ax = Ax2; Q.b = b2; Q.c = c2; Q.nf = nfr2;
    SocpOpts o2; if (opt) o2 = *opt; else socp_default_opts(&o2);
    if (o2.verbose > 0) socp_printf("the dual solved: %d rows instead of %d (%d slack blocks; %d free variables)\n", nrow2, m, (int)(n - nf - (nrow2 - nf)), nfr2);
    SNOTED("the dual solved by the SOCP solver: %d rows instead of %d", nrow2, m);
    const char *pfx = g_stat_prefix; g_stat_prefix = "the dual: ";
    SocpRes R2; memset(&R2, 0, sizeof R2);
    /* (the default log's rows: the original b'y = K - (the dual's c2'x2) and c'x = K - (the dual's b2'y2), K = sum over the
     * slack rows of b_i c_s / a_is: the two objectives exchanged and negated about K) */
    double Kd = 0.0; for (int i = 0; i < m; i++) if (rfree[i] < 0) { const int s_ = rs_[i]; Kd += P->b[i] * P->c[s_] / P->Ax[P->Ap[s_]]; }
    const double ra0 = g_row_a, rk0 = g_row_k; const int rs0 = g_row_swap;
    g_row_k = rk0 + ra0 * Kd; g_row_a = -ra0; g_row_swap = !rs0;
    const int st2 = socp_solve_seq_empty(&Q, &o2, &R2);
    g_row_a = ra0; g_row_k = rk0; g_row_swap = rs0;
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
        const int it = R->iters; const double t1 = R->time;
        SocpOpts o2 = *opt; o2.wx = o2.wy = o2.wz = NULL;
        if (o2.timelimit > 0) o2.timelimit = fmax(o2.timelimit - t1, 1e-3);      /* (one time limit for both attempts) */
        socp_result_free(R);
        st = socp_solve_core(P, &o2, R);
        R->iters += it; R->time += t1;
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
    SNOTE("presolve: the problem folded by its symmetry: %d rows -> %d, %d variables -> %d", m, nr, n, nc); if (opt && opt->verbose > 0) socp_user_presolve("the problem folded by its symmetry: %d rows -> %d, %d variables -> %d", m, nr, n, nc);
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
/* (31 October 2026; the owner's SOCP benchmark audit) the data checked before anything else: an entry of A, b or c that is NaN
 * or infinite, a column pointer that decreases or a row index out of range is an input error (SOCP_INPUT, with a message),
 * where a NaN cost had run to numerical difficulties with a NaN objective */
static int socp_input_ok(const SocpProb *P) {
    if (!P || P->m < 0 || P->n < 0 || (P->n > 0 && (!P->Ap || !P->c)) || (P->m > 0 && !P->b)) return 0;
    if (P->Ap && P->Ap[0] != 0) return 0;
    for (int j = 0; j < P->n; j++) { if (P->Ap[j + 1] < P->Ap[j]) return 0; if (!isfinite(P->c[j])) return 0; }
    for (int i = 0; i < P->m; i++) if (!isfinite(P->b[i])) return 0;
    const int nz = P->n > 0 ? P->Ap[P->n] : 0;
    for (int p = 0; p < nz; p++) if (!isfinite(P->Ax[p]) || P->Ai[p] < 0 || P->Ai[p] >= P->m) return 0;
    long nd = (long)P->nf + P->nl; if (P->nf < 0 || P->nl < 0) return 0;
    for (int k = 0; k < P->nq; k++) { if (P->q[k] < 1) return 0; nd += P->q[k]; }
    for (int k = 0; k < P->nr; k++) { if (P->r[k] < 2) return 0; nd += P->r[k]; }
    return nd == P->n;
}
/* (10 November 2026, the SOCP developer; bench/socp_1110/BOUNDS.md, the owner's request) the count of the rows that are bounds or
 * inequalities written in SeDuMi's form, on the problem as the SOCP solver receives it (BRISK_CONEBOUNDCOUNT=1: one line, then the
 * solve goes on; =2: the line and no solve). A slack is a nonnegative variable with one entry in A. A row is
 *   a bound row: two entries, one of them a slack, the other a variable that appears in other rows too (x_j + t = u, x_j - t = l);
 *   a ranged row's box row: two entries, a slack and a nonnegative variable whose only other entry is in an inequality row (s + t = r);
 *   an inequality row: a slack among three entries or more (a'x + s = b);
 *   an equation: the rest.
 * For the bound rows: whether the bounded variable sits in a cone (then its rows include the cone's) and the largest row count of the
 * bounded variables' columns (the fill a bound row brings to the normal equations). */
static int socp_bound_count_at(const SocpProb *P, int where) {
    static int on = -1; if (on < 0) { const char *e_ = getenv("BRISK_CONEBOUNDCOUNT"); on = e_ ? atoi(e_) : 0; }
    if (!on || (on >= 3) != where) return 0;      /* (1, 2: at the entry, the problem as received; 3, 4: at the interior-point method's start, after the presolve) */
    const int m = P->m, n = P->n, nf = P->nf, nl = P->nl;
    int *rc = (int *)calloc((size_t)m + 1, sizeof(int)), *slk = (int *)malloc(sizeof(int) * ((size_t)m + 1)), *oth = (int *)malloc(sizeof(int) * ((size_t)m + 1));
    for (int i = 0; i < m; i++) { slk[i] = -1; oth[i] = -1; }
    for (int j = 0; j < n; j++) for (int p = P->Ap[j]; p < P->Ap[j + 1]; p++) rc[P->Ai[p]]++;
    for (int j = nf; j < nf + nl; j++) if (P->Ap[j + 1] - P->Ap[j] == 1) { const int i = P->Ai[P->Ap[j]]; if (slk[i] < 0) slk[i] = j; }
    for (int j = 0; j < n; j++) for (int p = P->Ap[j]; p < P->Ap[j + 1]; p++) { const int i = P->Ai[p]; if (rc[i] == 2 && slk[i] >= 0 && j != slk[i]) oth[i] = j; }
    long nbound = 0, nbcone = 0, nrange = 0, nineq = 0, neq = 0, maxcol = 0;
    for (int i = 0; i < m; i++) {
        if (slk[i] >= 0 && rc[i] == 2 && oth[i] >= 0) {
            const int j = oth[i], cj = P->Ap[j + 1] - P->Ap[j];
            int ranged = 0;
            if (j >= nf && j < nf + nl && cj == 2) {      /* (s + t = r with s's other entry in an inequality row) */
                for (int p = P->Ap[j]; p < P->Ap[j + 1]; p++) { const int i2 = P->Ai[p]; if (i2 != i && rc[i2] >= 3) ranged = 1; }
            }
            if (ranged) nrange++;
            else { nbound++; if (j >= nf + nl) nbcone++; if (cj > maxcol) maxcol = cj; }
        } else if (slk[i] >= 0 && rc[i] >= 3) nineq++;
        else neq++;
    }
    fprintf(stderr, "[bound count%s: rows %d, variables %d (free %d, nonnegative %d, in cones %d); bound rows %ld (%.1f %%; on a cone's variable %ld; "
                "the bounded variables' columns up to %ld entries), ranged rows' box rows %ld, inequality rows %ld, equations %ld]\n",
                where ? " after the presolve" : "", m, n, nf, nl, n - nf - nl, nbound, m ? 100.0 * nbound / m : 0.0, nbcone, maxcol, nrange, nineq, neq);
    free(rc); free(slk); free(oth);
    return on == 2 || on == 4;
}
/* (10 November 2026, the SOCP developer; plan item 3, a count: BRISK_CONEFRCOUNT=1 prints, 2 prints and stops) the faces that single rows
 * prove, chained: a row with b_i = 0, no free variable in it, and every part of one sign in the dual cone (a nonnegative variable's
 * coefficient, a cone's part a_k in the cone or its negative; the second-order and rotated cones are self-dual) proves that every
 * feasible x is orthogonal to its parts: the nonnegative variables with a coefficient are zero, a cone whose part is inside the cone is
 * zero, a cone whose part is on the boundary lies on one ray (x_k = lambda r_k, r_k = (s_0, -sbar) for a second-order cone, (s_v, s_u,
 * -sbar) for a rotated one), and later rows see that cone as one nonnegative variable with the coefficient a_k'r_k. The rows are swept
 * until none is found; a row b_i != 0 left with no variable proves infeasibility. Rounds: the chain's depth. */
static int socp_fr_count(const SocpProb *P) {
    static int on = -1; if (on < 0) { const char *e_ = getenv("BRISK_CONEFRCOUNT"); on = e_ ? atoi(e_) : 0; }
    if (!on) return 0;
    const int m = P->m, n = P->n, nf = P->nf, nl = P->nl, nq = P->nq, nr = P->nr, nk = nq + nr;
    int *rp = (int *)calloc((size_t)m + 2, sizeof(int)); for (int p = 0; p < P->Ap[n]; p++) rp[P->Ai[p] + 1]++;
    for (int i = 0; i < m; i++) rp[i + 1] += rp[i];
    int *rj = (int *)malloc(sizeof(int) * ((size_t)P->Ap[n] + 1)); double *rx = (double *)malloc(sizeof(double) * ((size_t)P->Ap[n] + 1));
    { int *w = (int *)malloc(sizeof(int) * ((size_t)m + 1)); memcpy(w, rp, sizeof(int) * (size_t)m);
      for (int j = 0; j < n; j++) for (int p = P->Ap[j]; p < P->Ap[j + 1]; p++) { const int k = w[P->Ai[p]]++; rj[k] = j; rx[k] = P->Ax[p]; } free(w); }
    int *cone = (int *)malloc(sizeof(int) * ((size_t)n + 1)), *ks = (int *)malloc(sizeof(int) * ((size_t)nk + 1)), *kd = (int *)malloc(sizeof(int) * ((size_t)nk + 1));
    for (int j = 0; j < n; j++) cone[j] = -1;
    { int s0 = nf + nl; for (int k = 0; k < nk; k++) { kd[k] = k < nq ? P->q[k] : P->r[k - nq]; ks[k] = s0; for (int t = 0; t < kd[k]; t++) cone[s0 + t] = k; s0 += kd[k]; } }
    unsigned char *zero = (unsigned char *)calloc((size_t)n + 1, 1), *kst = (unsigned char *)calloc((size_t)nk + 1, 1), *rdone = (unsigned char *)calloc((size_t)m + 1, 1);
    double *ray = (double *)calloc((size_t)n + 1, sizeof(double)), *wv = (double *)calloc((size_t)n + 1, sizeof(double));
    int *tk = (int *)malloc(sizeof(int) * ((size_t)nk + 1)), *mark = (int *)malloc(sizeof(int) * ((size_t)nk + 1)); for (int k = 0; k < nk; k++) mark[k] = -1;
    long ncert = 0, nlz = 0, nkz = 0, nkray = 0, ninf = 0, ninterior = 0, nboundary = 0; int rounds = 0, stamp = 0; const double tol = 1e-12;
    for (int changed = 1; changed && rounds < 1000; ) {
        changed = 0; int found = 0;
        for (int i = 0; i < m; i++) {
            if (rdone[i]) continue;
            int sg = 0, ok = 1, ntk = 0, nz = 0; stamp++;
            for (int p = rp[i]; p < rp[i + 1] && ok; p++) {
                const int j = rj[p]; const double a = rx[p]; if (zero[j] || a == 0.0) continue;
                nz++;
                if (j < nf) { ok = 0; break; }
                if (j < nf + nl) { const int s_ = a > 0 ? 1 : -1; if (sg && s_ != sg) ok = 0; sg = s_; continue; }
                const int k = cone[j]; if (mark[k] != stamp) { mark[k] = stamp; tk[ntk++] = k; for (int t = 0; t < kd[k]; t++) wv[ks[k] + t] = 0.0; }
                wv[j] += a;
            }
            if (nz == 0) { if (P->b[i] != 0.0) { ninf++; } rdone[i] = 1; continue; }
            if (!ok || P->b[i] != 0.0) continue;
            int kind[1]; (void)kind;
            for (int u = 0; u < ntk && ok; u++) {
                const int k = tk[u], j0 = ks[k], d = kd[k];
                if (kst[k] == 1) { double lam = 0; for (int t = 0; t < d; t++) lam += wv[j0 + t] * ray[j0 + t]; if (lam == 0.0) continue; const int s_ = lam > 0 ? 1 : -1; if (sg && s_ != sg) ok = 0; sg = s_; continue; }
                double s0, s1 = 0, t2 = 0;
                if (k < nq) { s0 = wv[j0]; for (int t = 1; t < d; t++) t2 += wv[j0 + t] * wv[j0 + t];
                    const int s_ = s0 > 0 ? 1 : s0 < 0 ? -1 : 0; if (!s_) { if (t2 > 0) ok = 0; continue; }
                    if (sg && s_ != sg) { ok = 0; continue; } sg = s_;
                    if (s0 * s0 < t2 * (1.0 - tol)) ok = 0;
                } else { s0 = wv[j0]; s1 = wv[j0 + 1]; for (int t = 2; t < d; t++) t2 += wv[j0 + t] * wv[j0 + t];
                    const int s_ = (s0 > 0 || s1 > 0) ? 1 : (s0 < 0 || s1 < 0) ? -1 : 0; if (!s_) { if (t2 > 0) ok = 0; continue; }
                    if (s0 * s_ < 0 || s1 * s_ < 0) { ok = 0; continue; }
                    if (sg && s_ != sg) { ok = 0; continue; } sg = s_;
                    if (2.0 * s0 * s1 < t2 * (1.0 - tol)) ok = 0;
                }
            }
            if (!ok || !sg) continue;
            /* a certificate: apply it */
            ncert++; found = 1; rdone[i] = 1;
            for (int p = rp[i]; p < rp[i + 1]; p++) { const int j = rj[p]; if (zero[j] || rx[p] == 0.0) continue; if (j >= nf && j < nf + nl) { zero[j] = 1; nlz++; } }
            for (int u = 0; u < ntk; u++) {
                const int k = tk[u], j0 = ks[k], d = kd[k]; double nn = 0; for (int t = 0; t < d; t++) nn += wv[j0 + t] * wv[j0 + t]; if (nn == 0.0) continue;
                if (kst[k] == 1) { for (int t = 0; t < d; t++) zero[j0 + t] = 1; kst[k] = 2; nkz++; continue; }
                double s0 = sg * wv[j0], s1 = k < nq ? 0 : sg * wv[j0 + 1], t2 = 0; for (int t = (k < nq ? 1 : 2); t < d; t++) t2 += wv[j0 + t] * wv[j0 + t];
                const int inter = k < nq ? (s0 * s0 > t2 * (1.0 + tol)) : (s0 > 0 && s1 > 0 && 2.0 * s0 * s1 > t2 * (1.0 + tol));
                if (inter) { for (int t = 0; t < d; t++) zero[j0 + t] = 1; kst[k] = 2; nkz++; ninterior++; }
                else { kst[k] = 1; nkray++; nboundary++;
                    if (k < nq) { ray[j0] = s0; for (int t = 1; t < d; t++) ray[j0 + t] = -sg * wv[j0 + t]; }
                    else { ray[j0] = s1; ray[j0 + 1] = s0; for (int t = 2; t < d; t++) ray[j0 + t] = -sg * wv[j0 + t]; } }
            }
        }
        if (found) { changed = 1; rounds++; }
    }
    fprintf(stderr, "[facial reduction count: rows %d, variables %d (free %d, nonnegative %d, cones %d + rotated %d); single-row certificates %ld in %d rounds: "
                    "nonnegative variables to zero %ld, cones to zero %ld (inside %ld), cones to a ray %ld (later zero %ld); rows that prove infeasibility %ld]\n",
            m, n, nf, nl, nq, nr, ncert, rounds, nlz, nkz, ninterior, nkray, nkz - ninterior, ninf);
    free(rp); free(rj); free(rx); free(cone); free(ks); free(kd); free(zero); free(kst); free(rdone); free(ray); free(wv); free(tk); free(mark);
    return on == 2;
}
static int socp_bound_count(const SocpProb *P) { return socp_bound_count_at(P, 0); }
int socp_solve(const SocpProb *P, const SocpOpts *opt, SocpRes *R) {
    if (socp_input_ok(P) && socp_bound_count(P)) { if (R) { memset(R, 0, sizeof *R); R->status = SOCP_INPUT; } return SOCP_INPUT; }
    if (!socp_input_ok(P)) {
        if (!opt || opt->verbose >= 0) SUSER("the SOCP solver: invalid input (an entry of A, b or c not finite, a column pointer or row index out of range, or cone sizes that do not add up to the columns)\n");
        if (R) { memset(R, 0, sizeof *R); R->status = SOCP_INPUT; }
        return SOCP_INPUT;
    }
    { SocpOpts d_; if (!opt) socp_default_opts(&d_); g_cone_lowtol = (opt ? opt->tol : d_.tol) >= 5e-7; }     /* (cone_pkg: -acc low) */
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
/* (31 October 2026, the SOCP developer; the owner's SOCP benchmark of 31 October; BRISK_CONEDUALROT=0: off) rotated cones
 * written as second-order cones for the dual form. The dual form needs each block's columns to be slacks of their own rows
 * (dual_shape); a rotated cone 2 x0 x1 >= |x2:|^2 whose two first columns mix two rows (a model that changed the coordinates
 * x0 = (t + u) / sqrt 2, x1 = (t - u) / sqrt 2 of a cone t >= |(u, x2:)|) fails that test, while the same cone in the coordinates
 * t, u passes it: the benchmark's 60 rotated copies of its problems never had the dual form, against every Lorentz copy of 14
 * of its 16 families (mpc P115, SOCP: 8 iterations and 27.7 s in the Lorentz form, 14 and 58.1 s rotated). The map
 * (t, u) = T (x0, x1), T = [1 1; 1 -1] / sqrt 2, is orthogonal and its own inverse; the columns become A T, the costs T c,
 * the dual slack z' = T z, the rows and y unchanged. A cone is mapped only when the map leaves each of its two first columns
 * a single entry (the other entries exactly zero); the mapped cones become second-order cones (after the problem's own, in
 * their order), and the dual form is then judged as usual: when it is not taken, the problem is solved as given, rotated
 * cones and their own rules (the boost, the fixed side) included */
static int socp_solve_dualform_rot(const SocpProb *P, const SocpOpts *opt, SocpRes *R) {
    static int on = -1; if (on < 0) { const char *e_ = getenv("BRISK_CONEDUALROT"); on = !(e_ && atoi(e_) == 0); }
    const char *e = getenv("BRISK_CONEDUAL");
    if (!on || P->nr == 0 || (e && atoi(e) == 0)) return socp_solve_dualform(P, opt, R);
    const int n = P->n, nf = P->nf, nl = P->nl, nq = P->nq, nr = P->nr;
    int s0 = nf + nl; for (int k = 0; k < nq; k++) s0 += P->q[k];
    unsigned char *mp = (unsigned char *)calloc((size_t)nr + 1, 1); int nmap = 0;
    const double h = sqrt(0.5);
    for (int k = 0, s = s0; k < nr; s += P->r[k], k++) {
        if (P->r[k] < 2 || P->Ap[s + 1] - P->Ap[s] != 2 || P->Ap[s + 2] - P->Ap[s + 1] != 2) continue;
        const int p0 = P->Ap[s], p1 = P->Ap[s + 1];
        if (P->Ai[p0] != P->Ai[p1] || P->Ai[p0 + 1] != P->Ai[p1 + 1]) continue;     /* (the same two rows, ascending) */
        const double a0 = P->Ax[p0], a1 = P->Ax[p0 + 1], b0 = P->Ax[p1], b1 = P->Ax[p1 + 1];
        const double u0 = h * (a0 + b0), u1 = h * (a1 + b1), v0 = h * (a0 - b0), v1 = h * (a1 - b1);
        if (((u0 != 0.0) + (u1 != 0.0)) == 1 && ((v0 != 0.0) + (v1 != 0.0)) == 1 && (u0 != 0.0) != (v0 != 0.0)) { mp[k] = 1; nmap++; }
    }
    if (dbg_on()) socp_printf("  [rotated cones that become slack blocks as second-order cones: %d of %d]\n", nmap, nr);
    if (nmap == 0) { free(mp); return socp_solve_dualform(P, opt, R); }
    /* the new column order: free, linear, the second-order cones, the mapped cones, the other rotated cones; old[jn] = jo */
    int *old = (int *)malloc(sizeof(int) * ((size_t)n + 1)), *qn = (int *)malloc(sizeof(int) * ((size_t)(nq + nmap) + 1)), *rn = (int *)malloc(sizeof(int) * ((size_t)(nr - nmap) + 1));
    int jn = 0; for (int j = 0; j < s0; j++) old[jn++] = j;
    for (int k = 0; k < nq; k++) qn[k] = P->q[k];
    int kq = nq, kr = 0;
    for (int pass = 0; pass < 2; pass++)
        for (int k = 0, s = s0; k < nr; s += P->r[k], k++) if ((pass == 0) == (mp[k] != 0)) {
            for (int t = 0; t < P->r[k]; t++) old[jn++] = s + t;
            if (pass == 0) qn[kq++] = P->r[k]; else rn[kr++] = P->r[k];
        }
    unsigned char *head = (unsigned char *)calloc((size_t)n + 1, 1);     /* (head[jn] = 1: the first column of a mapped pair) */
    for (int k = 0, s = s0; k < nr; s += P->r[k], k++) if (mp[k]) head[s] = 1;
    int *Ap = (int *)malloc(sizeof(int) * ((size_t)n + 1)), *Ai = (int *)malloc(sizeof(int) * ((size_t)P->Ap[n] + 1));
    double *Ax = (double *)malloc(sizeof(double) * ((size_t)P->Ap[n] + 1)), *c = (double *)malloc(sizeof(double) * ((size_t)n + 1));
    int nz = 0; Ap[0] = 0;
    for (int j = 0; j < n; j++) {
        const int jo = old[j];
        if (head[jo] || (jo > 0 && head[jo - 1] && j > 0 && old[j - 1] == jo - 1)) {     /* (a mapped pair: t = (x0 + x1) / sqrt 2, u = (x0 - x1) / sqrt 2) */
            const int j0 = head[jo] ? jo : jo - 1, sg = head[jo] ? 1 : -1;
            const int p0 = P->Ap[j0], p1 = P->Ap[j0 + 1];
            for (int t = 0; t < 2; t++) { const double v = h * (P->Ax[p0 + t] + sg * P->Ax[p1 + t]); if (v != 0.0) { Ai[nz] = P->Ai[p0 + t]; Ax[nz++] = v; } }
            c[j] = h * (P->c[j0] + sg * P->c[j0 + 1]);
        } else {
            for (int p = P->Ap[jo]; p < P->Ap[jo + 1]; p++) { Ai[nz] = P->Ai[p]; Ax[nz++] = P->Ax[p]; }
            c[j] = P->c[jo];
        }
        Ap[j + 1] = nz;
    }
    SocpProb Q = *P; Q.Ap = Ap; Q.Ai = Ai; Q.Ax = Ax; Q.c = c; Q.nq = nq + nmap; Q.q = qn; Q.nr = nr - nmap; Q.r = rn;
    SocpRes R2; memset(&R2, 0, sizeof R2);
    g_dual_only = 1;
    int st = socp_solve_dualform(&Q, opt, &R2);
    g_dual_only = 0;
    if (st == SOCP_DUAL_UNUSED) st = socp_solve_dualform(P, opt, R);       /* (the dual form not taken: the problem as given) */
    else {
        if (opt && opt->verbose > 0) socp_printf("  (%d rotated cones written as second-order cones for the dual form)\n", nmap);
        *R = R2;
        if (R2.x && R2.z) {         /* back to the given columns and coordinates: x = T x', z = T z' (T its own inverse) */
            double *x = (double *)calloc((size_t)n + 1, sizeof(double)), *z = (double *)calloc((size_t)n + 1, sizeof(double));
            for (int j = 0; j < n; j++) {
                const int jo = old[j];
                if (head[jo]) { x[jo] = h * (R2.x[j] + R2.x[j + 1]); x[jo + 1] = h * (R2.x[j] - R2.x[j + 1]); z[jo] = h * (R2.z[j] + R2.z[j + 1]); z[jo + 1] = h * (R2.z[j] - R2.z[j + 1]); j++; }
                else { x[jo] = R2.x[j]; z[jo] = R2.z[j]; }
            }
            free(R2.x); free(R2.z); R->x = x; R->z = z;
        }
    }
    free(mp); free(old); free(qn); free(rn); free(head); free(Ap); free(Ai); free(Ax); free(c);
    return st;
}
static int socp_fold_then_dual(const SocpProb *P, const SocpOpts *opt, SocpRes *R) {     /* (the folding by symmetry, then the choice of the dual form) */
    int folded = 0; const int st = socp_fold_try(P, opt, R, socp_solve_dualform_rot, &folded);
    return folded ? st : socp_solve_dualform_rot(P, opt, R);
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
    const double capx_ = getenv("BRISK_DEPCAPX") ? atof(getenv("BRISK_DEPCAPX")) : 1.0;     /* (2 November 2026, a test: the caps times this factor) */
    const double capw = capx_ * ((double)nz + 2e6); const long caps = (long)(capx_ * (double)(nz + 1000000));     /* (the work counts the rows' loads, the eliminations and the stored entries: at most the entries of A and 2e6 more, about 5 ms beyond a pass over A; with 1e7 more, a control SOCP of 238,000 entries spent 30 ms finding nothing) */
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
    if (nd > 0 && opt && opt->verbose > 0) socp_user_presolve("%d rows that are multiples of another row removed (rows %d -> %d)", nd, m, m - nd);
    if (ndep > 0) SNOTE("presolve: %d linearly dependent rows removed (their dual values 0)", ndep);
    if (ndep > 0 && opt && opt->verbose > 0) socp_user_presolve("%d linearly dependent rows removed", ndep);
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
    /* (28 October 2026) the count of the start and the rule during the solve (brisk_team_begin, brisk_team_adapt each iteration) */
    const int ntt = omp_in_parallel() ? 1 : brisk_team_begin(ntb > 0 ? ntb : nt);     /* (the count asked for: the rule may raise it again when cores free up) */
    g_team_nt0 = ntt;
    { const char *e = getenv("BRISK_CONEPANEL"); if (!(e && atoi(e) == 0) && !omp_in_parallel()) schol_set_panel_threads(ntt); }
    g_cone_nt = ntt; g_cone_nt_ldl = ntt;
    schol_set_cone_tiles(1);        /* (the dense panels by fixed tiles on the panel threads: the same arithmetic at any thread count) */
    int st;
    /* (31 October 2026; A5) an allocation that fails on this thread returns here (schol_oom) instead of ending the process:
     * the solve's state is reset below as on any return, the result empty, the status numerical difficulties */
    jmp_buf oomj; const int own_oom = schol_oom_jmp_get() == NULL;
    if (own_oom && setjmp(oomj)) {
        schol_set_oom_jmp(NULL);
        SUSER("the SOCP solver: out of memory; the solve was stopped (the memory it held is not returned)\n");
        memset(R, 0, sizeof(*R)); R->status = SOCP_NUMERR;
        st = SOCP_NUMERR;
    } else {
        if (own_oom) schol_set_oom_jmp(&oomj);
        if (opt && (opt->wx || opt->wsave > 0)) st = socp_solve_warm(P, opt, R);
        else st = socp_solve_nodup(P, opt, R);      /* (rows that are multiples of others removed, then the folding by symmetry and the dual form) */
        if (own_oom) schol_set_oom_jmp(NULL);
    }
    g_cone_nt = 1; g_cone_nt_ldl = 1;
    if (g_team_nt0 >= 1 && !omp_in_parallel()) brisk_team_end();
    g_team_nt0 = 0;
    schol_set_cone_tiles(0);
    schol_set_panel_threads(0);
    if (nt > 1) omp_set_num_threads(nt);
    if (ntb > 0) omp_set_num_threads(ntb);
    return st;
#else
    schol_set_cone_tiles(1);
    const int st = (opt && (opt->wx || opt->wsave > 0)) ? socp_solve_warm(P, opt, R) : socp_solve_nodup(P, opt, R);
    schol_set_cone_tiles(0);
    return st;
#endif
}
/* (11 November 2026, the SOCP developer; ISSUES 865, a test: BRISK_CONEPURIFY=1) the end step when a solve stops with every error but
 * complementarity within the tolerance: each pair rounded to its face. Near mu 1e-13 the iterate in double holds only a few digits of a
 * cone's distance to its boundary, the steps block, and x'z (mu times about a million blocks on the limit-analysis SOCPs) stays just above
 * the tolerance while the gap and the residuals are far below it. Rounding: a nonnegative pair keeps the larger of x_j, z_j relative to its
 * vector's scale and zeroes the other; a cone pair in its spectral form (x = l1 c1 + l2 c2, c1,2 = (1, +-u)/2) sets x to zero when it is
 * negligible beside z (l1 / |x| below 1e-6 of m1 / |z|), z to zero in the opposite case, and otherwise puts both on the boundary of one frame,
 * x = l1 c1(u), z = m1 c2(u), whose product is zero. The rounded point is measured on the problem as given (the four relative errors and
 * the cones' membership) and kept only when every error is within the tolerance. */
static int socp_purify(const SocpProb *P, const SocpOpts *opt, SocpRes *R, double tol, int mode) {
    const int m = P->m, n = P->n, nf = P->nf, nl = P->nl, nk = P->nq + P->nr;
    if (!R->x || !R->y || !R->z) return 0;
    double *x = (double *)malloc(sizeof(double) * ((size_t)n + 1)), *z = (double *)malloc(sizeof(double) * ((size_t)n + 1));
    memcpy(x, R->x, sizeof(double) * (size_t)n); memcpy(z, R->z, sizeof(double) * (size_t)n);
    double sx = 0, sz = 0; for (int j = nf; j < n; j++) { sx = fmax(sx, fabs(x[j])); sz = fmax(sz, fabs(z[j])); }
    if (!(sx > 0) || !(sz > 0)) { free(x); free(z); return 0; }
    unsigned char *zc = (unsigned char *)calloc((size_t)n + 1, 1); signed char *kf = (signed char *)calloc((size_t)nk + 1, 1);
    double *ray = (double *)calloc((size_t)n + 1, sizeof(double)); int *ks = (int *)malloc(sizeof(int) * ((size_t)nk + 1)), *kd = (int *)malloc(sizeof(int) * ((size_t)nk + 1));
    for (int j = nf; j < nf + nl; j++) { if (x[j] / sx >= z[j] / sz) z[j] = 0.0; else { x[j] = 0.0; zc[j] = 1; } }
    const double rs = sqrt(0.5); int j0 = nf + nl; long nzx = 0, nzz = 0, nface = 0;
    double *u = NULL; int umax = 0;
    for (int k = 0; k < nk; k++) {
        const int d = k < P->nq ? P->q[k] : P->r[k - P->nq], rot = k >= P->nq;
        if (d > umax) { free(u); umax = d; u = (double *)malloc(sizeof(double) * (size_t)umax); }
        double *xa = x + j0, *za = z + j0; ks[k] = j0; kd[k] = d;
        if (rot) { const double a = xa[0], b = xa[1]; xa[0] = rs * (a + b); xa[1] = rs * (a - b); const double c = za[0], e = za[1]; za[0] = rs * (c + e); za[1] = rs * (c - e); }
        double nx = 0, nz = 0; for (int t = 1; t < d; t++) { nx += xa[t] * xa[t]; nz += za[t] * za[t]; } nx = sqrt(nx); nz = sqrt(nz);
        const double l1 = xa[0] + nx, m1 = za[0] + nz;
        if (l1 / sx < 1e-6 * (m1 / sz)) { for (int t = 0; t < d; t++) { xa[t] = 0.0; zc[j0 + t] = 1; } kf[k] = 1; nzx++; }
        else if (m1 / sz < 1e-6 * (l1 / sx)) { for (int t = 0; t < d; t++) za[t] = 0.0; nzz++; }
        else {      /* both on the boundary of x's frame: x = l1 (1, u)/2, z = m1 (1, -u)/2 */
            if (nx > 0) for (int t = 1; t < d; t++) u[t] = xa[t] / nx; else if (nz > 0) for (int t = 1; t < d; t++) u[t] = -za[t] / nz; else for (int t = 1; t < d; t++) u[t] = t == 1;
            xa[0] = 0.5 * l1; za[0] = 0.5 * m1; for (int t = 1; t < d; t++) { xa[t] = 0.5 * l1 * u[t]; za[t] = -0.5 * m1 * u[t]; } nface++;
            kf[k] = 2; { double rn = 0; double *rr = ray + j0; rr[0] = 1.0; for (int t = 1; t < d; t++) rr[t] = u[t];
              if (rot) { const double a = rr[0], b = rr[1]; rr[0] = rs * (a + b); rr[1] = rs * (a - b); }
              for (int t = 0; t < d; t++) rn += rr[t] * rr[t]; rn = sqrt(rn); for (int t = 0; t < d; t++) { rr[t] /= rn; zc[j0 + t] = 1; } }
        }
        if (rot) { const double a = xa[0], b = xa[1]; xa[0] = rs * (a + b); xa[1] = rs * (a - b); const double c = za[0], e = za[1]; za[0] = rs * (c + e); za[1] = rs * (c - e); }
        j0 += d;
    }
    free(u);
    /* the measure on the problem as given */
    double *ra = (double *)malloc(sizeof(double) * ((size_t)m + 1)); double nb = 0, nc = 0, po = 0, dob = 0, rp = 0, rd = 0, xz = 0;
    for (int i = 0; i < m; i++) { ra[i] = -P->b[i]; nb += P->b[i] * P->b[i]; dob += P->b[i] * R->y[i]; }
    for (int j = 0; j < n; j++) { double aty = 0; for (int p = P->Ap[j]; p < P->Ap[j + 1]; p++) { ra[P->Ai[p]] += P->Ax[p] * x[j]; aty += P->Ax[p] * R->y[P->Ai[p]]; }
        const double r = P->c[j] - aty - (j < nf ? 0.0 : z[j]); rd += r * r; po += P->c[j] * x[j]; nc += P->c[j] * P->c[j]; if (j >= nf) xz += x[j] * z[j]; }
    for (int i = 0; i < m; i++) rp += ra[i] * ra[i];
    free(ra);
    const double den = 1.0 + fabs(po) + fabs(dob), e0 = sqrt(rp) / (1.0 + sqrt(nb)), e1 = sqrt(rd) / (1.0 + sqrt(nc)), e2 = (po - dob) / den, e3 = xz / den;
    const double e = fmax(fmax(e0, e1), fmax(fabs(e2), fabs(e3)));
    if (getenv("BRISK_CONEPURIFYDBG")) socp_printf("  [the end step: %ld cones x to zero, %ld z to zero, %ld on one frame; errors %.2e %.2e %.2e %.2e (before %.2e %.2e %.2e %.2e)]\n",
        nzx, nzz, nface, e0, e1, e2, e3, R->err[0], R->err[1], R->err[2], R->err[3]);
    if (!(e <= tol) && mode >= 2 && opt) {
        /* (mode 2) the rounding moved the point too far: the face it chose (the zeroed variables and cones, the cones on one ray) solved
         * as a problem of its own, all rows kept, and mapped back; kept when the measure on the problem as given passes */
        unsigned char *rdel = (unsigned char *)calloc((size_t)m + 1, 1); FrFace F; fr_face_build(P, ks, kd, zc, kf, ray, rdel, &F); free(rdel);
        SocpOpts o3 = *opt; o3.verbose = getenv("BRISK_CONEPURIFYDBG") ? 1 : 0; o3.attempt = 0; SocpRes R3; memset(&R3, 0, sizeof R3);
        const int st3 = F.P.n > 0 ? socp_solve_pre(&F.P, &o3, &R3) : SOCP_INPUT;
        int took = 0;
        if (getenv("BRISK_CONEPURIFYDBG")) socp_printf("  [the end step's face: %d rows, %d variables (free %d, nonnegative %d, cones %d + rotated %d); its solve %s]\n", F.P.m, F.P.n, F.P.nf, F.P.nl, F.P.nq, F.P.nr, socp_status_str(st3));
        if (R3.x && R3.y && (st3 == SOCP_OPTIMAL || st3 == SOCP_REDUCED)) {
            double *x3 = (double *)calloc((size_t)n + 1, sizeof(double)), *z3 = (double *)calloc((size_t)n + 1, sizeof(double));
            fr_face_x(P, ks, kd, ray, &F, R3.x, x3);
            for (int j = 0; j < n; j++) { double v = P->c[j]; for (int p = P->Ap[j]; p < P->Ap[j + 1]; p++) v -= P->Ax[p] * R3.y[P->Ai[p]]; z3[j] = j < nf ? 0.0 : v; }
            double po3, do3, ep3; const double e3 = fr_measure_parts(P, x3, R3.y, z3, &po3, &do3, &ep3);
            if (getenv("BRISK_CONEPURIFYDBG")) socp_printf("  [the end step on the face: %s in %d iterations (%.1fs); the measure on the problem as given %.2e (objective %.10e)]\n", socp_status_str(st3), R3.iters, R3.time, e3, po3);
            if (e3 <= tol) { memcpy(R->x, x3, sizeof(double) * (size_t)n); memcpy(R->y, R3.y, sizeof(double) * (size_t)m); memcpy(R->z, z3, sizeof(double) * (size_t)n);
                R->maxerr = e3; R->pobj = po3; R->dobj = do3; R->iters += R3.iters; took = 1;
                SNOTE("the end step: the face that rounding each pair chose solved again (%d iterations): complementarity %.1e -> within the tolerance", R3.iters, R->err[3]); }
            free(x3); free(z3);
        }
        free(R3.x); free(R3.y); free(R3.z); free(R3.cx); free(R3.cy); free(R3.cz); fr_face_free(&F);
        free(x); free(z); free(zc); free(kf); free(ray); free(ks); free(kd); return took;
    }
    if (!(e <= tol)) { free(x); free(z); free(zc); free(kf); free(ray); free(ks); free(kd); return 0; }
    free(zc); free(kf); free(ray); free(ks); free(kd);
    memcpy(R->x, x, sizeof(double) * (size_t)n); memcpy(R->z, z, sizeof(double) * (size_t)n); free(x); free(z);
    R->err[0] = e0; R->err[1] = e1; R->err[2] = e2; R->err[3] = e3; R->maxerr = e; R->pobj = po; R->dobj = dob;
    SNOTE("the end step: each pair rounded to its face (complementarity %.1e -> %.1e, the other errors within the tolerance)", R->err[3], e3);
    return 1;
}
/* (11 November 2026, the SOCP developer) the measure of a result taken on a face read after a failed solve (the primal on the problem as given,
 * the dual on the face), for the caller's closing line; -1: none */
double socp_face_err(void) { return g_fr_face_err; }
static int socp_solve_seq(const SocpProb *P, const SocpOpts *opt, SocpRes *R) {
    const double t0 = sc_time();
    SocpOpts o1; if (opt) o1 = *opt; else socp_default_opts(&o1);
    const int retry = !(o1.step > 0) && !getenv("BRISK_CONENORETRY");
    g_fr_face_err = -1; g_fr_adv[0] = -1; g_fr_adv[1] = g_fr_adv[2] = 0;
    o1.attempt = retry ? 1 : 0;
    int st = socp_solve_pre(P, &o1, R);
    { static int pur = -1; if (pur < 0) { const char *e_ = getenv("BRISK_CONEPURIFY"); pur = e_ ? atoi(e_) : 0; }
      if (pur && (st == SOCP_REDUCED || st == SOCP_NUMERR) && socp_purify(P, &o1, R, o1.tol, pur)) { st = SOCP_OPTIMAL; R->status = st; } }
    if (!retry || (st != SOCP_REDUCED && st != SOCP_NUMERR) || *socp_stop || (socp_stop_call && *socp_stop_call)) return st;
    if (o1.noretry && st == SOCP_REDUCED) return st;
    if (R->stalled) return st;      /* (the stall stop: a second attempt would repeat the stall) */      /* (the caller recovers a point near the tolerance; a failed attempt is repeated) */
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
