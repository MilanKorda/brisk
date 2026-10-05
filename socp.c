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
#include <string.h>
#include <math.h>
#include <time.h>
#include "socp.h"
#include "amd/amd.h"

/* the supernodal sparse factorization of quasi-definite matrices (sparsechol.c): L S L' with the
 * signs S prescribed, fill-reducing orders, BLAS panels, threaded over the elimination tree */
typedef struct SChol SChol;
SChol *schol_analyze_adj(int m, int *deg, int **nbr, size_t fillcap);
void   schol_free(SChol *S);
double *schol_values(SChol *S);
size_t schol_offset(const SChol *S, size_t r, size_t c);
void   schol_set_signs(SChol *S, const signed char *sgn_orig);
void   schol_set_tinypiv(SChol *S, double t);
int    schol_factor(SChol *S, double shift);
void   schol_solve(const SChol *S, double *b, double *work);
size_t schol_nnz(const SChol *S);
int    schol_set_amd(int mode);
const int *schol_perm(const SChol *S);
void   schol_set_amd_dense(double d);
void   schol_set_solve_seq(int on);
void   schol_set_perm(const int *perm);
void   schol_zero(SChol *S);
double schol_flops(const SChol *S);
int    schol_ntiny(const SChol *S);

/* printing and interruption are the caller's: socp_printf (default: printf), *socp_stop (default: never) */
int (*socp_printf)(const char *fmt, ...) = printf;
static volatile int socp_never = 0;
volatile int *socp_stop = &socp_never;
static double sc_time(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + 1e-9 * t.tv_nsec; }

static int dbg_on(void) { static int v = -1; if (v < 0) v = getenv("BRISK_CONEDBG") != NULL; return v; }

#define NE_MAXFREE 8
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
    int dd;                           /* the current factor is the double-double one */
    long cap;                         /* ldl_symbolic gives up beyond this many entries (0: no bound) */
    double dyn_delta; int ndyn; int pivdbg, pivnf, pivn, pivnaux; double pivmin[4]; long pivcnt[4];
    long lnz;
} Ldl;

static int ldl_symbolic(Ldl *F) {
    const int n = F->n;
    int *work = F->iw;
    long tot = 0;
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
    return 0;
}

/* numeric factorization; returns the number of pivots that were replaced */
static int ldl_numeric(Ldl *F) {
    const int n = F->n;
    int *yIdx = F->iw, *elim = F->iw + n, *next = F->iw + 2 * n;
    unsigned char *mark = F->bw;
    double *y = F->fw;
    F->ndyn = 0;
    for (int i = 0; i < n; i++) { mark[i] = 0; y[i] = 0.0; next[i] = F->Lp[i]; }
    for (int k = 0; k < n; k++) {
        int nny = 0;
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
        if (!(F->sign[k] * dk > 0)) { dk = F->sign[k] * F->dyn_delta; F->ndyn++; }       /* a pivot of the wrong sign: replaced (the caller turns to double-double) */
        F->D[k] = dk; F->Dinv[k] = 1.0 / dk;
        if (F->pivdbg) { const int o = F->perm[k]; const int cl = o < F->pivnf ? 0 : o < F->pivn ? 1 : o < F->pivn + F->pivnaux ? 2 : 3; if (fabs(dk) < F->pivmin[cl]) F->pivmin[cl] = fabs(dk); if (fabs(dk) < 1e-6) F->pivcnt[cl]++; }
    }
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
static int ldl_numeric_dd(Ldl *F) {
    const int n = F->n;
    if (!F->Lxl) {
        F->Lxl = (double *)malloc(sizeof(double) * (size_t)(F->lnz + 1)); F->Dl = (double *)malloc(sizeof(double) * (size_t)n);
        F->Dinvl = (double *)malloc(sizeof(double) * (size_t)n); F->fwl = (double *)malloc(sizeof(double) * (size_t)n);
    }
    int *yIdx = F->iw, *elim = F->iw + n, *next = F->iw + 2 * n;
    unsigned char *mark = F->bw;
    double *y = F->fw, *yl = F->fwl;
    F->ndyn = 0;
    for (int i = 0; i < n; i++) { mark[i] = 0; y[i] = 0.0; yl[i] = 0.0; next[i] = F->Lp[i]; }
    for (int k = 0; k < n; k++) {
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
        if (!(F->sign[k] * dk.h > 0)) { dk.h = F->sign[k] * F->dyn_delta; dk.l = 0.0; F->ndyn++; }
        const sdd di = dd_inv(dk);
        F->D[k] = dk.h; F->Dl[k] = dk.l; F->Dinv[k] = di.h; F->Dinvl[k] = di.l;
    }
    F->dd = 1;
    return F->ndyn;
}
static void ldl_solve_perm_dd(const Ldl *F, double *x) {
    const int n = F->n;
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

/* x <- (L D L')^{-1} x in the permuted order */
static void ldl_solve_perm(const Ldl *F, double *x) {
    const int n = F->n;
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
    int *Rp, *Ri; double *Rx;      /* A scaled, by rows (column indices) */
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
    double sreg, sregx, sregf, sreg0, bump; int ref_pred, bump_on, nbump;
    int use_dd, dd_on, ndd, want_dd; long ncor_total;          /* the factorizations are in double-double from now on; allowed; how many */
    int dd_polish; double it_maxres;                            /* double-double asked by the end-phase rule (its own work limit); worst GMRES residual of the iteration */
    double *w1, *w2, *w3, *w4, *rk, *sk;   /* work: n+m sized */
    double *lastx, *lasty, *swork2;
    double *gs, *gV, *gZ, *gH, *gcs; int gm_max; double gm_tol;   /* GMRES on the KKT system */
    double rr_est; double t_mul; double cflops; double last_res; double t_setup, t_factor, t_solve; long nfact; long nback, slnz, nfallback; int sc_ok;
    /* the normal equations A H^-1 A' + delta I (no free variable, small cones, no dense column) */
    SChol *ne; size_t *ne_off, *ne_doff; double *ne_G; int *ne_r, **ne_rows; int ne_ok, ne_fallback, ne_tiny, ne_left, free_last, dz_refine; double *wdz, *wdx, *wdy, *wdy2; long dz_count; double *ne_W, ne_T[NE_MAXFREE * NE_MAXFREE]; long ndyn_total; long ne_nfact, ne_lnz;
    SChol *sc; size_t *soff; long knz; double *swork;   /* the supernodal factorization, when it is used */
    int refine_max, verbose;
    long refine_total, solves;
} Socp;

/* y += alpha A x ; y += alpha A' x */
static void A_mul(const Socp *S, double al, const double *x, double *y) {
    for (int j = 0; j < S->n; j++) { const double xj = al * x[j]; if (xj != 0.0) for (int p = S->Ap[j]; p < S->Ap[j + 1]; p++) y[S->Ai[p]] += S->Ax[p] * xj; }
}
static void At_mul(const Socp *S, double al, const double *x, double *y) {
    for (int j = 0; j < S->n; j++) { double s = 0.0; for (int p = S->Ap[j]; p < S->Ap[j + 1]; p++) s += S->Ax[p] * x[S->Ai[p]]; y[j] += al * s; }
}
static double dotv(int n, const double *a, const double *b) { double s = 0; for (int i = 0; i < n; i++) s += a[i] * b[i]; return s; }
static double nrm2(int n, const double *a) { return sqrt(dotv(n, a, a)); }
static double nrmi(int n, const double *a) { double s = 0; for (int i = 0; i < n; i++) if (fabs(a[i]) > s) s = fabs(a[i]); return s; }

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
    for (int i = 0; i < S->nl; i++) out[S->nf + i] = S->h[i] * in[S->nf + i];
    for (int k = 0; k < S->nq; k++) {
        const int d = S->q[k], s = S->qs[k]; const double *w = S->wbar + s; const double e2 = S->eta[k] * S->eta[k];
        double wx = 0.0;
        for (int i = 0; i < d; i++) wx += w[i] * in[s + i];
        out[s] = e2 * (2.0 * wx * w[0] - in[s]);
        for (int i = 1; i < d; i++) out[s + i] = e2 * (2.0 * wx * w[i] + in[s + i]);
    }
}

/* the values of the factored matrix from the current scaling */
static void kkt_fill(Socp *S) {
    Ldl *F = &S->F;
    if (S->ne && !S->ne_fallback) return;            /* (the normal equations are assembled by ne_factor) */
    const double sr = S->sreg;
    for (int i = 0; i < S->nf; i++) F->Ax[S->kdiag[i]] = -(S->use_dd ? sr : S->sregf);
    const double sx = S->sregx;
    for (int i = 0; i < S->nl; i++) F->Ax[S->kdiag[S->nf + i]] = -S->h[i] - sx;
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
                    F->Ax[pos[t++]] = i == j ? -e2 * lam - sx : 0.0;
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
                    F->Ax[pos[t++]] = -e2 * v - (i == j ? sx : 0.0);
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
            for (int i = 0; i < d; i++) F->Ax[S->kdiag[s + i]] = -e2 - sx;
            F->Ax[S->ku[k][0]] = eta * su; F->Ax[S->kv[k][0]] = eta * sv;
            for (int i = 1; i < d; i++) { F->Ax[S->ku[k][i]] = eta * su * iq * w[i]; F->Ax[S->kv[k][i]] = -eta * sv * iq * w[i]; }
            F->Ax[S->kdiag[S->auxu[k]]] = 1.0;
            F->Ax[S->kdiag[S->auxv[k]]] = -1.0;
        }
    }
    for (int i = 0; i < S->m; i++) F->Ax[S->kdiag[S->n + S->naux + i]] = sr;
}

/* factors the matrix of kkt_fill; returns the number of pivots replaced (own factorization) or 1 when the
 * supernodal factorization met a pivot of the wrong sign */
/* out = H^-1 in on the cone part (free variables: 0); H^-1 = eta^-2 (2 u u' - J), u = J w */
static void Hinv_apply(const Socp *S, const double *in, double *out) {
    for (int i = 0; i < S->nf; i++) out[i] = 0.0;
    for (int i = 0; i < S->nl; i++) out[S->nf + i] = in[S->nf + i] / S->h[i];
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
static void ne_free(Socp *S) {
    if (S->ne) schol_free(S->ne);
    S->ne = NULL;
    free(S->ne_off); free(S->ne_doff); free(S->ne_G); free(S->ne_r);
    if (S->ne_rows) { free(S->ne_rows[0]); free(S->ne_rows); }
    S->ne_off = S->ne_doff = NULL; S->ne_G = NULL; S->ne_r = NULL; S->ne_rows = NULL;
    free(S->ne_W); S->ne_W = NULL;
}
static int cmp_ll(const void *a, const void *b) { const long long x = *(const long long *)a, y = *(const long long *)b; return x < y ? -1 : x > y; }
/* returns 1 when the normal equations are set up (S->ne), 0 when they are not used */
static int ne_setup(Socp *S) {
    const int n = S->n, m = S->m, nb = S->nl + S->nq;
    if (S->nf > NE_MAXFREE || nb <= 0 || m < 64 || getenv("BRISK_CONENONE")) return 0;
    for (int k = 0; k < S->nq; k++) if (S->q[k] > NE_MAXCONE) return 0;
    /* the rows of every block; the number of pairs */
    int *r = (int *)malloc(sizeof(int) * (size_t)nb), *mark = (int *)malloc(sizeof(int) * (size_t)m);
    for (int i = 0; i < m; i++) mark[i] = -1;
    long tot = 0; double pairs = 0;
    for (int k = 0; k < nb; k++) {
        const int c0 = k < S->nl ? S->nf + k : S->qs[k - S->nl], c1 = k < S->nl ? c0 + 1 : c0 + S->q[k - S->nl];
        int cnt = 0;
        for (int p = S->Ap[c0]; p < S->Ap[c1]; p++) if (mark[S->Ai[p]] != k) { mark[S->Ai[p]] = k; cnt++; }
        r[k] = cnt; tot += cnt; pairs += 0.5 * (double)cnt * (cnt + 1);
    }
    if (getenv("BRISK_CONEDBG")) socp_printf("  [normal equations: %.0f pairs, nnz(A) %d, m %d]\n", pairs, S->Ap[n], m);
    if (!getenv("BRISK_CONENE") && pairs > 0.75 * (double)S->F.lnz) { free(r); free(mark); return 0; }   /* (no gain to expect: no analysis) */
    if (pairs > 40.0 * (double)S->Ap[n] + 1e5 || pairs > 2e8) { free(r); free(mark); return 0; }        /* dense columns */
    int **rows = (int **)malloc(sizeof(int *) * (size_t)nb), *pool = (int *)malloc(sizeof(int) * (size_t)(tot + 1));
    for (int i = 0; i < m; i++) mark[i] = -1;
    { long o = 0;
      for (int k = 0; k < nb; k++) {
        const int c0 = k < S->nl ? S->nf + k : S->qs[k - S->nl], c1 = k < S->nl ? c0 + 1 : c0 + S->q[k - S->nl];
        rows[k] = pool + o; int cnt = 0;
        for (int p = S->Ap[c0]; p < S->Ap[c1]; p++) if (mark[S->Ai[p]] != k) { mark[S->Ai[p]] = k; rows[k][cnt++] = S->Ai[p]; }
        for (int a = 1; a < cnt; a++) { const int v = rows[k][a]; int b_ = a; while (b_ > 0 && rows[k][b_ - 1] > v) { rows[k][b_] = rows[k][b_ - 1]; b_--; } rows[k][b_] = v; }
        o += cnt;
      } }
    /* the pattern: the distinct pairs */
    const size_t np = pairs > (double)tot ? (size_t)(pairs - (double)tot) : 0;   /* off-diagonal pairs with repetitions */
    long long *key = (long long *)malloc(sizeof(long long) * (np + 1));
    long q = 0;
    for (int k = 0; k < nb; k++) for (int a = 1; a < r[k]; a++) for (int b_ = 0; b_ < a; b_++) key[q++] = ((long long)rows[k][a] << 32) | (unsigned)rows[k][b_];
    qsort(key, (size_t)q, sizeof(long long), cmp_ll);
    long u = 0;
    for (long e = 0; e < q; e++) if (e == 0 || key[e] != key[e - 1]) key[u++] = key[e];
    int *deg = (int *)calloc((size_t)m + 1, sizeof(int)), **nbr = (int **)malloc(sizeof(int *) * ((size_t)m + 1));
    for (long e = 0; e < u; e++) { deg[(int)(key[e] >> 32)]++; deg[(int)(key[e] & 0xffffffffLL)]++; }
    int *npool = (int *)malloc(sizeof(int) * (size_t)(2 * u + 1));
    { long o = 0; for (int i = 0; i < m; i++) { nbr[i] = npool + o; o += deg[i]; deg[i] = 0; } }
    for (long e = 0; e < u; e++) { const int i = (int)(key[e] >> 32), j = (int)(key[e] & 0xffffffffLL); nbr[i][deg[i]++] = j; nbr[j][deg[j]++] = i; }
    free(key);
    const int om = schol_set_amd(2);
    SChol *sc = schol_analyze_adj(m, deg, nbr, (size_t)1 << 31);
    schol_set_amd(om);
    free(deg); free(nbr); free(npool); free(mark);
    if (!sc) { free(r); free(pool); free(rows); return 0; }
    /* worth it against the factorization of the full system? (work of the two, the same unit) */
    if (!getenv("BRISK_CONENE") && schol_flops(sc) > 1.5 * S->cflops + 1e6) { schol_free(sc); free(r); free(pool); free(rows); return 0; }
    S->ne = sc; S->ne_r = r; S->ne_rows = rows; S->ne_lnz = (long)schol_nnz(sc);
    S->ne_off = (size_t *)malloc(sizeof(size_t) * (size_t)(pairs + 1)); S->ne_doff = (size_t *)malloc(sizeof(size_t) * (size_t)m);
    size_t o = 0; int bad = 0; long gsz = 0;
    for (int k = 0; k < nb; k++) {
        for (int a = 0; a < r[k]; a++) for (int b_ = 0; b_ <= a; b_++) { const size_t f = schol_offset(sc, (size_t)rows[k][a], (size_t)rows[k][b_]); if (f == (size_t)-1) bad = 1; S->ne_off[o++] = f; }
        if (k >= S->nl) gsz += (long)r[k] * S->q[k - S->nl];
    }
    for (int i = 0; i < m; i++) { S->ne_doff[i] = schol_offset(sc, (size_t)i, (size_t)i); if (S->ne_doff[i] == (size_t)-1) bad = 1; }
    if (bad) { ne_free(S); return 0; }
    /* the cones' columns of A as dense r x d matrices (by rows) */
    S->ne_G = (double *)calloc((size_t)gsz + 1, sizeof(double));
    int *posr = (int *)malloc(sizeof(int) * (size_t)m);
    long g = 0;
    for (int k = S->nl; k < nb; k++) {
        const int d = S->q[k - S->nl], c0 = S->qs[k - S->nl];
        for (int a = 0; a < r[k]; a++) posr[rows[k][a]] = a;
        for (int t = 0; t < d; t++) for (int p = S->Ap[c0 + t]; p < S->Ap[c0 + t + 1]; p++) S->ne_G[g + (long)posr[S->Ai[p]] * d + t] += S->Ax[p];
        g += (long)r[k] * d;
    }
    free(posr);
    return 1;
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
    const double *G = S->ne_G;
    double Hi[NE_MAXCONE * NE_MAXCONE], T[NE_MAXCONE];
    for (int k = S->nl; k < nb; k++) {
        const int kc = k - S->nl, d = S->q[kc], s = S->qs[kc], rk = S->ne_r[k];
        const double *w = S->wbar + s; const double ie2 = 1.0 / (S->eta[kc] * S->eta[kc]);
        for (int i = 0; i < d; i++) for (int j = 0; j < d; j++) {
            const double ui = i == 0 ? w[0] : -w[i], uj = j == 0 ? w[0] : -w[j];
            Hi[i * d + j] = ie2 * (2.0 * ui * uj - (i == j ? (i == 0 ? 1.0 : -1.0) : 0.0));
        }
        for (int a = 0; a < rk; a++) {
            const double *ga = G + (size_t)a * d;
            for (int j = 0; j < d; j++) { double t = 0; for (int i = 0; i < d; i++) t += ga[i] * Hi[i * d + j]; T[j] = t; }
            for (int b_ = 0; b_ <= a; b_++) { const double *gb = G + (size_t)b_ * d; double v = 0; for (int j = 0; j < d; j++) v += T[j] * gb[j]; pm[*off++] += v; }
        }
        G += (size_t)rk * d;
    }
    for (int i = 0; i < m; i++) pm[S->ne_doff[i]] += S->sreg0;
    schol_set_tinypiv(S->ne, 1e-14);
    const int bad = schol_factor(S->ne, 0.0);
    const int nt = (int)schol_ntiny(S->ne);
    S->ne_tiny += nt;
    S->ne_nfact++;
    if (bad == 0 && S->nf > 0) {
        /* W = S^-1 A_f, T = A_f' W (+ a small shift) = L L' */
        const int nf = S->nf;
        if (!S->ne_W) S->ne_W = (double *)malloc(sizeof(double) * (size_t)m * nf);
        for (int f = 0; f < nf; f++) {
            double *w = S->ne_W + (size_t)f * m;
            for (int i = 0; i < m; i++) w[i] = 0.0;
            for (int p = S->Ap[f]; p < S->Ap[f + 1]; p++) w[S->Ai[p]] += S->Ax[p];
            schol_set_solve_seq(1); schol_solve(S->ne, w, S->swork2); schol_set_solve_seq(0);
        }
        double *T = S->ne_T;
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
    return bad != 0 || nt > 0;       /* (a tiny pivot: dependent rows, the product form has lost them) */
}
/* v = (dx; dy) for r = (rx; ry) */
static void ne_prec(Socp *S, const double *r, double *v) {
    const int n = S->n, m = S->m;
    double *hx = S->w4, *t = S->rk;
    Hinv_apply(S, r, hx);
    for (int i = 0; i < m; i++) t[i] = r[n + i];
    A_mul(S, 1.0, hx, t);
    { const double ts = sc_time(); schol_set_solve_seq(1); schol_solve(S->ne, t, S->swork2); schol_set_solve_seq(0); S->t_solve += sc_time() - ts; S->nback++; }
    double xf[NE_MAXFREE];
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
    double *pm = schol_values(S->sc);
    for (long q = 0; q < S->knz; q++) pm[S->soff[q]] = F->Ax[q];
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
            pm = schol_values(S->sc);
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
static void kkt_back(Socp *S, double *t) {
    if (S->sc && S->sc_ok) {
        const int nk = S->nk; const Ldl *F = &S->F;
        double *u = S->swork + nk;
        for (int i = 0; i < nk; i++) u[i] = t[F->iperm[i]];
        schol_set_solve_seq(1); schol_solve(S->sc, u, S->swork); schol_set_solve_seq(0);     /* (the sequential solve: the tree-parallel one is slower here) */
        for (int i = 0; i < nk; i++) t[F->iperm[i]] = u[i];
    } else if (S->F.dd) ldl_solve_perm_dd(&S->F, t);
    else ldl_solve_perm(&S->F, t);
}

/* fill and factor; a supernodal factorization that fails is repeated with more regularization */
static void kkt_fill(Socp *S);
static int kkt_factor_reg(Socp *S) {
    const double ts = sc_time();
    if (S->ne) {
        S->ne_ok = !ne_factor(S);
        if (S->ne_ok) { S->F.ndyn = 0; S->F.dd = 0; S->t_factor += sc_time() - ts; S->nfact++; return 0; }
        ne_free(S); kkt_fill(S);                                   /* failed: the full system from here on */
    }
    if (dbg_on() && !S->sc) { S->F.pivdbg = 1; S->F.pivnf = S->nf; S->F.pivn = S->n; S->F.pivnaux = S->naux; for (int c = 0; c < 4; c++) { S->F.pivmin[c] = 1e300; S->F.pivcnt[c] = 0; } }
    int f = kkt_factor(S);
    if (S->F.pivdbg) socp_printf("      [pivots: min |d| free %.1e cone %.1e aux %.1e rows %.1e; below 1e-6: %ld %ld %ld %ld]\n", S->F.pivmin[0], S->F.pivmin[1], S->F.pivmin[2], S->F.pivmin[3], S->F.pivcnt[0], S->F.pivcnt[1], S->F.pivcnt[2], S->F.pivcnt[3]);
    /* pivots of the wrong sign: from here on the factorization is in double-double, with a
     * regularization at its rounding level */
    const double ddmax = 5e8;
    if (S->dd_on && !S->use_dd && S->want_dd && S->sc && S->cflops > ddmax && !S->dd_polish) {
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
static void kkt_prec(Socp *S, const double *r, double *v) {
    const int n = S->n, m = S->m, nk = S->nk;
    if (S->ne && S->ne_ok) { ne_prec(S, r, v); return; }
    const Ldl *F = &S->F;
    double *t = S->rk;
    for (int i = 0; i < nk; i++) t[i] = 0.0;
    for (int i = 0; i < n; i++) t[F->iperm[i]] = r[i];
    for (int i = 0; i < m; i++) t[F->iperm[n + S->naux + i]] = r[n + i];
    /* rotated cones: the unknown is Q' x */
    for (int k = 0; k < S->nq; k++) if (S->rot[k]) {
        const int d = S->q[k], s = S->qs[k]; const double *Q = S->Qr + 16 * (size_t)k;
        double rr[4];
        for (int i = 0; i < d; i++) { double a = 0; for (int j = 0; j < d; j++) a += Q[j + i * d] * r[s + j]; rr[i] = a; }
        for (int i = 0; i < d; i++) t[F->iperm[s + i]] = rr[i];
    }
    { const double ts = sc_time(); kkt_back(S, t); S->t_solve += sc_time() - ts; S->nback++; }
    for (int i = 0; i < n; i++) v[i] = t[F->iperm[i]];
    for (int i = 0; i < m; i++) v[n + i] = t[F->iperm[n + S->naux + i]];
    for (int k = 0; k < S->nq; k++) if (S->rot[k]) {
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
    At_mul(S, 1.0, v + n, w);
    const double dl = S->use_dd ? S->sreg : S->sreg0;       /* the (base) dual regularization is part of the system that is solved */
    for (int i = 0; i < m; i++) w[n + i] = dl * v[n + i];
    A_mul(S, 1.0, v, w + n);
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
    if (maxref > 0 && bn > 0) {
        const int kmax = maxref < S->gm_max ? maxref : S->gm_max;
        double *V = S->gV, *Z = S->gZ, *Hm = S->gH, *cs = S->gcs, *sn = S->gcs + kmax + 1, *g = S->gcs + 2 * (kmax + 1);
        const double tol = S->gm_tol * bn;
        double *w = S->lastx;                 /* N entries of work (lastx and lasty are contiguous: see the allocation) */
        kkt_mul(S, s, w);
        for (int i = 0; i < n; i++) w[i] = rx[i] - w[i];
        for (int i = 0; i < m; i++) w[n + i] = ry[i] - w[n + i];
        double beta = nrm2(N, w);
        S->last_res = beta / bn;
        if (beta > tol) {
            for (int i = 0; i < N; i++) V[i] = w[i] / beta;
            g[0] = beta;
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
                if (fabs(g[k + 1]) <= tol || !(hn > 0)) { k++; break; }
            }
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
    for (int i = S->nf; i < S->nf + S->nl; i++) if (dx[i] < 0) { const double r = -x[i] / dx[i]; if (r < al) al = r; }
    for (int k = 0; k < S->nq; k++) al = soc_step(S->q[k], x + S->qs[k], dx + S->qs[k], al);
    return al;
}

/* 1 if x and z are in the interior of the cone (the test of nt_update) */
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

/* builds the pattern of the factored matrix, orders it and maps the entries */
static int kkt_setup(Socp *S) {
    const double flop_ratio = 110.0;
    double nd0 = 0;                   /* dense rows found by the first AMD pass */
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
    if (getenv("BRISK_CONEROT")) {        /* (off: no gain measured, see TRICKS.md section 8) */
        int *mark = (int *)malloc(sizeof(int) * (size_t)m); for (int i = 0; i < m; i++) mark[i] = -1;
        for (int k = 0; k < S->nq; k++) {
            const int d = S->q[k], s = S->qs[k];
            if (S->auxu[k] >= 0 || d < 2) continue;
            int cnt_ = 0;
            for (int j = 0; j < d; j++) for (int p = S->Ap[s + j]; p < S->Ap[s + j + 1]; p++) if (mark[S->Ai[p]] != k) { mark[S->Ai[p]] = k; cnt_++; }
            S->rot[k] = 1; S->rnr[k] = cnt_;
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
        static const double dn[2] = { 3.0, 10.0 };
        nd0 = 1;
        const long big = 1000000L;
        int var_gain = 0;                 /* the variant of the first pass reduced the work */
        for (int t = 0; t < 2; t++) {
            if (t == 1 && bestfl >= 0 && (bestlnz <= big || nd0 == 0)) break;       /* (no dense row at the low threshold: the same order) */
            /* the first pass's dense rows were the right ones when moving their neighbours last paid: no
             * second pass then (AMD at the higher threshold orders the rows between the two thresholds, at
             * a cost that can be large: 27 s on a lasso problem with 4,000 such rows, for the same order) */
            if (t == 1 && var_gain && !getenv("BRISK_CONEAMD2")) break;
            double ctl[AMD_CONTROL], info[AMD_INFO], fl_;
            amd_defaults(ctl);
            ctl[AMD_DENSE] = dn[t];
            const double ta = sc_time();
            {
                int *cred = nsing ? (int *)malloc(sizeof(int) * (size_t)(nred + 1)) : cand;
                if (amd_order(nred, cntr, fir, cred, ctl, info) < 0) for (int i = 0; i < nred; i++) cred[i] = i;
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
         * delta + a^2 / H_jj). nql60: 568 -> 349 backsolves, 17 -> 15 iterations. */
        if (!getenv("BRISK_CONENOROWFIX")) {
            int *pos = (int *)malloc(sizeof(int) * (size_t)nk), *after = (int *)malloc(sizeof(int) * (size_t)nk), nmoved = 0;
            for (int i = 0; i < nk; i++) { pos[bestp[i]] = i; after[i] = -1; }
            int *cmin = (int *)malloc(sizeof(int) * (size_t)m);
            for (int i = 0; i < m; i++) cmin[i] = -1;
            for (int j = 0; j < n; j++) for (int p = S->Ap[j]; p < S->Ap[j + 1]; p++) { const int i = S->Ai[p]; if (cmin[i] < 0 || pos[j] < pos[cmin[i]]) cmin[i] = j; }
            for (int i = 0; i < m; i++) { const int r = n + S->naux + i; if (cmin[i] >= 0 && pos[r] < pos[cmin[i]] && owner[r] < 0) { after[r] = cmin[i]; nmoved++; } }
            /* a local repair, not a reordering: with more than a fifth of the rows (tv_100000: half of
             * them; AMD's order is then the right one, 17 instead of 20 iterations) nothing is moved */
            if (nmoved > m / 5) { if (dbg_on()) socp_printf("  [%d rows before all their columns: more than a fifth of the rows, left in place]\n", nmoved); nmoved = 0; }
            if (nmoved) {
                int *head = (int *)malloc(sizeof(int) * (size_t)nk), *next = (int *)malloc(sizeof(int) * (size_t)nk), *tail = (int *)malloc(sizeof(int) * (size_t)nk);
                for (int i = 0; i < nk; i++) head[i] = tail[i] = -1;
                for (int i = 0; i < nk; i++) { const int r = bestp[i]; if (after[r] >= 0) { const int j = after[r]; next[r] = -1; if (head[j] < 0) head[j] = r; else next[tail[j]] = r; tail[j] = r; } }
                int w = 0;
                for (int i = 0; i < nk; i++) { const int v = bestp[i]; if (after[v] >= 0) continue; F->perm[w++] = v; for (int r = head[v]; r >= 0; r = next[r]) F->perm[w++] = r; }
                memcpy(bestp, F->perm, sizeof(int) * (size_t)nk);
                free(head); free(next); free(tail);
                if (dbg_on()) socp_printf("  [%d rows moved after their first column]\n", nmoved);
            }
            free(pos); free(after); free(cmin);
        }
        memcpy(F->perm, bestp, sizeof(int) * (size_t)nk);
        KKT_COUNT();
        S->cflops = fl_;
    }
#undef KKT_COUNT
    free(fi); free(nx); free(cand); free(owner); free(rmap); free(sing_ptr); free(sing_row); if (nsing) { free(cntr); free(fir); }
    if (symb_fail) { free(bestp); free(cnt); free(pos); free(ti); free(tj); free(blk0); free(u0); free(v0); return 1; }
    /* the supernodal factorization, with the same order, when the factor is large or dense */
    {
        const char *e = getenv("BRISK_CONEBACKEND");
        /* (dense rows: the supernodal update batches the data rows into one product per tile, svm_5000 1.0 -> 0.65 s) */
        const int want = e ? atoi(e) : (F->lnz > 40000 && (F->lnz > 40L * nk || S->cflops > flop_ratio * (double)F->lnz || (nd0 >= 32 && S->Ap[n] >= 20L * m)));
        if (want && nk >= 64) {
            int *deg = (int *)xcalloc((size_t)nk + 1, sizeof(int)), **nbr = (int **)malloc(sizeof(int *) * ((size_t)nk + 1));
            for (long q = nk; q < nz; q++) { deg[ti[q]]++; deg[tj[q]]++; }
            int *pool = (int *)malloc(sizeof(int) * (size_t)(2 * (nz - nk) + 1));
            { long o = 0; for (int i = 0; i < nk; i++) { nbr[i] = pool + o; o += deg[i]; deg[i] = 0; } }
            for (long q = nk; q < nz; q++) { nbr[ti[q]][deg[ti[q]]++] = tj[q]; nbr[tj[q]][deg[tj[q]]++] = ti[q]; }
            const int om = schol_set_amd(2);
            schol_set_perm(bestp);
            S->sc = schol_analyze_adj(nk, deg, nbr, (size_t)1 << 31);
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
        free(sg);
        S->soff = (size_t *)malloc(sizeof(size_t) * (size_t)(nz + 1)); S->knz = nz;
        int bad = 0;
        for (long q = 0; q < nz; q++) { S->soff[pos[q]] = schol_offset(S->sc, (size_t)ti[q], (size_t)tj[q]); if (S->soff[pos[q]] == (size_t)-1) bad = 1; }
        if (bad) { schol_free(S->sc); S->sc = NULL; free(S->soff); S->soff = NULL; }
        else S->swork = (double *)xcalloc(2 * (size_t)nk + 2, sizeof(double));
    }
    F->Li = (int *)malloc(sizeof(int) * (size_t)(F->lnz + 1)); F->Lx = (double *)malloc(sizeof(double) * (size_t)(F->lnz + 1));
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
    free(F->sign); free(F->iw); free(F->bw); free(F->fw); free(F->perm); free(F->iperm);
    if (S->sc) schol_free(S->sc);
    free(S->soff); free(S->swork);
    for (int k = 0; k < S->nq; k++) { if (S->kblk) free(S->kblk[k]); if (S->ku) free(S->ku[k]); if (S->kv) free(S->kv[k]); }
    for (int k = 0; k < S->nq; k++) { if (S->rrows) free(S->rrows[k]); if (S->rG) free(S->rG[k]); if (S->rapos) free(S->rapos[k]); }
    free(S->rot); free(S->rrows); free(S->rnr); free(S->rG); free(S->rpos0); free(S->Qr); free(S->rapos);
    free(S->kblk); free(S->ku); free(S->kv); free(S->kdiag); free(S->auxu); free(S->auxv);
    free(S->q); free(S->qs); free(S->Ap); free(S->Ai); free(S->Ax); free(S->Rp); free(S->Ri); free(S->Rx); free(S->b); free(S->c);
    free(S->Dr); free(S->Ec); free(S->x); free(S->y); free(S->z); free(S->h); free(S->eta); free(S->wbar); free(S->lam);
    free(S->w1); free(S->w2); free(S->w3); free(S->w4); free(S->rk); free(S->sk); ne_free(S); free(S->swork2);
    free(S->lastx); free(S->gs); free(S->gV); free(S->gZ); free(S->gH); free(S->gcs);
}

void socp_default_opts(SocpOpts *o) { o->tol = 1e-8; o->maxit = 200; o->timelimit = 0; o->verbose = 1; o->equil = 1; o->step = 0; o->attempt = 0; }
const char *socp_status_str(int s) {
    return s == SOCP_OPTIMAL ? "OPTIMAL" : s == SOCP_PINF ? "PRIMAL INFEASIBLE" : s == SOCP_DINF ? "DUAL INFEASIBLE" : s == SOCP_MAXIT ? "ITERATION LIMIT"
         : s == SOCP_TIME ? "TIME LIMIT" : s == SOCP_REDUCED ? "SOLVED TO REDUCED ACCURACY" : s == SOCP_NUMERR ? "NUMERICAL DIFFICULTIES" : "INPUT ERROR";
}

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
    S->q = (int *)xcalloc((size_t)S->nq, sizeof(int)); S->qs = (int *)xcalloc((size_t)S->nq + 1, sizeof(int));
    { int s = S->nf + S->nl; for (int k = 0; k < S->nq; k++) { S->q[k] = k < P->nq ? P->q[k] : P->r[k - P->nq]; S->qs[k] = s; s += S->q[k]; } S->qs[S->nq] = s; }
    /* cones of dimension 1 are half-lines: kept as cones (x0 >= 0), the algebra is the same */
    /* ---- copy of A by columns; rotated cones: columns (c1, c2) -> ((c1 + c2), (c1 - c2)) / sqrt 2 */
    const double rs = sqrt(0.5);
    {
        long nz = P->Ap[n];
        for (int k = P->nq; k < S->nq; k++) { const int j = S->qs[k]; nz += (P->Ap[j + 2] - P->Ap[j]); }
        S->Ap = (int *)malloc(sizeof(int) * ((size_t)n + 1)); S->Ai = (int *)malloc(sizeof(int) * (size_t)(nz + 1)); S->Ax = (double *)malloc(sizeof(double) * (size_t)(nz + 1));
        double *acc = (double *)xcalloc((size_t)m, sizeof(double)); int *mk = (int *)malloc(sizeof(int) * ((size_t)m + 1)); int *lst = (int *)malloc(sizeof(int) * ((size_t)m + 1));
        for (int i = 0; i < m; i++) mk[i] = -1;
        long w = 0; int kr = P->nq;
        for (int j = 0; j < n; j++) {
            S->Ap[j] = (int)w;
            const int rot = kr < S->nq && (j == S->qs[kr] || j == S->qs[kr] + 1);
            if (!rot) { for (int p = P->Ap[j]; p < P->Ap[j + 1]; p++) if (P->Ax[p] != 0.0) { S->Ai[w] = P->Ai[p]; S->Ax[w] = P->Ax[p]; w++; } }
            else {
                const int j1 = S->qs[kr], j2 = j1 + 1; const double s2 = j == j1 ? 1.0 : -1.0;
                int nl_ = 0;
                for (int p = P->Ap[j1]; p < P->Ap[j1 + 1]; p++) { const int i = P->Ai[p]; if (mk[i] != j) { mk[i] = j; lst[nl_++] = i; acc[i] = 0; } acc[i] += rs * P->Ax[p]; }
                for (int p = P->Ap[j2]; p < P->Ap[j2 + 1]; p++) { const int i = P->Ai[p]; if (mk[i] != j) { mk[i] = j; lst[nl_++] = i; acc[i] = 0; } acc[i] += s2 * rs * P->Ax[p]; }
                for (int t = 0; t < nl_; t++) if (acc[lst[t]] != 0.0) { S->Ai[w] = lst[t]; S->Ax[w] = acc[lst[t]]; w++; }
                if (j == j2) kr++;
            }
        }
        S->Ap[n] = (int)w;
        free(acc); free(mk); free(lst);
    }
    S->b = (double *)xcalloc((size_t)m, sizeof(double)); S->c = (double *)xcalloc((size_t)n, sizeof(double));
    memcpy(S->b, P->b, sizeof(double) * (size_t)m); memcpy(S->c, P->c, sizeof(double) * (size_t)n);
    for (int k = P->nq; k < S->nq; k++) { const int j = S->qs[k]; const double c1 = S->c[j], c2 = S->c[j + 1]; S->c[j] = rs * (c1 + c2); S->c[j + 1] = rs * (c1 - c2); }
    /* the problem as given (after the map): kept for the error measures */
    double *A0x = (double *)malloc(sizeof(double) * (size_t)(S->Ap[n] + 1)); memcpy(A0x, S->Ax, sizeof(double) * (size_t)S->Ap[n]);
    double *b0 = (double *)xcalloc((size_t)m, sizeof(double)), *c0 = (double *)xcalloc((size_t)n, sizeof(double));
    memcpy(b0, S->b, sizeof(double) * (size_t)m); memcpy(c0, S->c, sizeof(double) * (size_t)n);
    const double nb0 = nrmi(m, b0), nc0 = nrmi(n, c0);
    /* ---- equilibration */
    S->Dr = (double *)malloc(sizeof(double) * ((size_t)m + 1)); S->Ec = (double *)malloc(sizeof(double) * (size_t)n);
    for (int i = 0; i < m; i++) S->Dr[i] = 1.0;
    for (int j = 0; j < n; j++) S->Ec[j] = 1.0;
    if (opt->equil) {
        double *rn = (double *)malloc(sizeof(double) * ((size_t)m + 1)), *cn = (double *)malloc(sizeof(double) * (size_t)n);
        for (int pass = 0; pass < 10; pass++) {
            for (int i = 0; i < m; i++) rn[i] = 0.0;
            for (int j = 0; j < n; j++) { double s = 0; for (int p = S->Ap[j]; p < S->Ap[j + 1]; p++) { const double v = fabs(S->Ax[p]); if (v > s) s = v; if (v > rn[S->Ai[p]]) rn[S->Ai[p]] = v; } cn[j] = s; }
            for (int k = 0; k < S->nq; k++) { double s = 0; for (int i = S->qs[k]; i < S->qs[k + 1]; i++) if (cn[i] > s) s = cn[i]; for (int i = S->qs[k]; i < S->qs[k + 1]; i++) cn[i] = s; }
            double dev = 0;
            for (int i = 0; i < m; i++) { if (rn[i] > 0) { dev = fmax(dev, fabs(1.0 - rn[i])); rn[i] = 1.0 / sqrt(rn[i]); } else rn[i] = 1.0; }
            for (int j = 0; j < n; j++) { if (cn[j] > 0) { dev = fmax(dev, fabs(1.0 - cn[j])); cn[j] = 1.0 / sqrt(cn[j]); } else cn[j] = 1.0; }
            for (int j = 0; j < n; j++) for (int p = S->Ap[j]; p < S->Ap[j + 1]; p++) S->Ax[p] *= rn[S->Ai[p]] * cn[j];
            for (int i = 0; i < m; i++) S->Dr[i] *= rn[i];
            for (int j = 0; j < n; j++) S->Ec[j] *= cn[j];
            if (dev < 1e-2) break;
        }
        free(rn); free(cn);
    }
    for (int i = 0; i < m; i++) S->b[i] *= S->Dr[i];
    for (int j = 0; j < n; j++) S->c[j] *= S->Ec[j];
    S->sb = fmax(1.0, nrmi(m, S->b)); S->scl = fmax(1.0, nrmi(n, S->c));
    if (!opt->equil) { S->sb = 1.0; S->scl = 1.0; }
    for (int i = 0; i < m; i++) S->b[i] /= S->sb;
    for (int j = 0; j < n; j++) S->c[j] /= S->scl;
    /* ---- workspace, pattern, ordering */
    S->x = (double *)xcalloc((size_t)n, sizeof(double)); S->y = (double *)xcalloc((size_t)m, sizeof(double)); S->z = (double *)xcalloc((size_t)n, sizeof(double));
    S->h = (double *)xcalloc((size_t)S->nl, sizeof(double)); S->eta = (double *)xcalloc((size_t)S->nq, sizeof(double));
    S->wbar = (double *)xcalloc((size_t)n, sizeof(double)); S->lam = (double *)xcalloc((size_t)n, sizeof(double));
    { const double ts = sc_time(); const int rc_ = kkt_setup(S); S->t_setup = sc_time() - ts;
      if (rc_) { socp_free(S); free(A0x); free(b0); free(c0); R->status = SOCP_INPUT; return SOCP_INPUT; } }
    const size_t wn = (size_t)S->nk + 8;
    S->w1 = (double *)xcalloc(wn, sizeof(double)); S->w2 = (double *)xcalloc(wn, sizeof(double)); S->w3 = (double *)xcalloc(wn, sizeof(double));
    S->w4 = (double *)xcalloc(wn, sizeof(double)); S->rk = (double *)xcalloc(wn, sizeof(double)); S->sk = (double *)xcalloc(wn, sizeof(double));
    S->lastx = (double *)xcalloc(2 * wn, sizeof(double)); S->lasty = S->lastx + wn;
    S->swork2 = (double *)xcalloc((size_t)m + 8, sizeof(double));
    { const double ts = sc_time(); ne_setup(S); S->t_setup += sc_time() - ts; }
    S->gm_max = 10; S->gm_tol = 1e-12;
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
    S->F.dyn_delta = 1e-7; S->ref_pred = 0;
    S->sreg0 = S->sreg; S->bump = 1.0; S->bump_on = getenv("BRISK_CONENOBUMP") == NULL;
    S->dd_on = getenv("BRISK_CONENODD") == NULL;
    S->dz_refine = getenv("BRISK_CONENODZ") == NULL;
    S->wdz = (double *)xcalloc((size_t)n + 1, sizeof(double)); S->wdx = (double *)xcalloc((size_t)n + 1, sizeof(double));
    S->wdy = (double *)xcalloc((size_t)m + 1, sizeof(double)); S->wdy2 = (double *)xcalloc((size_t)m + 1, sizeof(double));
    const double ncs = nrm2(n, S->c);
    /* ---- start: the identity of the cone */
    for (int i = S->nf; i < S->nf + S->nl; i++) { S->x[i] = 1.0; S->z[i] = 1.0; }
    for (int k = 0; k < S->nq; k++) { S->x[S->qs[k]] = 1.0; S->z[S->qs[k]] = 1.0; }
    S->tau = 1.0; S->kap = 1.0;
    if (!getenv("BRISK_CONENOLS") && nt_update(S) == 0) {
        /* the start: x the solution of A x = b of least norm, (y, z) the least-squares dual point
         * (min |z|, z = c - A'y), each moved into the cone along the identity when it is outside.
         * Both come from the first factorization (identity scaling). */
        kkt_fill(S);
        kkt_factor_reg(S);
        for (int j = 0; j < n; j++) dx[j] = 0.0;
        for (int i = 0; i < m; i++) dy[i] = 0.0;
        kkt_solve(S, dx, S->b, px, py, S->refine_max);
        kkt_solve(S, S->c, dy, qx, qy, S->refine_max);
        double sx = 1e300, sz = 1e300;
        for (int i = S->nf; i < S->nf + S->nl; i++) { sx = fmin(sx, px[i]); sz = fmin(sz, -qx[i]); }
        for (int k = 0; k < S->nq; k++) {
            const int j = S->qs[k], d = S->q[k];
            double a = 0, b_ = 0;
            for (int i = 1; i < d; i++) { a += px[j + i] * px[j + i]; b_ += qx[j + i] * qx[j + i]; }
            sx = fmin(sx, px[j] - sqrt(a)); sz = fmin(sz, -qx[j] - sqrt(b_));
        }
        int ok = 1;
        for (int j = 0; j < n; j++) if (!(fabs(px[j]) < 1e8) || !(fabs(qx[j]) < 1e8)) ok = 0;      /* (the scaled data are at most 1) */
        for (int i = 0; i < m; i++) if (!(fabs(qy[i]) < 1e8)) ok = 0;
        if (ok) {
            const double ax = sx < 1e-8 ? 1.0 - sx : 0.0, az = sz < 1e-8 ? 1.0 - sz : 0.0;
            for (int j = 0; j < S->nf; j++) { S->x[j] = px[j]; S->z[j] = 0.0; }
            for (int i = S->nf; i < S->nf + S->nl; i++) { S->x[i] = px[i] + ax; S->z[i] = -qx[i] + az; }
            for (int k = 0; k < S->nq; k++) {
                const int j = S->qs[k], d = S->q[k];
                S->x[j] = px[j] + ax; S->z[j] = -qx[j] + az;
                for (int i = 1; i < d; i++) { S->x[j + i] = px[j + i]; S->z[j + i] = -qx[j + i]; }
            }
            for (int i = 0; i < m; i++) S->y[i] = qy[i];
        }
    }
    const double nu = (double)(S->nl + S->nq + 1);
    int status = SOCP_MAXIT, it;
    if (S->verbose > 0) {
        socp_printf("BRISK second-order cone solver: m = %d, n = %d (free %d, linear %d, cones %d", m, n, S->nf, S->nl, S->nq);
        if (S->nq) { int mx = 0; for (int k = 0; k < S->nq; k++) if (S->q[k] > mx) mx = S->q[k]; socp_printf(", largest %d", mx); }
        socp_printf("), nnz(A) = %d, factor: %d unknowns, nnz(L) = %ld\n", S->Ap[n], S->nk, S->F.lnz);
        socp_printf("Number of threads: 1\n");   /* (this solver is sequential) */
    }
    if (S->verbose > 1) socp_printf("  it      pobj             dobj         pinf     dinf     gap      mu       tau    step  sigma  ref\n");
    double pobj = 0, dobj = 0, e_p = 1, e_d = 1, e_g = 1, e_c = 1;
    const double ne_k = 3.0;
    const double gm_switch = 1e-4, gm_tol_early = 1e-9, gm_tol_late = 1e-12;
    const int mcc_ref = 10, mcc_max = getenv("BRISK_CONEMCC") ? atoi(getenv("BRISK_CONEMCC")) : 3;
    const double mcc_da = 0.2, mcc_bmin = 0.1, mcc_bmax = 10.0, mcc_acc = 0.1, step_frac = opt->step > 0 && opt->step < 1 ? opt->step : 0.99;
    double *dxa = (double *)xcalloc((size_t)n, sizeof(double)), *dya = (double *)xcalloc((size_t)m + 1, sizeof(double)), *dza = (double *)xcalloc((size_t)n, sizeof(double));
    double *ctv = (double *)xcalloc((size_t)n, sizeof(double)), *dxs = (double *)xcalloc((size_t)n, sizeof(double)), *dzs = (double *)xcalloc((size_t)n, sizeof(double));
    double mu0 = 1.0;
    /* up to mcc_max correctors in every iteration (a rule by the cost of a factorization against a
     * solve was tried: the iterations saved are worth the two solves of a corrector on all the sets) */
    const int mcc_now = mcc_max;
    {
        const double lz = (double)(S->sc ? S->slnz : S->F.lnz);
        const double tf = S->cflops / (S->sc ? 6e9 : 1.2e9), tsv = lz * (S->sc ? 3e-9 : 1.5e-9) + (double)S->Ap[n] * 3e-9 + (double)(n + m) * 2e-8;
        S->rr_est = tf / tsv;
    }
    int ddbump = 0;
    int nfeas = 0, feas_pending = 0, nsmall = 0; const int feas_on = getenv("BRISK_CONENOFEAS") == NULL;
    double feas_prev = 0;
    double *sv_x = (double *)xcalloc((size_t)n, sizeof(double)), *sv_z = (double *)xcalloc((size_t)n, sizeof(double)), *sv_y = (double *)xcalloc((size_t)m + 1, sizeof(double));
    for (it = 0; ; it++) {
        /* ---- residuals of the homogeneous model (scaled problem) */
        for (int i = 0; i < m; i++) rP[i] = S->b[i] * S->tau;
        A_mul(S, -1.0, S->x, rP);                                        /* b tau - A x */
        for (int j = 0; j < n; j++) rD[j] = S->c[j] * S->tau - S->z[j];
        At_mul(S, -1.0, S->y, rD);                                       /* c tau - A'y - z */
        const double cx = dotv(n, S->c, S->x), byv = dotv(m, S->b, S->y);
        const double rG = byv - cx - S->kap;
        double xz = 0; for (int j = S->nf; j < n; j++) xz += S->x[j] * S->z[j];
        const double mu = (xz + S->tau * S->kap) / nu;
        /* ---- errors of the problem as given */
        {
            const double f = 1.0 / S->tau;
            for (int j = 0; j < n; j++) { xo[j] = S->Ec[j] * S->x[j] * f * S->sb; zo[j] = S->z[j] * f * S->scl / S->Ec[j]; }
            for (int i = 0; i < m; i++) yo[i] = S->Dr[i] * S->y[i] * f * S->scl;
            double *ra = S->w1, *rb = S->w2;
            for (int i = 0; i < m; i++) ra[i] = -b0[i];
            for (int j = 0; j < n; j++) { const double xj = xo[j]; if (xj != 0.0) for (int p = S->Ap[j]; p < S->Ap[j + 1]; p++) ra[S->Ai[p]] += A0x[p] * xj; }
            for (int j = 0; j < n; j++) { double s = 0; for (int p = S->Ap[j]; p < S->Ap[j + 1]; p++) s += A0x[p] * yo[S->Ai[p]]; rb[j] = c0[j] - s - zo[j]; }
            pobj = dotv(n, c0, xo); dobj = dotv(m, b0, yo);
            e_p = nrm2(m, ra) / (1.0 + nb0); e_d = nrm2(n, rb) / (1.0 + nc0);
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
        const double err = fmax(fmax(e_p, e_d), fmax(fabs(e_g), e_c));
        if (err < best && S->tau > 0) { best = err; memcpy(bx, xo, sizeof(double) * (size_t)n); memcpy(by, yo, sizeof(double) * (size_t)m); memcpy(bz, zo, sizeof(double) * (size_t)n); best_e[0] = e_p; best_e[1] = e_d; best_e[2] = e_g; best_e[3] = e_c; R->pobj = pobj; R->dobj = dobj; }
        if (S->verbose > 1 && it == 0) socp_printf("%4d %+.9e %+.9e %8.1e %8.1e %8.1e %8.1e %8.1e\n", it, pobj, dobj, e_p, e_d, e_g, mu, S->tau);
        if (err <= opt->tol) { status = SOCP_OPTIMAL; break; }
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
        if (opt->timelimit > 0 && sc_time() - t0 > opt->timelimit) { status = SOCP_TIME; break; }
        if (*socp_stop) { status = SOCP_TIME; break; }
        /* the accuracy of the solves: that of the regularized factorization while the errors are large,
         * the full one near the end */
        if (it == 0) mu0 = mu;
        S->gm_tol = err > gm_switch && mu > 1e-5 * mu0 && S->tau > 1e-3 * fmax(1.0, S->kap) ? gm_tol_early : gm_tol_late;      /* (also full accuracy on the way to an infeasibility certificate) */
        /* ---- scaling and factorization */
        S->it_maxres = 0.0;
        if (nt_update(S)) { status = SOCP_NUMERR; break; }
        kkt_fill(S);
        kkt_factor_reg(S);
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
        double sigma = 0.0, alpha = 0.0;
        double dtau = 0, dkap = 0;
        const long ref0 = S->refine_total;
        const long sol0 = S->solves;
        for (int pass = 0; pass < 2; pass++) {
            const double etaf = 1.0 - sigma;
            /* r3 = W (lambda \ d_s);  d_s = sigma mu e - lambda o lambda - (W dx_a) o (W^-1 dz_a) */
            for (int i = 0; i < S->nf; i++) r3[i] = 0.0;
            for (int i = S->nf; i < S->nf + S->nl; i++) {
                double d = sigma * mu - S->x[i] * S->z[i];
                if (pass) d -= dx[i] * dz[i];
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
                    dd[0] -= uv;
                    for (int i = 1; i < d; i++) dd[i] -= t1[s] * t2[s + i] + t2[s] * t1[s + i];
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
            if (pass) r5 -= dtau * dkap;
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
                sigma = (1.0 - aa) * (1.0 - aa) * (1.0 - aa);
                if (sigma < 1e-12) sigma = 1e-12;
            } else alpha = fmin(1.0, step_frac * al);
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
        /* the correctors' directions were added: the same refinement of the sum, and its step length */
        if (dual_refine(S, opt->tol, 1.0 - sigma, rD, dtau, byv, cx, ncs, dx, dy, dz, t2)) {
            double a2 = 1e300;
            a2 = cone_step(S, S->x, dx, a2); a2 = cone_step(S, S->z, dz, a2);
            if (dtau < 0) a2 = fmin(a2, -S->tau / dtau);
            if (dkap < 0) a2 = fmin(a2, -S->kap / dkap);
            alpha = fmin(alpha, fmin(1.0, step_frac * a2));
        }
        for (int j = 0; j < n; j++) { S->x[j] += alpha * dx[j]; S->z[j] += alpha * dz[j]; }
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
        if (S->ne && S->refine_total - ref0 > ne_k * (double)(S->solves - sol0)) { ne_free(S); S->ne_ok = 0; S->ne_left = it + 1; }
        if (S->verbose > 1) socp_printf("%4d %+.9e %+.9e %8.1e %8.1e %8.1e %8.1e %8.1e %5.3f %6.0e %3ld%s\n", it + 1, pobj, dobj, e_p, e_d, e_g, mu, S->tau, alpha, sigma, S->refine_total - ref0, S->F.ndyn ? (S->F.dd ? " dd*" : " *") : S->F.dd ? " dd" : "");
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
        nsmall = alpha < 1e-3 ? nsmall + 1 : 0;
        if (!(alpha > 1e-10) || nsmall >= 2 || !(S->tau > 0) || !(S->tau == S->tau)) { status = SOCP_NUMERR; break; }
    }
    /* ---- result: the point (the best one when the iteration did not end at the tolerance), rotated cones mapped back */
    R->x = (double *)xcalloc((size_t)n, sizeof(double)); R->y = (double *)xcalloc((size_t)m + 1, sizeof(double)); R->z = (double *)xcalloc((size_t)n, sizeof(double));
    if (status == SOCP_PINF || status == SOCP_DINF) {
        /* the certificate, normalised: b'y = 1 (primal infeasible) or c'x = -1 (dual infeasible) */
        for (int j = 0; j < n; j++) { R->x[j] = S->Ec[j] * S->x[j] * S->sb; R->z[j] = S->z[j] * S->scl / S->Ec[j]; }
        for (int i = 0; i < m; i++) R->y[i] = S->Dr[i] * S->y[i] * S->scl;
        const double v = status == SOCP_PINF ? dotv(m, b0, R->y) : -dotv(n, c0, R->x);
        if (v > 0) { for (int j = 0; j < n; j++) { R->x[j] = status == SOCP_DINF ? R->x[j] / v : 0.0; R->z[j] = status == SOCP_PINF ? R->z[j] / v : 0.0; } for (int i = 0; i < m; i++) R->y[i] = status == SOCP_PINF ? R->y[i] / v : 0.0; }
        R->pobj = status == SOCP_DINF ? -1.0 : 0.0; R->dobj = status == SOCP_PINF ? 1.0 : 0.0;
    } else {
        memcpy(R->x, bx, sizeof(double) * (size_t)n); memcpy(R->y, by, sizeof(double) * (size_t)m); memcpy(R->z, bz, sizeof(double) * (size_t)n);
        R->err[0] = best_e[0]; R->err[1] = best_e[1]; R->err[2] = best_e[2]; R->err[3] = best_e[3];
        if (status != SOCP_OPTIMAL && best <= 1e-6 && status != SOCP_TIME) status = SOCP_REDUCED;
        if (status == SOCP_TIME && best <= opt->tol) status = SOCP_OPTIMAL;
    }
    for (int k = P->nq; k < S->nq; k++) {
        const int j = S->qs[k];
        double a = R->x[j], b_ = R->x[j + 1]; R->x[j] = rs * (a + b_); R->x[j + 1] = rs * (a - b_);
        a = R->z[j]; b_ = R->z[j + 1]; R->z[j] = rs * (a + b_); R->z[j + 1] = rs * (a - b_);
    }
    R->status = status; R->iters = it; R->time = sc_time() - t0; R->maxerr = best; R->lnz = S->F.lnz;
    const int att = opt->attempt % 10, by_elim = opt->attempt >= 10;      /* attempt + 10: called by the substitution presolve, which prints the final status */
    const int att_line = att == 2 || (att == 1 && (status == SOCP_REDUCED || status == SOCP_NUMERR));   /* the retry wrapper prints the final status */
    if (S->verbose > 0) {
        if (att_line || !by_elim) {
            if (att_line) socp_printf("attempt %d%s: %s", att, att == 2 ? " (shorter steps)" : "", socp_status_str(status));
            else socp_printf("status: %s", socp_status_str(status));
            if (status != SOCP_PINF && status != SOCP_DINF) socp_printf(" (max rel error %.1e)", best);
            socp_printf("   iterations: %d   time: %.3fs\n", it, R->time);
        }
        if (S->verbose > 1) socp_printf("  time: setup %.3f, factorizations %.3f, %ld solves %.3f + products %.3f (%s factorization, nnz(L) = %ld%s; %.2e flops by columns)\n", S->t_setup, S->t_factor, S->nback, S->t_solve, S->t_mul, S->sc ? "supernodal" : "column", S->sc ? S->slnz : S->F.lnz, S->nfallback ? "; some by columns" : "", S->cflops);
        if (dbg_on()) socp_printf("  [factorization / solve: estimated %.1f, measured %.1f]\n", S->rr_est, (S->t_factor / fmax(S->nfact, 1)) / fmax((S->t_solve + S->t_mul) / fmax(S->nback, 1), 1e-12));
        if (S->verbose > 1 && S->ncor_total) socp_printf("  %ld centrality correctors\n", S->ncor_total);
        if (S->verbose > 1 && S->ne_nfact) socp_printf("  %ld factorizations of the normal equations (nnz(L) = %ld, %d tiny pivots)\n", S->ne_nfact, S->ne_lnz, S->ne_tiny);
        if (S->verbose > 1 && S->nbump) socp_printf("  regularization of the factored matrix raised to %.0e\n", 1e-8 * S->bump);
        if (S->verbose > 1 && S->ndd) socp_printf("  %d factorizations in double-double\n", S->ndd);
        if (S->verbose > 1 && S->dz_count) socp_printf("  %ld refinements of the direction against the dual equation\n", S->dz_count);
        if (!att_line && !by_elim && status != SOCP_PINF && status != SOCP_DINF) socp_printf("  primal obj c'x = %+.12e   dual obj b'y = %+.12e\n  errors: pinf %.1e  dinf %.1e  gap %.1e  x'z %.1e\n", R->pobj, R->dobj, R->err[0], R->err[1], R->err[2], R->err[3]);
    }
    free(px); free(py); free(qx); free(qy); free(dx); free(dy); free(dz); free(rP); free(rD); free(r3); free(rhsx); free(rhsy); free(ds); free(t1); free(t2);
    free(sv_x); free(sv_y); free(sv_z); free(dxa); free(dya); free(dza); free(ctv); free(dxs); free(dzs);
    free(xo); free(yo); free(zo); free(bx); free(by); free(bz); free(A0x); free(b0); free(c0);
    free(S->wdz); free(S->wdx); free(S->wdy); free(S->wdy2);
    socp_free(S);
    return status;
}

void socp_result_free(SocpRes *R) { free(R->x); free(R->y); free(R->z); R->x = R->y = R->z = NULL; }


/* ---- presolve: free variables eliminated by substitution ----
 * A free variable u with a_iu != 0 in row i is determined by that row:  x_u = (b_i - sum_{j != u} a_ij x_j) / a_iu.
 * Substituting it removes the row and the column: the other rows k with u get row_k - (a_ku/a_iu) row_i,
 * the costs get c_j - (c_u/a_iu) a_ij. Postsolve: x_u from the formula, y_i = c_u/a_iu - sum_k (a_ku/a_iu) y_k,
 * in the reverse order of the eliminations; the slack z of the other variables is unchanged.
 * Only eliminations without fill to speak of and with a pivot that dominates its row are done: a
 * short pivot row (at most ELIM_ROWMAX entries), (len(row) - 1)(len(col) - 1) <= len(row) + len(col),
 * |pivot| >= 0.5 max |row| and >= 0.1 max |col|. (Eliminating as much as possible was tried: on
 * nql180old the reduced problem was much harder.) */
#define ELIM_ROWMAX 4
typedef struct {
    int m, n, nf;
    int *rlen, *rcap; int **rcol; double **rval; unsigned char *ralive;
    int *fcnt, *flen, *fcap; int **flist;
    unsigned char *cgone;
    double *b, *c;
    long nnz;
    int ne, ecap; int *eu, *ei, *elen, *emn; int **ecol; double **eval; double *epiv, *ebi, *ecu; int **emk; double **emv;
} Elim;
static int row_find(const Elim *E, int i, int col) {
    int lo = 0, hi = E->rlen[i] - 1; const int *rc = E->rcol[i];
    while (lo <= hi) { const int mid = (lo + hi) >> 1; if (rc[mid] == col) return mid; if (rc[mid] < col) lo = mid + 1; else hi = mid - 1; }
    return -1;
}
static void flist_add(Elim *E, int u, int row) {
    if (E->flen[u] == E->fcap[u]) { E->fcap[u] = 2 * E->fcap[u] + 4; E->flist[u] = (int *)realloc(E->flist[u], sizeof(int) * (size_t)E->fcap[u]); }
    E->flist[u][E->flen[u]++] = row;
}
static int elim_run(Elim *E, const SocpProb *P) {
    const int m = P->m, n = P->n, nf = P->nf;
    E->m = m; E->n = n; E->nf = nf;
    E->rlen = (int *)calloc((size_t)m + 1, sizeof(int)); E->rcap = (int *)calloc((size_t)m + 1, sizeof(int));
    E->rcol = (int **)calloc((size_t)m + 1, sizeof(int *)); E->rval = (double **)calloc((size_t)m + 1, sizeof(double *));
    E->ralive = (unsigned char *)malloc((size_t)m + 1); memset(E->ralive, 1, (size_t)m + 1);
    E->cgone = (unsigned char *)calloc((size_t)n + 1, 1);
    E->b = (double *)malloc(sizeof(double) * ((size_t)m + 1)); memcpy(E->b, P->b, sizeof(double) * (size_t)m);
    E->c = (double *)malloc(sizeof(double) * ((size_t)n + 1)); memcpy(E->c, P->c, sizeof(double) * (size_t)n);
    for (int p = 0; p < P->Ap[n]; p++) E->rlen[P->Ai[p]]++;
    for (int i = 0; i < m; i++) { E->rcap[i] = E->rlen[i] + 4; E->rcol[i] = (int *)malloc(sizeof(int) * (size_t)E->rcap[i]); E->rval[i] = (double *)malloc(sizeof(double) * (size_t)E->rcap[i]); E->rlen[i] = 0; }
    for (int j = 0; j < n; j++) for (int p = P->Ap[j]; p < P->Ap[j + 1]; p++) { const int i = P->Ai[p]; E->rcol[i][E->rlen[i]] = j; E->rval[i][E->rlen[i]++] = P->Ax[p]; }
    E->nnz = P->Ap[n];
    E->fcnt = (int *)calloc((size_t)nf + 1, sizeof(int)); E->flen = (int *)calloc((size_t)nf + 1, sizeof(int)); E->fcap = (int *)calloc((size_t)nf + 1, sizeof(int));
    E->flist = (int **)calloc((size_t)nf + 1, sizeof(int *));
    for (int u = 0; u < nf; u++) for (int p = P->Ap[u]; p < P->Ap[u + 1]; p++) { flist_add(E, u, P->Ai[p]); E->fcnt[u]++; }
    int wcap = 64; int *wc = (int *)malloc(sizeof(int) * (size_t)wcap); double *wv = (double *)malloc(sizeof(double) * (size_t)wcap);
    int *rows = NULL; double *avals = NULL; int rwcap = 0;
    for (int pass = 0; pass < 3; pass++) {
        int done = 0;
        for (int u = 0; u < nf; u++) {
            if (E->cgone[u] || E->fcnt[u] == 0) continue;
            if (E->flen[u] > rwcap) { rwcap = 2 * E->flen[u] + 8; rows = (int *)realloc(rows, sizeof(int) * (size_t)rwcap); avals = (double *)realloc(avals, sizeof(double) * (size_t)rwcap); }
            int nr = 0; double amax = 0;
            for (int t = 0; t < E->flen[u]; t++) {
                const int i = E->flist[u][t];
                if (!E->ralive[i]) continue;
                int dup = 0; for (int q = 0; q < nr; q++) if (rows[q] == i) { dup = 1; break; }
                if (dup) continue;
                const int pos = row_find(E, i, u);
                if (pos < 0) continue;
                rows[nr] = i; avals[nr] = E->rval[i][pos]; if (fabs(avals[nr]) > amax) amax = fabs(avals[nr]); nr++;
                if (nr > 64) break;
            }
            if (nr <= 64) { memcpy(E->flist[u], rows, sizeof(int) * (size_t)nr); E->flen[u] = nr; E->fcnt[u] = nr; }
            if (nr == 0 || nr > 64 || !(amax > 0)) continue;
            int best = -1;
            for (int q = 0; q < nr; q++) {
                const int i = rows[q], li = E->rlen[i];
                if (li > ELIM_ROWMAX || fabs(avals[q]) < 0.1 * amax) continue;
                double rmax = 0; for (int t = 0; t < li; t++) rmax = fmax(rmax, fabs(E->rval[i][t]));
                if (fabs(avals[q]) < 0.5 * rmax) continue;
                if ((double)(li - 1) * (nr - 1) > (double)(li + nr)) continue;
                if (best < 0 || li < E->rlen[rows[best]] || (li == E->rlen[rows[best]] && fabs(avals[q]) > fabs(avals[best]))) best = q;
            }
            if (best < 0) continue;
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
            const int *ic = E->rcol[pi]; const double *iv = E->rval[pi];
            const double cu = E->c[u] / piv;
            if (cu != 0) for (int t = 0; t < li; t++) if (ic[t] != u) E->c[ic[t]] -= cu * iv[t];
            int mq = 0;
            for (int q = 0; q < nr; q++) {
                if (q == best) continue;
                const int k = rows[q]; const double mk = avals[q] / piv;
                E->emk[e][mq] = k; E->emv[e][mq++] = mk;
                const int lk = E->rlen[k];
                if (lk + li > wcap) { wcap = 2 * (lk + li) + 16; wc = (int *)realloc(wc, sizeof(int) * (size_t)wcap); wv = (double *)realloc(wv, sizeof(double) * (size_t)wcap); }
                const int *kc = E->rcol[k]; const double *kv = E->rval[k];
                int a = 0, b_ = 0, w = 0;
                while (a < lk || b_ < li) {
                    if (b_ >= li || (a < lk && kc[a] < ic[b_])) { wc[w] = kc[a]; wv[w++] = kv[a]; a++; }
                    else if (a >= lk || ic[b_] < kc[a]) {
                        const int col = ic[b_];
                        wc[w] = col; wv[w++] = -mk * iv[b_]; b_++;
                        if (col < nf) { E->fcnt[col]++; flist_add(E, col, k); }
                    } else {
                        const int col = kc[a]; const double v = kv[a] - mk * iv[b_]; a++; b_++;
                        if (col == u) continue;
                        if (v != 0) { wc[w] = col; wv[w++] = v; }
                        else if (col < nf) E->fcnt[col]--;
                    }
                }
                if (w > E->rcap[k]) { E->rcap[k] = w + w / 2 + 4; E->rcol[k] = (int *)realloc(E->rcol[k], sizeof(int) * (size_t)E->rcap[k]); E->rval[k] = (double *)realloc(E->rval[k], sizeof(double) * (size_t)E->rcap[k]); }
                memcpy(E->rcol[k], wc, sizeof(int) * (size_t)w); memcpy(E->rval[k], wv, sizeof(double) * (size_t)w);
                E->nnz += w - lk; E->rlen[k] = w;
                E->b[k] -= mk * E->b[pi];
            }
            for (int t = 0; t < li; t++) if (ic[t] < nf) E->fcnt[ic[t]]--;
            E->ralive[pi] = 0; E->cgone[u] = 1; E->nnz -= li;
            E->rcol[pi] = NULL; E->rval[pi] = NULL; E->rlen[pi] = 0;
            done++;
        }
        if (!done) break;
    }
    free(wc); free(wv); free(rows); free(avals);
    return E->ne;
}
static void elim_free(Elim *E) {
    for (int i = 0; i < E->m; i++) { free(E->rcol[i]); free(E->rval[i]); }
    for (int u = 0; u < E->nf; u++) free(E->flist[u]);
    for (int e = 0; e < E->ne; e++) { free(E->ecol[e]); free(E->eval[e]); free(E->emk[e]); free(E->emv[e]); }
    free(E->rlen); free(E->rcap); free(E->rcol); free(E->rval); free(E->ralive); free(E->fcnt); free(E->flen); free(E->fcap); free(E->flist); free(E->cgone); free(E->b); free(E->c);
    free(E->eu); free(E->ei); free(E->elen); free(E->emn); free(E->ecol); free(E->eval); free(E->epiv); free(E->ebi); free(E->ecu); free(E->emk); free(E->emv);
}
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
    Q.m = m2; Q.n = n2; Q.Ap = Ap; Q.Ai = Ai; Q.Ax = Ax; Q.b = b2; Q.c = c2; Q.nf = nf2;
    if (opt && opt->verbose > 0) socp_printf("presolve: %d free variables eliminated by substitution (rows %d -> %d, nonzeros %d -> %d)\n", E->ne, m, m2, P->Ap[n], Ap[n2]);
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
    } else st = socp_solve_core(&Q, &o2, R);
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
        socp_printf("status: %s", socp_status_str(st2));
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
int socp_solve(const SocpProb *P, const SocpOpts *opt, SocpRes *R) {
    const double t0 = sc_time();
    SocpOpts o1; if (opt) o1 = *opt; else socp_default_opts(&o1);
    const int retry = !(o1.step > 0) && !getenv("BRISK_CONENORETRY");
    o1.attempt = retry ? 1 : 0;
    int st = socp_solve_pre(P, &o1, R);
    if (!retry || (st != SOCP_REDUCED && st != SOCP_NUMERR) || *socp_stop) return st;
    SocpOpts o2 = o1; o2.step = 0.8; o2.attempt = 2;
    int second = 0;
    if (!(o1.timelimit > 0) || sc_time() - t0 < 0.4 * o1.timelimit) {
        if (o1.timelimit > 0) o2.timelimit = o1.timelimit - (sc_time() - t0);
        SocpRes R2; memset(&R2, 0, sizeof R2);
        const int st2 = socp_solve_pre(P, &o2, &R2);
        const int it = R->iters + R2.iters;
        const int better = st2 == SOCP_OPTIMAL || st2 == SOCP_PINF || st2 == SOCP_DINF || ((st2 == SOCP_REDUCED || st2 == SOCP_NUMERR) && R2.maxerr < R->maxerr);
        if (better) { socp_result_free(R); *R = R2; st = st2; second = 1; } else socp_result_free(&R2);
        R->iters = it;
    }
    R->time = sc_time() - t0;
    if (o1.verbose > 0) {
        const int cert = st == SOCP_PINF || st == SOCP_DINF;
        socp_printf("status: %s", socp_status_str(st));
        if (!cert) socp_printf(" (max rel error %.1e)", R->maxerr);
        socp_printf("   iterations: %d   time: %.3fs%s\n", R->iters, R->time, second ? "   (second attempt)" : "");
        if (!cert) socp_printf("  primal obj c'x = %+.12e   dual obj b'y = %+.12e\n  errors: pinf %.1e  dinf %.1e  gap %.1e  x'z %.1e\n", R->pobj, R->dobj, R->err[0], R->err[1], R->err[2], R->err[3]);
    }
    return st;
}
