/* momaff.c - moment relaxations over an ellipsoid, solved in the coordinates of the unit ball.
 *
 * A moment relaxation of order d of  min f(x) s.t. g(x) >= 0,  g quadratic with a negative definite
 * Hessian (an ellipsoid), given as an SDP: the moment matrix M(y) of the monomials of degree <= d and
 * the localizing matrix of g. With x = x0 + L u and g(x0 + L u) = c (1 - |u|^2) the monomials satisfy
 * v(x) = B v(u) (B lower block-triangular by degree), the moments m_x = T m_u, M_x = B M_u B' and the
 * localizing matrix c B_l L_u B_l', B_l the leading block of B. The relaxation in u is the same SDP up
 * to these congruences and the linear change of the moments: the same optimal value (plus a constant),
 * and solutions map back exactly. In u the constraint set is the unit ball: the moment block's data
 * are the same, the localizing block is that of 1 - |u|^2 (sparser), and the objective is f(x0 + L u).
 * On a user's relaxations the iterations fell from 9-12 to 5-6 on the high-order ones (the monomial
 * basis centred away from the set, with axes 0.8-1.8, costs the centred start its centrality).
 *
 * The structure is read from the data and checked entry by entry (momaff_apply returns NULL
 * otherwise): two SDP blocks; the larger a moment matrix (every entry in one constraint, the same
 * value in all the entries of a constraint, the constant monomial first with its entry in C only),
 * its monomials recovered from the products of the degree-one ones; the smaller the localizing matrix
 * of a quadratic g on the leading monomials of the moment basis, every entry present and equal to
 * its coefficient of g times the moment block's value of the constraint. Relaxations of order d <= 2
 * are left alone (no gain measured there). */
#include "brisk.h"
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>

static const double DONE = 1.0, DZERO = 0.0;

struct MomAff {
    int n, nl, N, m, k0, kl;
    double c;            /* g(x0): g(x0 + L u) = c (1 - |u|^2) */
    double cm00;         /* C(0,0) of the moment block (the constant moment's coefficient) */
    double *vm;          /* the moment block's value of each constraint */
    double *B, *Bi;      /* n x n, column-major: v(x) = B v(u), v(u) = Bi v(x) */
    int *rep;            /* a moment-block entry (i <= j) of each constraint: rep[2t], rep[2t+1] */
    int *tc;             /* the constraint of each moment-block entry (r <= c, at r + c n; -1 for (0,0)) */
};

static double sp_get00(const SpSym *A) {
    for (int q = 0; q < A->nnz; q++) if (A->row[q] == 0 && A->col[q] == 0) return A->val[q];
    return 0.0;
}

#define MA_HASH(K_, N_, HS_) ({ unsigned h_ = 2166136261u; for (int i_ = 0; i_ < (N_); i_++) { h_ ^= (K_)[i_]; h_ *= 16777619u; } (int)(h_ % (unsigned)(HS_)); })

/* the expansion of the monomials E_j (graded order) of x = a + M u on the monomials in u: column-major
 * n x n, row j the coefficients of x^{E_j}; ht/hs the hash of E, nx the index of E_j + e_i */
static double *ma_expand(int n, int N, const unsigned char *E, const int *ht, int hs, const int *nx, const double *a, const double *M) {
    double *Bm = calloc((size_t)n * n, sizeof(double));
    unsigned char key[64];
    Bm[0] = 1.0;
    for (int j = 1; j < n; j++) {
        int iv = 0; while (E[(size_t)j * N + iv] == 0) iv++;
        memcpy(key, E + (size_t)j * N, N); key[iv]--;
        int h = MA_HASH(key, N, hs), bt = -1;
        while (ht[h] >= 0) { if (!memcmp(E + (size_t)ht[h] * N, key, N)) { bt = ht[h]; break; } h = (h + 1) % hs; }
        if (bt < 0 || bt >= j) { free(Bm); return NULL; }
        for (int g2 = 0; g2 < n; g2++) {
            const double v = Bm[bt + (size_t)g2 * n];
            if (v == 0.0) continue;
            Bm[j + (size_t)g2 * n] += a[iv] * v;
            for (int jj = 0; jj < N; jj++) { const int to = nx[(size_t)g2 * N + jj]; if (to < 0) { free(Bm); return NULL; } Bm[j + (size_t)to * n] += M[iv + (size_t)jj * N] * v; }
        }
    }
    return Bm;
}

typedef struct { int r, c; double v; } MAEnt;
static int ma_cmp(const void *a, const void *b) {
    const MAEnt *x = a, *y = b;
    if (x->c != y->c) return x->c - y->c;
    return x->r - y->r;
}

int g_momaff_off = 0;     /* set by the driver for the re-solve of the problem as given */

MomAff *momaff_apply(Problem *P, int verbose) {
    const int nb = P->nblk, m = P->m;
    if (g_momaff_off || getenv("BRISK_NOAFF") || nb != 2 || P->blk[0].type != BLK_SDP || P->blk[1].type != BLK_SDP) return NULL;
    const int k0 = P->blk[0].n >= P->blk[1].n ? 0 : 1, kl = 1 - k0;
    Block *M0 = &P->blk[k0], *Lb = &P->blk[kl];
    const int n = M0->n, nl = Lb->n;
    if (nl >= n || n < 3 || n > 4000 || M0->ncon < m || M0->ncon < n) return NULL;
    /* (cheap tests first: C of both blocks only at (0,0), as many constraints as distinct entries at most) */
    for (int b = 0; b < 2; b++) { const SpSym *C = b ? &Lb->C : &M0->C; for (int q = 0; q < C->nnz; q++) if ((C->row[q] || C->col[q]) && C->val[q] != 0.0) return NULL; }
    if ((double)m > 0.5 * (double)n * (n + 1)) return NULL;
    MomAff *R = NULL;
    int ok = 1, N = 0, ng = 0, d = 0;
    int *tc = malloc(sizeof(int) * (size_t)n * n), *tj = malloc(sizeof(int) * n), *deg = calloc(n, sizeof(int));
    double *tv = malloc(sizeof(double) * (size_t)n * n), *vm = calloc(m + 1, sizeof(double));
    char *inner = calloc(m + 1, 1), *ek = calloc(n, 1), *mk = calloc(m + 1, 1);
    unsigned char *E = NULL, *Mo = NULL, *gx = NULL;
    double *gc = NULL, *Bm = NULL, *Bi = NULL, *G = NULL, *bv = NULL, *x0 = NULL, *w = NULL, *L = NULL, *H = NULL;
    int *ht = NULL, *nx = NULL, *rep = NULL, *mh = NULL;
    for (size_t q = 0; q < (size_t)n * n; q++) tc[q] = -1;
    for (int t = 0; t < M0->ncon && ok; t++) {
        const SpSym *A = &M0->A[t]; const int gt = M0->con[t];
        for (int q = 0; q < A->nnz; q++) {
            int r = A->row[q], c = A->col[q]; if (r > c) { const int x = r; r = c; c = x; }
            const size_t id = r + (size_t)c * n;
            if (tc[id] >= 0) { ok = 0; break; }
            tc[id] = gt; tv[id] = A->val[q];
        }
    }
    const double cm00 = sp_get00(&M0->C), cl00 = sp_get00(&Lb->C);
    if (!ok || tc[0] >= 0 || cm00 == 0.0) goto out;
    for (int c = 1; c < n && ok; c++) for (int r = 1; r <= c; r++) { const int t = tc[r + (size_t)c * n]; if (t < 0) { ok = 0; break; } inner[t] = 1; }
    tj[0] = -1;
    for (int j = 1; j < n && ok; j++) { tj[j] = tc[(size_t)j * n]; if (tj[j] < 0) ok = 0; }
    if (!ok) goto out;
    for (int j = 1; j < n; j++) if (!inner[tj[j]]) N++;
    if (N < 1 || N > 64) goto out;
    E = calloc((size_t)n * N, 1); Mo = calloc((size_t)m * N + 1, 1);
    ek[0] = 1;
    { int v = 0; for (int j = 1; j < n; j++) if (!inner[tj[j]]) { E[(size_t)j * N + v++] = 1; ek[j] = 1; mk[tj[j]] = 1; memcpy(Mo + (size_t)tj[j] * N, E + (size_t)j * N, N); } }
    for (int ch = 1; ch && ok; ) {
        ch = 0;
        for (int c = 1; c < n && ok; c++) for (int r = 1; r <= c; r++) {
            const int t = tc[r + (size_t)c * n];
            if (mk[t] || !ek[r] || !ek[c]) continue;
            for (int i = 0; i < N; i++) { const int s = E[(size_t)r * N + i] + E[(size_t)c * N + i]; if (s > 255) { ok = 0; break; } Mo[(size_t)t * N + i] = (unsigned char)s; }
            mk[t] = 1; ch = 1;
        }
        for (int j = 1; j < n; j++) if (!ek[j] && mk[tj[j]]) { memcpy(E + (size_t)j * N, Mo + (size_t)tj[j] * N, N); ek[j] = 1; ch = 1; }
    }
    for (int j = 0; j < n && ok; j++) if (!ek[j]) ok = 0;
    for (int t = 0; t < m && ok; t++) if (!mk[t]) ok = 0;                 /* every moment in the moment matrix */
    if (!ok) goto out;
    rep = malloc(sizeof(int) * 2 * (size_t)m);
    for (int t = 0; t < m; t++) rep[2 * t] = -1;
    for (int c = 0; c < n && ok; c++) for (int r = 0; r <= c; r++) {
        if (r == 0 && c == 0) continue;
        const int t = tc[r + (size_t)c * n];
        for (int i = 0; i < N; i++) if (Mo[(size_t)t * N + i] != E[(size_t)r * N + i] + E[(size_t)c * N + i]) { ok = 0; break; }
        const double v = tv[r + (size_t)c * n];
        if (vm[t] == 0.0) vm[t] = v; else if (fabs(v - vm[t]) > 1e-12 * fabs(vm[t])) ok = 0;
        if (rep[2 * t] < 0) { rep[2 * t] = r; rep[2 * t + 1] = c; }
    }
    for (int t = 0; t < m && ok; t++) if (vm[t] == 0.0) ok = 0;
    for (int q = 0; q < M0->C.nnz && ok; q++) if ((M0->C.row[q] != 0 || M0->C.col[q] != 0) && M0->C.val[q] != 0.0) ok = 0;
    if (!ok) goto out;
    for (int j = 0; j < n; j++) { int s = 0; for (int i = 0; i < N; i++) s += E[(size_t)j * N + i]; deg[j] = s; if (s > d) d = s; }
    for (int j = 1; j < n && ok; j++) if (deg[j] < deg[j - 1]) ok = 0;       /* (graded order) */
    if (!ok || d <= 2) goto out;
    const int dl = deg[nl - 1];
    if (deg[nl] <= dl) goto out;
    /* g: the (0,0) entries over the moment values, its constant from C */
    gx = calloc((size_t)(Lb->ncon + 1) * N, 1); gc = calloc(Lb->ncon + 1, sizeof(double));
    gc[ng++] = cl00 / cm00;
    for (int t = 0; t < Lb->ncon; t++) {
        const double v = sp_get00(&Lb->A[t]);
        if (v == 0.0) continue;
        const int gt = Lb->con[t];
        memcpy(gx + (size_t)ng * N, Mo + (size_t)gt * N, N); gc[ng++] = v / vm[gt];
    }
    for (int q = 1; q < ng && ok; q++) { int s = 0; for (int i = 0; i < N; i++) s += gx[(size_t)q * N + i]; if (s < 1 || s > 2) ok = 0; }
    if (!ok || gc[0] <= 0.0) goto out;
    {   /* every entry of the localizing block, and none missing */
        double gmax = 0; for (int q = 0; q < ng; q++) gmax = fmax(gmax, fabs(gc[q]));
        long cnt = 0, exp_cnt = 0;
        unsigned char dlt[64];
        for (int t = 0; t < Lb->ncon && ok; t++) {
            const SpSym *A = &Lb->A[t]; const int gt = Lb->con[t];
            for (int q = 0; q < A->nnz && ok; q++) {
                int r = A->row[q], c = A->col[q]; if (r > c) { const int x = r; r = c; c = x; }
                for (int i = 0; i < N; i++) { const int v2 = (int)Mo[(size_t)gt * N + i] - E[(size_t)r * N + i] - E[(size_t)c * N + i]; if (v2 < 0) { ok = 0; break; } dlt[i] = (unsigned char)v2; }
                if (!ok) break;
                int f = -1; for (int q2 = 0; q2 < ng; q2++) if (!memcmp(gx + (size_t)q2 * N, dlt, N)) { f = q2; break; }
                if (f < 0 || fabs(A->val[q] / vm[gt] - gc[f]) > 1e-9 * gmax) ok = 0;
                cnt++;
            }
        }
        for (int c = 0; c < nl && ok; c++) for (int r = 0; r <= c; r++) exp_cnt += (r == 0 && c == 0) ? ng - 1 : ng;
        if (cnt != exp_cnt) ok = 0;
        for (int q = 0; q < Lb->C.nnz && ok; q++) if ((Lb->C.row[q] != 0 || Lb->C.col[q] != 0) && Lb->C.val[q] != 0.0) ok = 0;
    }
    if (!ok) goto out;
    /* the ellipsoid: g = g0 + b'x + x'Gx, x0 = -G^{-1} b / 2, c = g(x0), -G/c = V diag(w) V', L = V diag(w)^{-1/2} */
    G = calloc((size_t)N * N, sizeof(double)); bv = calloc(N, sizeof(double)); x0 = calloc(N, sizeof(double));
    w = malloc(sizeof(double) * N); L = malloc(sizeof(double) * (size_t)N * N); H = malloc(sizeof(double) * (size_t)N * N);
    for (int q = 1; q < ng; q++) {
        int i1 = -1, i2 = -1;
        for (int i = 0; i < N; i++) for (int e = 0; e < gx[(size_t)q * N + i]; e++) { if (i1 < 0) i1 = i; else i2 = i; }
        if (i2 < 0) bv[i1] += gc[q];
        else if (i1 == i2) G[i1 + (size_t)i1 * N] += gc[q];
        else { G[i1 + (size_t)i2 * N] += 0.5 * gc[q]; G[i2 + (size_t)i1 * N] += 0.5 * gc[q]; }
    }
    double cval = gc[0];
    {
        int info = 0, one = 1;
        for (size_t i = 0; i < (size_t)N * N; i++) H[i] = -G[i];
        memcpy(x0, bv, sizeof(double) * N);
        BL(dpotrf_)("L", &N, H, &N, &info);
        if (info) goto out;
        BL(dpotrs_)("L", &N, &one, H, &N, x0, &N, &info);
        if (info) goto out;
        for (int i = 0; i < N; i++) x0[i] *= 0.5;                            /* (-G) x0 = b / 2 */
        for (int i = 0; i < N; i++) { cval += bv[i] * x0[i]; for (int j = 0; j < N; j++) cval += x0[i] * G[i + (size_t)j * N] * x0[j]; }
        if (!(cval > 0)) goto out;
        for (size_t i = 0; i < (size_t)N * N; i++) H[i] = -G[i] / cval;
        double wq; int lw = -1;
        BL(dsyev_)("V", "L", &N, H, &N, w, &wq, &lw, &info);
        lw = (int)wq + 1; double *wk = malloc(sizeof(double) * lw);
        BL(dsyev_)("V", "L", &N, H, &N, w, wk, &lw, &info);
        free(wk);
        if (info || !(w[0] > 0)) goto out;
        double dev = 0;
        for (int j = 0; j < N; j++) for (int i = 0; i < N; i++) L[i + (size_t)j * N] = H[i + (size_t)j * N] / sqrt(w[j]);
        for (int i = 0; i < N; i++) { dev = fmax(dev, fabs(x0[i])); dev = fmax(dev, fabs(w[i] - 1.0)); }
        if (dev <= 1e-12) goto out;                                          /* the unit ball centred at 0 */
    }
    {   /* B: row j (monomial E_j in x) = its coefficients on the monomials in u, x^a = (x0_i + L_i u) x^{a - e_i} */
        const int hs = 4 * n; ht = malloc(sizeof(int) * hs); for (int i = 0; i < hs; i++) ht[i] = -1;
        for (int j = 0; j < n; j++) { int h = MA_HASH(E + (size_t)j * N, N, hs); while (ht[h] >= 0) h = (h + 1) % hs; ht[h] = j; }
        unsigned char key[64];
#define MA_FIND(K_) ({ int h2_ = MA_HASH(K_, N, hs), r_ = -1; while (ht[h2_] >= 0) { if (!memcmp(E + (size_t)ht[h2_] * N, (K_), N)) { r_ = ht[h2_]; break; } h2_ = (h2_ + 1) % hs; } r_; })
        nx = malloc(sizeof(int) * (size_t)n * N);
        for (int j = 0; j < n; j++) for (int i = 0; i < N; i++) {
            if (deg[j] >= d) { nx[(size_t)j * N + i] = -1; continue; }
            memcpy(key, E + (size_t)j * N, N); key[i]++; nx[(size_t)j * N + i] = MA_FIND(key);
            if (nx[(size_t)j * N + i] < 0) ok = 0;
        }
        if (!ok) goto out;
        Bm = ma_expand(n, N, E, ht, hs, nx, x0, L);
        /* the inverse map u = L^{-1} (x - x0): L^{-1} = diag(w)^{1/2} V' */
        double *Li = malloc(sizeof(double) * (size_t)N * N), *ai = calloc(N, sizeof(double));
        for (int j = 0; j < N; j++) for (int i = 0; i < N; i++) Li[i + (size_t)j * N] = sqrt(w[i]) * H[j + (size_t)i * N];
        for (int i = 0; i < N; i++) { double sacc = 0; for (int j = 0; j < N; j++) sacc += Li[i + (size_t)j * N] * x0[j]; ai[i] = -sacc; }
        Bi = ma_expand(n, N, E, ht, hs, nx, ai, Li);
        free(Li); free(ai);
        if (!Bm || !Bi) ok = 0;
#undef MA_FIND
        if (!ok) goto out;
    }
    {   /* the objective: m_t = -y_t vm[t] / cm00 (the normalized moments), b'y = beta'm with beta_t = -b_t cm00 / vm[t];
         * beta'm_x = beta'(T m_u + t0): the new b from T'beta, the constant beta't0 into obj_off */
        const int hs2 = 4 * m + 8; mh = malloc(sizeof(int) * hs2); for (int i = 0; i < hs2; i++) mh[i] = -1;
        unsigned char zero[64]; memset(zero, 0, sizeof(zero));
        for (int t = 0; t < m; t++) { int h = MA_HASH(Mo + (size_t)t * N, N, hs2); while (mh[h] >= 0) h = (h + 1) % hs2; mh[h] = t; }
        double *tb = calloc(m + 1, sizeof(double)), t0 = 0;
        unsigned char key[64];
        for (int t = 0; t < m && ok; t++) {
            if (P->b[t] == 0.0) continue;
            const double beta = -P->b[t] * cm00 / vm[t];
            const int r = rep[2 * t], c = rep[2 * t + 1];
            for (int g1 = 0; g1 < n && ok; g1++) {
                const double v1 = Bm[r + (size_t)g1 * n];
                if (v1 == 0.0) continue;
                for (int g2 = 0; g2 < n; g2++) {
                    const double v2 = Bm[c + (size_t)g2 * n];
                    if (v2 == 0.0) continue;
                    for (int i = 0; i < N; i++) key[i] = (unsigned char)(E[(size_t)g1 * N + i] + E[(size_t)g2 * N + i]);
                    if (!memcmp(key, zero, N)) { t0 += beta * v1 * v2; continue; }
                    int h = MA_HASH(key, N, hs2), s = -1;
                    while (mh[h] >= 0) { if (!memcmp(Mo + (size_t)mh[h] * N, key, N)) { s = mh[h]; break; } h = (h + 1) % hs2; }
                    if (s < 0) { ok = 0; break; }
                    tb[s] += beta * v1 * v2;
                }
            }
        }
        if (!ok) { free(tb); goto out; }
        /* the u-problem: b_s = -tb_s vm[s] / cm00, the constant into obj_off */
        for (int s = 0; s < m; s++) { P->b[s] = -tb[s] * vm[s] / cm00; P->b0[s] = P->b[s]; }
        P->obj_off += t0;
        free(tb);
    }
    {   /* the localizing block of 1 - |u|^2: entries (r, c) of the constraint E_r + E_c + delta, delta in {0, 2 e_i},
         * values vm[s] (delta = 0) and -vm[s]; C(0,0) = cm00 */
        size_t cap = 0, ne = 0;
        for (int c = 0; c < nl; c++) cap += (size_t)(c + 1) * (N + 1);
        int *es = malloc(sizeof(int) * cap); MAEnt *ee = malloc(sizeof(MAEnt) * cap);
        unsigned char key[64];
        for (int c = 0; c < nl && ok; c++) for (int r = 0; r <= c && ok; r++) for (int q = -1; q < N; q++) {
            for (int i = 0; i < N; i++) key[i] = (unsigned char)(E[(size_t)r * N + i] + E[(size_t)c * N + i] + (q == i ? 2 : 0));
            if (r == 0 && c == 0 && q < 0) continue;                          /* (the constant: C) */
            int h = MA_HASH(key, N, 4 * m + 8), s = -1;
            while (mh[h] >= 0) { if (!memcmp(Mo + (size_t)mh[h] * N, key, N)) { s = mh[h]; break; } h = (h + 1) % (4 * m + 8); }
            if (s < 0) { ok = 0; break; }
            es[ne] = s; ee[ne].r = r; ee[ne].c = c; ee[ne].v = q < 0 ? vm[s] : -vm[s]; ne++;
        }
        if (!ok) { free(es); free(ee); goto out; }
        /* by constraint */
        int *cntc = calloc(m + 1, sizeof(int));
        for (size_t q = 0; q < ne; q++) cntc[es[q]]++;
        int ncon = 0; for (int s = 0; s < m; s++) if (cntc[s]) ncon++;
        int *off = calloc(m + 2, sizeof(int)); for (int s = 0; s < m; s++) off[s + 1] = off[s] + cntc[s];
        MAEnt *by = malloc(sizeof(MAEnt) * (ne + 1)); int *pos = malloc(sizeof(int) * (m + 1)); memcpy(pos, off, sizeof(int) * m);
        for (size_t q = 0; q < ne; q++) by[pos[es[q]]++] = ee[q];
        for (int t = 0; t < Lb->ncon; t++) { SpSym *A = &Lb->A[t]; free(A->row); free(A->col); free(A->val); free(A->fr); free(A->fc); free(A->fv); free(A->rows); }
        free(Lb->A); free(Lb->con);
        Lb->ncon = ncon; Lb->con = malloc(sizeof(int) * (ncon + 1)); Lb->A = calloc(ncon + 1, sizeof(SpSym));
        int k = 0;
        for (int s = 0; s < m; s++) {
            if (!cntc[s]) continue;
            MAEnt *Es = by + off[s]; const int cnt = cntc[s];
            qsort(Es, cnt, sizeof(MAEnt), ma_cmp);
            SpSym *A = &Lb->A[k]; Lb->con[k] = s; k++;
            A->nnz = cnt; A->row = malloc(sizeof(int) * cnt); A->col = malloc(sizeof(int) * cnt); A->val = malloc(sizeof(double) * cnt);
            for (int q = 0; q < cnt; q++) { A->row[q] = Es[q].r; A->col[q] = Es[q].c; A->val[q] = Es[q].v; }
            spsym_finish(A, nl, 0);
        }
        {   SpSym *C = &Lb->C; free(C->row); free(C->col); free(C->val); free(C->fr); free(C->fc); free(C->fv); free(C->rows); memset(C, 0, sizeof(*C));
            C->nnz = 1; C->row = malloc(sizeof(int)); C->col = malloc(sizeof(int)); C->val = malloc(sizeof(double));
            C->row[0] = 0; C->col[0] = 0; C->val[0] = cm00; spsym_finish(C, nl, 0); }
        free(cntc); free(off); free(by); free(pos); free(es); free(ee);
    }
    R = calloc(1, sizeof(MomAff));
    R->n = n; R->nl = nl; R->N = N; R->m = m; R->k0 = k0; R->kl = kl; R->c = cval; R->cm00 = cm00;
    R->vm = vm; vm = NULL; R->B = Bm; Bm = NULL; R->Bi = Bi; Bi = NULL; R->rep = rep; rep = NULL; R->tc = tc; tc = NULL;
    if (verbose >= 0) {
        double r0 = 0, amin = 1e300, amax = 0;
        for (int i = 0; i < N; i++) { r0 += x0[i] * x0[i]; amin = fmin(amin, 1 / sqrt(w[i])); amax = fmax(amax, 1 / sqrt(w[i])); }
        printf("presolve: moment relaxation over an ellipsoid (%d variables, order %d; centre at distance %.3g, axes %.3g-%.3g): solved in the coordinates of the unit ball\n", N, d, sqrt(r0), amin, amax);
        if (getenv("BRISK_AFFDBG")) printf("   [|B|_F |B^-1|_F %.3g]\n", momaff_amplification(R));
    }
out:
    free(tc); free(tj); free(deg); free(tv); free(vm); free(inner); free(ek); free(mk);
    free(E); free(Mo); free(gx); free(gc); free(Bm); free(Bi); free(ht); free(nx); free(rep); free(mh);
    free(G); free(bv); free(x0); free(w); free(L); free(H);
    return R;
}

/* the solution of the problem in u back to x: y from the moment matrix B Z_u B' (Z_u = C - A'y_u on
 * the moment block, whose data are the same in both), X = Bi' X_u Bi on the moment block and
 * (1/c) Bi_l' X_u Bi_l on the localizing one (Bi_l the leading block of Bi). Xp may be NULL. */
void momaff_unmap(const MomAff *R, double **Xp, const double *yp, double ***Xo, double *yo) {
    const int n = R->n, nl = R->nl, m = R->m;
    double *Z = malloc(sizeof(double) * (size_t)n * n), *T = malloc(sizeof(double) * (size_t)n * n);
    for (int c = 0; c < n; c++) for (int r = 0; r <= c; r++) {
        const int t = R->tc[r + (size_t)c * n];
        const double v = t < 0 ? R->cm00 : -yp[t] * R->vm[t];
        Z[r + (size_t)c * n] = v; Z[c + (size_t)r * n] = v;
    }
    BL(dgemm_)("N", "N", &n, &n, &n, &DONE, R->B, &n, Z, &n, &DZERO, T, &n);
    BL(dgemm_)("N", "T", &n, &n, &n, &DONE, T, &n, R->B, &n, &DZERO, Z, &n);
    for (int t = 0; t < m; t++) { const int r = R->rep[2 * t], c = R->rep[2 * t + 1]; yo[t] = -Z[r + (size_t)c * n] / R->vm[t]; }
    if (Xp && Xo) {
        double **X = calloc(3, sizeof(double *));
        for (int b = 0; b < 2; b++) {
            const int k = b == 0 ? R->k0 : R->kl, nk = b == 0 ? n : nl;
            const double sc = b == 0 ? 1.0 : 1.0 / R->c;
            /* Bi_k' X_u Bi_k with Bi_k the leading nk x nk block of Bi (ld n) */
            BL(dgemm_)("T", "N", &nk, &nk, &nk, &sc, R->Bi, &n, Xp[k], &nk, &DZERO, T, &nk);
            X[k] = malloc(sizeof(double) * (size_t)nk * nk + 8);
            BL(dgemm_)("N", "N", &nk, &nk, &nk, &DONE, T, &nk, R->Bi, &n, &DZERO, X[k], &nk);
            for (int c = 0; c < nk; c++) for (int r = 0; r < c; r++) { const double v = 0.5 * (X[k][r + (size_t)c * nk] + X[k][c + (size_t)r * nk]); X[k][r + (size_t)c * nk] = v; X[k][c + (size_t)r * nk] = v; }
        }
        *Xo = X;
    }
    free(Z); free(T);
}

/* |B|_F |B^{-1}|_F: the factor by which the map back can enlarge the errors */
double momaff_amplification(const MomAff *R) {
    double fb = 0, fbi = 0;
    for (size_t i = 0; i < (size_t)R->n * R->n; i++) { fb += R->B[i] * R->B[i]; fbi += R->Bi[i] * R->Bi[i]; }
    return sqrt(fb) * sqrt(fbi);
}

void momaff_free(MomAff *R) {
    if (!R) return;
    free(R->vm); free(R->B); free(R->Bi); free(R->rep); free(R->tc); free(R);
}
