/*
 * Dictionary (shared-vector) Schur route.
 *
 * Many LMIs (Lyapunov terms A'P + PA, KYP-type blocks) have constraint matrices
 *     A_t = sum_{h in H_t} (e_h r_h' + r_h e_h')
 * with a few "hub" rows H_t, where the hub rows r_h are copies of a small set of
 * shared dense vectors v_c except at hub positions. Then
 *     A_t = D S_t D',   D = [unit columns e_j | shared columns v_c]   (n x q)
 * with S_t having a handful of entries, and
 *     tr(A_t X A_u Zi) = tr(S_t P S_u Q),   P = D'XD,  Q = D'Zi D   (q x q),
 * so the ordinary sparse Schur kernels apply to a virtual q x q block.
 *
 * Detection is linear in the number of nonzeros: cheap gates, a greedy vertex
 * cover (<= DICT_HMAX hubs), hashed clustering of hub rows with exact
 * compatibility checks, and an entry-by-entry verification of every
 * reconstructed constraint (failures fall back to unit form, which is exact).
 */
#include "brisk.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>

#define DICT_HMAX 4
#define DICT_KEY 4
#define DICT_SAMPLE 16

typedef struct { int t, h, start, len, nh; int hub[DICT_HMAX]; int cl; double beta; } HubRow;
typedef struct { uint64_t key; int row; } KeyRow;
typedef struct { int r, c; double v; } Trip;

static void *xm(size_t n) { void *p = malloc(n ? n : 1); if (!p) { fprintf(stderr, "brisk: out of memory\n"); exit(1); } return p; }
static void *xc(size_t n, size_t s) { void *p = calloc(n ? n : 1, s); if (!p) { fprintf(stderr, "brisk: out of memory\n"); exit(1); } return p; }

/* Greedy vertex cover (<= DICT_HMAX nodes) of the pattern of S (upper entries;
 * a diagonal entry is a loop and forces its node). cnt, mark: zeroed n-arrays,
 * zeroed again on return. Returns the number of hubs or -1. */
static int cover(const SpSym *S, int *H, int *cnt, int *mark) {
    int nh = 0;
    for (int q = 0; q < S->nnz; q++) {
        int r = S->row[q];
        if (r == S->col[q] && !mark[r]) {
            if (nh == DICT_HMAX) { for (int k = 0; k < nh; k++) mark[H[k]] = 0; return -1; }
            mark[r] = 1;
            H[nh++] = r;
        }
    }
    for (;;) {
        int any = 0;
        for (int q = 0; q < S->nnz; q++) {
            int r = S->row[q], c = S->col[q];
            if (mark[r] || mark[c]) continue;
            any = 1; cnt[r]++; cnt[c]++;
        }
        if (!any) break;
        int best = -1;
        for (int q = 0; q < S->nnz; q++) {
            int r = S->row[q], c = S->col[q];
            if (mark[r] || mark[c]) continue;
            if (best < 0 || cnt[r] > cnt[best] || (cnt[r] == cnt[best] && r < best)) best = r;
            if (cnt[c] > cnt[best] || (cnt[c] == cnt[best] && c < best)) best = c;
        }
        for (int q = 0; q < S->nnz; q++) { cnt[S->row[q]] = 0; cnt[S->col[q]] = 0; }
        if (nh == DICT_HMAX) { for (int k = 0; k < nh; k++) mark[H[k]] = 0; return -1; }
        mark[best] = 1;
        H[nh++] = best;
    }
    for (int k = 0; k < nh; k++) mark[H[k]] = 0;
    return nh;
}

static int dict_gate(const Block *B, int *cnt, int *mark) {
    if (B->n <= 12 || B->ncon < 2) return 0;
    double tot = 0;
    for (int t = 0; t < B->ncon; t++) tot += B->A[t].ef;
    if (tot / B->ncon <= 8) return 0;                   /* already sparse: nothing to gain */
    int step = B->ncon / DICT_SAMPLE > 1 ? B->ncon / DICT_SAMPLE : 1, tried = 0, ok = 0;
    int H[DICT_HMAX];
    for (int t = 0; t < B->ncon && tried < DICT_SAMPLE; t += step) {
        tried++;
        int nh = cover(&B->A[t], H, cnt, mark);
        if (nh >= 1 && B->A[t].nnz >= 3 * nh) ok++;     /* hub rows must carry vector content */
    }
    return ok >= 0.75 * tried;
}

static int in_hubs(const HubRow *R, int p) {
    for (int k = 0; k < R->nh; k++) if (R->hub[k] == p) return 1;
    return 0;
}

/* r_a = alpha * r_b on all positions outside both hub sets? (rows sorted by index) */
static int compat(const HubRow *A, const HubRow *Bn, const int *idx, const double *val, double *alpha) {
    int i = A->start, ie = A->start + A->len, j = Bn->start, je = Bn->start + Bn->len;
    double best = 0, al = 0;
    int common = 0;
    while (i < ie || j < je) {                      /* pass 1: scale from the largest common entry */
        int pi = i < ie ? idx[i] : 0x7fffffff, pj = j < je ? idx[j] : 0x7fffffff;
        int p = pi < pj ? pi : pj;
        double xa = (pi == p) ? val[i] : 0.0, xb = (pj == p) ? val[j] : 0.0;
        if (pi == p) i++;
        if (pj == p) j++;
        if (in_hubs(A, p) || in_hubs(Bn, p)) continue;
        if (xa != 0 && xb != 0) {
            common++;
            if (fabs(xa) > best) { best = fabs(xa); al = xa / xb; }
        }
    }
    if (common < 3 || al == 0) return 0;
    i = A->start; j = Bn->start;
    while (i < ie || j < je) {                      /* pass 2: every position must agree */
        int pi = i < ie ? idx[i] : 0x7fffffff, pj = j < je ? idx[j] : 0x7fffffff;
        int p = pi < pj ? pi : pj;
        double xa = (pi == p) ? val[i] : 0.0, xb = (pj == p) ? val[j] : 0.0;
        if (pi == p) i++;
        if (pj == p) j++;
        if (in_hubs(A, p) || in_hubs(Bn, p)) continue;
        if (fabs(xa - al * xb) > 1e-12 * (fabs(xa) + fabs(al * xb))) return 0;
    }
    *alpha = al;
    return 1;
}

static uint64_t mix(uint64_t h, uint64_t x) {
    h ^= x + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
    return h * 0xff51afd7ed558ccdULL;
}

/* hash of DICT_KEY masked entries starting at `from` (+1 forward, -1 backward) */
static int row_key(const HubRow *R, const int *idx, const double *val, int forward, uint64_t *out) {
    int got = 0;
    double ref = 0;
    uint64_t h = forward ? 1469598103934665603ULL : 1099511628211ULL;
    for (int k = 0; k < R->len && got < DICT_KEY; k++) {
        int q = forward ? R->start + k : R->start + R->len - 1 - k;
        if (in_hubs(R, idx[q]) || val[q] == 0) continue;
        if (got == 0) ref = val[q];
        double x = val[q] / ref;
        int64_t qx = (int64_t)llround(x * 1e10);
        h = mix(h, (uint64_t)idx[q]);
        h = mix(h, (uint64_t)qx);
        got++;
    }
    *out = h;
    return got == DICT_KEY;
}

static int cmp_key(const void *a, const void *b) {
    const KeyRow *x = a, *y = b;
    return (x->key > y->key) - (x->key < y->key);
}
static int cmp_trip(const void *a, const void *b) {
    const Trip *x = a, *y = b;
    if (x->c != y->c) return x->c - y->c;
    return x->r - y->r;
}

static int ufind(int *par, int x) {
    while (par[x] != x) { par[x] = par[par[x]]; x = par[x]; }
    return x;
}

/* sort + merge an entry list (upper, r <= c); drops |v| <= tol */
static size_t merge_trips(Trip *T, size_t nt, double tol) {
    for (size_t k = 0; k < nt; k++) if (T[k].r > T[k].c) { int s = T[k].r; T[k].r = T[k].c; T[k].c = s; }
    qsort(T, nt, sizeof(Trip), cmp_trip);
    size_t w = 0;
    for (size_t k = 0; k < nt; k++) {
        if (w > 0 && T[w-1].r == T[k].r && T[w-1].c == T[k].c) T[w-1].v += T[k].v;
        else T[w++] = T[k];
    }
    size_t z = 0;
    for (size_t k = 0; k < w; k++) if (fabs(T[k].v) > tol) T[z++] = T[k];
    return z;
}

void dict_free(Block *B) {
    free(B->dunit); B->dunit = NULL;
    free(B->dV); B->dV = NULL;
    if (B->vb) { block_free_contents(B->vb); free(B->vb); B->vb = NULL; }
    B->dict = 0; B->dnu = B->dnv = 0; B->dstruct = 0;
}

/* Build the dictionary representation. Returns the estimated per-iteration cost of
 * the route (cost-model units), or 1e300 if the block does not qualify. */
double dict_build(Block *B, const Params *par) {
    const int n = B->n, nc = B->ncon;
    int *cnt = xc(n, sizeof(int)), *mark = xc(n, sizeof(int));
    if (par->dict == 0 || (par->dict < 0 && !dict_gate(B, cnt, mark))) { free(cnt); free(mark); return 1e300; }

    /* ---- pass 1: covers and hub rows (entries assigned to exactly one hub) */
    size_t cap = 0;
    for (int t = 0; t < nc; t++) cap += B->A[t].nnz;
    HubRow *R = xm(sizeof(HubRow) * ((size_t)nc * DICT_HMAX + 1));
    int *ridx = xm(sizeof(int) * (cap + 1));
    double *rval = xm(sizeof(double) * (cap + 1));
    char *plain = xc(nc, 1);
    int *first_row = xm(sizeof(int) * (nc + 1));
    int nr = 0;
    size_t pos = 0;
    int H[DICT_HMAX];
    for (int t = 0; t < nc; t++) {
        const SpSym *S = &B->A[t];
        first_row[t] = nr;
        int nh = cover(S, H, cnt, mark);
        if (nh <= 0) { plain[t] = 1; continue; }
        for (int k = 0; k < nh; k++) mark[H[k]] = k + 1;
        for (int k = 0; k < nh; k++) {                           /* count entries per hub */
            R[nr + k].t = t; R[nr + k].h = H[k]; R[nr + k].len = 0; R[nr + k].nh = nh;
            R[nr + k].cl = -1; R[nr + k].beta = 0;
            memcpy(R[nr + k].hub, H, sizeof(int) * nh);
        }
        for (int q = 0; q < S->nnz; q++) {
            int i = S->row[q], j = S->col[q];
            int k = (mark[i] && (!mark[j] || mark[i] <= mark[j])) ? mark[i] - 1 : mark[j] - 1;
            R[nr + k].len++;
        }
        for (int k = 0; k < nh; k++) { R[nr + k].start = (int)pos; pos += R[nr + k].len; R[nr + k].len = 0; }
        for (int q = 0; q < S->nnz; q++) {
            int i = S->row[q], j = S->col[q];
            int k, other;
            if (mark[i] && (!mark[j] || mark[i] <= mark[j])) { k = mark[i] - 1; other = j; }
            else { k = mark[j] - 1; other = i; }
            HubRow *hr = &R[nr + k];
            int at = hr->start + hr->len++;
            ridx[at] = other;
            rval[at] = (i == j) ? 0.5 * S->val[q] : S->val[q];
        }
        for (int k = 0; k < nh; k++) {                           /* sort each row by index */
            HubRow *hr = &R[nr + k];
            for (int a = hr->start + 1; a < hr->start + hr->len; a++) {
                int xi = ridx[a]; double xv = rval[a]; int b = a - 1;
                while (b >= hr->start && ridx[b] > xi) { ridx[b + 1] = ridx[b]; rval[b + 1] = rval[b]; b--; }
                ridx[b + 1] = xi; rval[b + 1] = xv;
            }
            mark[H[k]] = 0;
        }
        nr += nh;
    }
    first_row[nc] = nr;
    free(cnt);

    /* ---- pass 2: hashed clustering of hub rows with exact checks */
    KeyRow *K = xm(sizeof(KeyRow) * (2 * (size_t)nr + 1));
    int nk = 0;
    for (int k = 0; k < nr; k++) {
        uint64_t h1, h2;
        if (row_key(&R[k], ridx, rval, 1, &h1)) { K[nk].key = h1; K[nk].row = k; nk++; }
        if (row_key(&R[k], ridx, rval, 0, &h2)) { K[nk].key = h2 ^ 0x5bd1e995ULL; K[nk].row = k; nk++; }
    }
    qsort(K, nk, sizeof(KeyRow), cmp_key);
    int *par_ = xm(sizeof(int) * (nr + 1));
    char *cand = xc(nr + 1, 1);
    for (int k = 0; k < nr; k++) par_[k] = k;
    for (int a = 0; a < nk; a++) cand[K[a].row] = 1;
    double al;
    for (int a = 0; a < nk; ) {
        int b = a;
        while (b < nk && K[b].key == K[a].key) b++;
        for (int c = a + 1; c < b; c++) {
            int x = ufind(par_, K[a].row), y = ufind(par_, K[c].row);
            if (x != y && compat(&R[K[a].row], &R[K[c].row], ridx, rval, &al)) par_[y] = x;
        }
        a = b;
    }
    free(K);

    /* ---- pass 3: shared vectors (consensus over members), scale of each member */
    int *clid = xm(sizeof(int) * (nr + 1));
    for (int k = 0; k < nr; k++) clid[k] = -1;
    int nv = 0;
    for (int k = 0; k < nr; k++) if (cand[k] && ufind(par_, k) == k) clid[k] = nv++;
    if (nv > 4 * n) nv = -1;                                     /* no compression: give up */
    double *V = NULL;
    if (nv > 0) {
        V = xm(sizeof(double) * (size_t)n * nv);
        for (size_t z = 0; z < (size_t)n * nv; z++) V[z] = NAN;
        for (int k = 0; k < nr; k++) {                           /* roots first */
            if (!cand[k] || ufind(par_, k) != k) continue;
            double *v = V + (size_t)clid[k] * n;
            for (int q = R[k].start; q < R[k].start + R[k].len; q++)
                if (!in_hubs(&R[k], ridx[q])) v[ridx[q]] = rval[q];
            R[k].cl = clid[k]; R[k].beta = 1.0;
        }
        for (int k = 0; k < nr; k++) {                           /* members */
            if (!cand[k]) continue;
            int root = ufind(par_, k);
            if (root == k) continue;
            if (!compat(&R[root], &R[k], ridx, rval, &al)) continue;   /* r_root = al r_k */
            double *v = V + (size_t)clid[root] * n;
            for (int q = R[k].start; q < R[k].start + R[k].len; q++)
                if (!in_hubs(&R[k], ridx[q]) && isnan(v[ridx[q]])) v[ridx[q]] = al * rval[q];
            R[k].cl = clid[root]; R[k].beta = 1.0 / al;
        }
        for (size_t z = 0; z < (size_t)n * nv; z++) if (isnan(V[z])) V[z] = 0.0;
    }
    free(par_); free(cand); free(clid);
    if (nv <= 0) {
        free(R); free(ridx); free(rval); free(plain); free(first_row); free(mark); free(V);
        return 1e300;
    }

    /* ---- pass 4: S_t in code space (unit code = row index, vector code = n + c),
     *              verification against A_t, fallback to unit form               */
    SpSym *SS = xc(nc, sizeof(SpSym));
    Trip *E = NULL, *Rc = NULL;
    size_t ecap = 0, rcap = 0;
    double *dense_v = NULL;
    int nstruct = 0;
    for (int t = 0; t < nc; t++) {
        const SpSym *A = &B->A[t];
        size_t ne = 0, nrc = 0;
        int ok = !plain[t];
        double amax = 0;
        for (int q = 0; q < A->nnz; q++) amax = fmax(amax, fabs(A->val[q]));
        int uses_vec = 0;
        if (ok) {
            size_t need = 0, needr = 0;
            for (int k = first_row[t]; k < first_row[t + 1]; k++) {
                need += (size_t)R[k].len + DICT_HMAX + 1;
                needr += (R[k].cl >= 0 ? (size_t)n : (size_t)R[k].len) + DICT_HMAX + 1;
            }
            if (need > ecap) { ecap = 2 * need; E = realloc(E, sizeof(Trip) * ecap); }
            if (needr > rcap) { rcap = 2 * needr; Rc = realloc(Rc, sizeof(Trip) * rcap); }
            for (int k = first_row[t]; k < first_row[t + 1]; k++) {
                const HubRow *hr = &R[k];
                int h = hr->h;
                if (hr->cl >= 0) {
                    uses_vec = 1;
                    const double *v = V + (size_t)hr->cl * n;
                    E[ne++] = (Trip){ h, n + hr->cl, hr->beta };
                    for (int z = 0; z < n; z++)                      /* reconstruction of beta(e_h v' + v e_h') */
                        if (v[z] != 0) Rc[nrc++] = (Trip){ h, z, hr->beta * v[z] * (z == h ? 2.0 : 1.0) };
                    for (int a = 0; a < hr->nh; a++) {               /* residual on hub positions */
                        int p = hr->hub[a];
                        double rp = 0;
                        for (int q = hr->start; q < hr->start + hr->len; q++) if (ridx[q] == p) { rp = rval[q]; break; }
                        double d = rp - hr->beta * v[p];
                        if (fabs(d) > 1e-15 * (fabs(rp) + fabs(hr->beta * v[p]))) {
                            double w = d * (p == h ? 2.0 : 1.0);
                            E[ne++] = (Trip){ h, p, w };
                            Rc[nrc++] = (Trip){ h, p, w };
                        }
                    }
                } else {
                    for (int q = hr->start; q < hr->start + hr->len; q++) {
                        double w = rval[q] * (ridx[q] == h ? 2.0 : 1.0);
                        E[ne++] = (Trip){ h, ridx[q], w };
                        Rc[nrc++] = (Trip){ h, ridx[q], w };
                    }
                }
            }
            /* verify: merged reconstruction == A_t (upper entries, same order) */
            nrc = merge_trips(Rc, nrc, 1e-12 * amax);
            if (nrc != (size_t)A->nnz) ok = 0;
            else {
                size_t na = (size_t)A->nnz;
                Trip *At = xm(sizeof(Trip) * (na + 1));
                for (size_t q = 0; q < na; q++) At[q] = (Trip){ A->row[q], A->col[q], A->val[q] };
                qsort(At, na, sizeof(Trip), cmp_trip);
                for (size_t q = 0; q < na && ok; q++)
                    if (At[q].r != Rc[q].r || At[q].c != Rc[q].c ||
                        fabs(At[q].v - Rc[q].v) > 1e-11 * amax) ok = 0;
                free(At);
            }
        }
        if (!ok || !uses_vec) {                                   /* exact unit form */
            if ((size_t)A->nnz + 1 > ecap) { ecap = 2 * (size_t)A->nnz + 2; E = realloc(E, sizeof(Trip) * ecap); }
            ne = 0;
            for (int q = 0; q < A->nnz; q++) E[ne++] = (Trip){ A->row[q], A->col[q], A->val[q] };
        } else nstruct++;
        ne = merge_trips(E, ne, 0.0);
        SpSym *S = &SS[t];
        S->nnz = (int)ne;
        S->row = xm(sizeof(int) * (ne + 1));
        S->col = xm(sizeof(int) * (ne + 1));
        S->val = xm(sizeof(double) * (ne + 1));
        for (size_t q = 0; q < ne; q++) { S->row[q] = E[q].r; S->col[q] = E[q].c; S->val[q] = E[q].v; }
    }
    free(E); free(Rc); free(dense_v);
    free(R); free(ridx); free(rval); free(plain); free(first_row);

    /* ---- pass 5: compact the codes into q columns, build the virtual block */
    int *ucol = mark;                                        /* reuse: unit code -> column (0 = unused) */
    int nu = 0;
    for (int t = 0; t < nc; t++)
        for (int q = 0; q < SS[t].nnz; q++) {
            int a = SS[t].row[q], b = SS[t].col[q];
            if (a < n && !ucol[a]) ucol[a] = ++nu;
            if (b < n && !ucol[b]) ucol[b] = ++nu;
        }
    const int qd = nu + nv;
    B->dunit = xm(sizeof(int) * (nu + 1));
    for (int r = 0; r < n; r++) if (ucol[r]) B->dunit[ucol[r] - 1] = r;
    for (int t = 0; t < nc; t++) {
        SpSym *S = &SS[t];
        Trip *T = xm(sizeof(Trip) * (S->nnz + 1));
        for (int q = 0; q < S->nnz; q++) {
            int a = S->row[q], b = S->col[q];
            int ca = a < n ? ucol[a] - 1 : nu + (a - n);
            int cb = b < n ? ucol[b] - 1 : nu + (b - n);
            T[q] = (Trip){ ca, cb, S->val[q] };
        }
        size_t ne = merge_trips(T, S->nnz, 0.0);
        S->nnz = (int)ne;
        for (size_t q = 0; q < ne; q++) { S->row[q] = T[q].r; S->col[q] = T[q].c; S->val[q] = T[q].v; }
        free(T);
        spsym_finish(S, qd, 0);
    }
    free(mark);
    Block *vb = xc(1, sizeof(Block));
    vb->type = BLK_SDP;
    vb->n = qd;
    vb->ncon = nc;
    vb->con = xm(sizeof(int) * (nc + 1));
    memcpy(vb->con, B->con, sizeof(int) * nc);
    vb->A = SS;
    vb->C.nnz = 0;
    vb->C.row = xm(sizeof(int)); vb->C.col = xm(sizeof(int)); vb->C.val = xm(sizeof(double));
    spsym_finish(&vb->C, qd, 0);
    Params pv = *par;
    pv.lowrank = 0;                                          /* P is singular: no factor-based route */
    pv.dict = 0;
    double cost_vb = analyze_sdp_block(vb, &pv);
    B->vb = vb;
    B->dV = V;
    B->dnu = nu;
    B->dnv = nv;
    B->dstruct = nstruct;
    double nd = n;
    double form = par->c_blas * 2.0 * (nd * nd * nv + nd * (double)nv * nv) + par->c_sparse * (double)qd * qd;
    return cost_vb + form;
}
