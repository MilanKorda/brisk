/*
 * Elimination of free variables before the solve.
 *
 * Sum-of-squares models as TSSOS builds them have free scalar variables: one multiplier
 * per equality constraint of the polynomial problem and the bound variable. In an SDPA
 * file they arrive as split pairs x = x+ - x- of LP variables (columns a, -a, costs
 * c, -c). An interior-point method does not like them: the pair has no interior on the
 * dual side, both halves grow, and the Schur complement of the problem without them is
 * singular (the constraints they hold are not held by anything else).
 *
 * Here each free variable is pivoted out on one of its rows, as in sparse Gaussian
 * elimination: with pivot a_rf,
 *     x_f = (b_r - sum_{j != f} a_rj x_j) / a_rf,
 *     row_e <- row_e - (a_ef / a_rf) row_r  for every other row e containing f,
 *     c <- c - (c_f / a_rf) row_r,
 * and row r and the pair are removed. The problem left is purely conic (the moment rows
 * of the eliminated equalities become dependent moments: in moment terms the equality
 * constraints are used to express some moments through others). Pivots are chosen by
 * the Markowitz count (fill) among entries within a threshold of the largest one in the
 * column; free variables whose elimination would fill too much stay as split pairs.
 *
 * Postsolve, in reverse order of elimination:
 *     x_f = (b_r - sum a_rj x_j) / a_rf         (the row as it was when eliminated),
 *     y_r = (c_f - sum_{e != r} a_ef y_e) / a_rf  (the column as it was),
 * which makes the reduced cost of the free column zero and restores the dual of every
 * earlier stage (the transformed reduced costs equal the original ones).
 */
#include "brisk.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>

typedef struct { uint64_t k; double v; } RE;
typedef struct { RE *e; int n, cap; } Row;

#define KEY(b, i, j) (((uint64_t)(b) << 42) | ((uint64_t)(i) << 21) | (uint64_t)(j))
#define KB(k) ((int)((k) >> 42))
#define KI(k) ((int)(((k) >> 21) & 0x1FFFFF))
#define KJ(k) ((int)((k) & 0x1FFFFF))

static void *fx(size_t n) { void *p = calloc(n ? n : 1, 1); if (!p) { fprintf(stderr, "brisk: out of memory\n"); exit(1); } return p; }
static int cmp_re(const void *a, const void *b) {
    uint64_t x = ((const RE *)a)->k, y = ((const RE *)b)->k;
    return x < y ? -1 : x > y;
}
static void row_push(Row *R, uint64_t k, double v) {
    if (R->n == R->cap) { R->cap = R->cap ? 2 * R->cap : 8; R->e = realloc(R->e, sizeof(RE) * R->cap); }
    R->e[R->n].k = k; R->e[R->n].v = v; R->n++;
}
static void row_sort(Row *R) {                  /* sort and merge duplicates */
    if (R->n < 2) return;
    qsort(R->e, R->n, sizeof(RE), cmp_re);
    int w = 0;
    for (int q = 0; q < R->n; q++) {
        if (w > 0 && R->e[w - 1].k == R->e[q].k) R->e[w - 1].v += R->e[q].v;
        else R->e[w++] = R->e[q];
    }
    R->n = w;
}
static double row_get(const Row *R, uint64_t k) {
    int lo = 0, hi = R->n - 1;
    while (lo <= hi) { int mid = (lo + hi) / 2; if (R->e[mid].k == k) return R->e[mid].v; if (R->e[mid].k < k) lo = mid + 1; else hi = mid - 1; }
    return 0.0;
}
/* R <- R - a S (both sorted); exact cancellation of key kz enforced; tiny results dropped */
static void row_axpy(Row *R, double a, const Row *S, uint64_t kz, RE **buf, int *bcap) {
    int need = R->n + S->n;
    if (need > *bcap) { *bcap = 2 * need; *buf = realloc(*buf, sizeof(RE) * *bcap); }
    RE *o = *buf;
    int p = 0, q = 0, w = 0;
    while (p < R->n || q < S->n) {
        uint64_t kr = p < R->n ? R->e[p].k : UINT64_MAX, ks = q < S->n ? S->e[q].k : UINT64_MAX;
        double v; uint64_t k;
        if (kr == ks) { k = kr; v = R->e[p].v - a * S->e[q].v; double sc = fabs(R->e[p].v) + fabs(a * S->e[q].v); p++; q++;
                        if (fabs(v) <= 1e-14 * sc) v = 0; }
        else if (kr < ks) { k = kr; v = R->e[p++].v; }
        else { k = ks; v = -a * S->e[q++].v; }
        if (k == kz) v = 0;
        if (v != 0) { o[w].k = k; o[w].v = v; w++; }
    }
    if (w > R->cap) { R->cap = w; R->e = realloc(R->e, sizeof(RE) * R->cap); }
    memcpy(R->e, o, sizeof(RE) * w);
    R->n = w;
}

typedef struct { uint64_t h; int p; } HP;
static int cmp_hp(const void *x, const void *y) { uint64_t a = ((const HP *)x)->h, b = ((const HP *)y)->h; return a < b ? -1 : a > b; }
static int cmpu(const void *x, const void *y) { uint64_t a = *(const uint64_t *)x, b = *(const uint64_t *)y; return a < b ? -1 : a > b; }
typedef struct { uint64_t k; int f; } KF;
typedef struct { uint64_t k; size_t q; } KQ;     /* (entry key, triplet index), for the dependent-row test */
static int cmp_kq(const void *x, const void *y) { uint64_t a = ((const KQ *)x)->k, b = ((const KQ *)y)->k; return a < b ? -1 : a > b; }
static int cmp_kf(const void *x, const void *y) { uint64_t a = ((const KF *)x)->k, b = ((const KF *)y)->k; return a < b ? -1 : a > b; }
static int kf_find(const KF *kf, int n, uint64_t key) {
    int lo = 0, hi = n - 1;
    while (lo <= hi) { int mid = (lo + hi) / 2; if (kf[mid].k == key) return kf[mid].f; if (kf[mid].k < key) lo = mid + 1; else hi = mid - 1; }
    return -1;
}

/* elimination record (see PSFreeElim in brisk.h) */
void free_elim_free(PSFreeElim *F) {
    if (!F) return;
    for (int s = 0; s < F->ns; s++) { free(F->st[s].rk); free(F->st[s].rv); free(F->st[s].ce); free(F->st[s].cv); }
    free(F->st); free(F->row2file); free(F->pb); free(F->pp); free(F->pm); free(F->lpoff); free(F->lpmap);
    free(F);
}

/* 4.30: split pairs x = x+ - x- of LP variables: equal-and-opposite columns and costs.
 * Returns the count; *out (malloc) holds (block, + slot, - slot) triples. */
int free_pairs_detect(const Problem *P, FreePair **out) {
    const int nb = P->nblk;
    FreePair *pr = NULL; int npr = 0, cpr = 0;
    for (int k = 0; k < nb; k++) {
        const Block *B = &P->blk[k];
        if (B->type != BLK_LP || B->n < 2) continue;
        const int n = B->n;
        Row *col = fx(sizeof(Row) * n);
        double *c = fx(sizeof(double) * n);
        for (int t = 0; t < B->ncon; t++)
            for (int q = 0; q < B->A[t].nnz; q++) row_push(&col[B->A[t].row[q]], (uint64_t)B->con[t], B->A[t].val[q]);
        for (int q = 0; q < B->C.nnz; q++) c[B->C.row[q]] += B->C.val[q];
        uint64_t *h = fx(sizeof(uint64_t) * n);
        int *ord = fx(sizeof(int) * n);
        for (int p = 0; p < n; p++) {
            row_sort(&col[p]);
            /* hash of the column up to sign: sign fixed by its first entry */
            double s = col[p].n ? (col[p].e[0].v > 0 ? 1.0 : -1.0) : (c[p] >= 0 ? 1.0 : -1.0);
            uint64_t hh = 1469598103934665603ULL;
            for (int q = 0; q < col[p].n; q++) {
                double v = s * col[p].e[q].v + 0.0; uint64_t u; memcpy(&u, &v, 8);
                hh = (hh ^ col[p].e[q].k) * 1099511628211ULL; hh = (hh ^ u) * 1099511628211ULL;
            }
            double v = s * c[p] + 0.0; uint64_t u; memcpy(&u, &v, 8);
            h[p] = (hh ^ u) * 1099511628211ULL;
            ord[p] = p;
        }
        {
            HP *hp = fx(sizeof(HP) * n);
            for (int p = 0; p < n; p++) { hp[p].h = h[p]; hp[p].p = p; }
            qsort(hp, n, sizeof(HP), cmp_hp);
            char *used = fx(n);
            for (int a = 0; a < n; ) {
                int bq = a + 1;
                while (bq < n && hp[bq].h == hp[a].h) bq++;
                for (int x = a; x < bq; x++) {
                    int p = hp[x].p;
                    if (used[p] || col[p].n == 0) continue;
                    for (int y = x + 1; y < bq; y++) {
                        int q = hp[y].p;
                        if (used[q] || col[q].n != col[p].n || c[q] != -c[p]) continue;
                        int ok = 1;
                        for (int z = 0; z < col[p].n && ok; z++)
                            if (col[q].e[z].k != col[p].e[z].k || col[q].e[z].v != -col[p].e[z].v) ok = 0;
                        if (!ok) continue;
                        used[p] = used[q] = 1;
                        if (npr == cpr) { cpr = cpr ? 2 * cpr : 64; pr = realloc(pr, sizeof(FreePair) * cpr); }
                        pr[npr].blk = k; pr[npr].ip = p; pr[npr].im = q; npr++;
                        break;
                    }
                }
                a = bq;
            }
            free(hp); free(used);
        }
        for (int p = 0; p < n; p++) free(col[p].e);
        free(col); free(c); free(h); free(ord);
    }
    *out = pr;
    return npr;
}

PSFreeElim *free_eliminate(Problem *P, int verbose, double maxfill, int automatic) {
    const int m = P->m, nb = P->nblk;
    typedef FreePair Pair;
    Pair *pr = NULL;
    int cheap = 0;
    const double tfe0 = wtime(); const int fedbg = getenv("BRISK_FEDBG") != NULL;
    int npr = free_pairs_detect(P, &pr);
    if (fedbg) printf("   [free-elim: %d pairs detected %.2fs]\n", npr, wtime() - tfe0);
    if (npr == 0) { free(pr); return NULL; }
    if (automatic) {
        /* 4.30 auto rule: the kernel-form SOS class (few, moderate SDP blocks; the free
         * variables are a sizeable fraction of the rows: 14-66% on the kernel forms, 96% on
         * the dual forms). TSSOS-type problems (hundreds of small blocks) are 3-8x slower
         * eliminated; a single pair (neu1) or 3.5% of the rows (NH2: 138 -> 283 s) is not
         * worth it. */
        int nsdp = 0;
        for (int k = 0; k < nb; k++) if (P->blk[k].type == BLK_SDP) nsdp++;
        if (npr < 2 || npr < 0.1 * m || npr > m) {
            /* 4.30: outside the class, only the eliminations that cost nothing per iteration
             * are made (a Markowitz product of at most 4: a free variable in a single row, or
             * a row of at most a few entries), without the trace bound and its fallbacks. A
             * lone objective variable (NCTSSOS: the bound of a state polynomial) otherwise
             * sends an 11 346-row dense problem through the saddle factorization, 2.5x the
             * Cholesky per iteration. */
            cheap = 1; maxfill = 4.0;
        }
    }
    /* ---- rows (and the objective as row m) over entry keys; pairs by their '+' key */
    Row *R = fx(sizeof(Row) * (m + 1));
    double *b = fx(sizeof(double) * (m + 1));
    memcpy(b, P->b, sizeof(double) * m);
    /* '-' slots are dropped from the rows (they mirror the '+' slots) */
    size_t nkeys_minus = npr;
    uint64_t *mk = fx(sizeof(uint64_t) * nkeys_minus);
    for (int f = 0; f < npr; f++) mk[f] = KEY(pr[f].blk, pr[f].im, pr[f].im);
    qsort(mk, nkeys_minus, sizeof(uint64_t), cmpu);
    for (int k = 0; k < nb; k++) {
        const Block *B = &P->blk[k];
        const int lp = B->type == BLK_LP;
        for (int t = -1; t < B->ncon; t++) {
            const SpSym *S = t < 0 ? &B->C : &B->A[t];
            Row *RR = t < 0 ? &R[m] : &R[B->con[t]];
            for (int q = 0; q < S->nnz; q++) {
                int i = S->row[q], j = lp ? i : S->col[q];
                if (i > j) { int tt = i; i = j; j = tt; }
                uint64_t key = KEY(k, i, j);
                if (lp && bsearch(&key, mk, nkeys_minus, sizeof(uint64_t), cmpu)) continue;
                row_push(RR, key, S->val[q]);
            }
        }
    }
    for (int i = 0; i <= m; i++) row_sort(&R[i]);
    /* ---- columns of the free variables (row lists; stale entries checked lazily) */
    uint64_t *fk = fx(sizeof(uint64_t) * npr);
    for (int f = 0; f < npr; f++) fk[f] = KEY(pr[f].blk, pr[f].ip, pr[f].ip);
    /* key -> free index: sorted copy */
    KF *kf = fx(sizeof(KF) * npr);
    for (int f = 0; f < npr; f++) { kf[f].k = fk[f]; kf[f].f = f; }
    qsort(kf, npr, sizeof(KF), cmp_kf);
#define freeof(key) kf_find(kf, npr, (key))
    int **cl = fx(sizeof(int *) * npr), *cn = fx(sizeof(int) * npr), *cc = fx(sizeof(int) * npr);
#define col_add(f_, r_) do { int f2_ = (f_); if (cn[f2_] == cc[f2_]) { cc[f2_] = cc[f2_] ? 2 * cc[f2_] : 4; cl[f2_] = realloc(cl[f2_], sizeof(int) * cc[f2_]); } cl[f2_][cn[f2_]++] = (r_); } while (0)
    for (int r = 0; r < m; r++)
        for (int q = 0; q < R[r].n; q++) { int f = freeof(R[r].e[q].k); if (f >= 0) col_add(f, r); }
    char *dead = fx(m + 1), *elim = fx(npr), *skip = fx(npr);
    PSFreeElim *F = fx(sizeof(PSFreeElim));
    F->mfile = m;
    F->st = fx(sizeof(PSFreeStep) * npr);
    F->np = npr;
    F->pb = fx(sizeof(int) * npr); F->pp = fx(sizeof(int) * npr); F->pm = fx(sizeof(int) * npr);
    for (int f = 0; f < npr; f++) { F->pb[f] = pr[f].blk; F->pp[f] = pr[f].ip; F->pm[f] = pr[f].im; }
    double offset = 0;
    RE *buf = NULL; int bcap = 0;
    int *tmp = fx(sizeof(int) * ((m > npr ? m : npr) + 1));   /* 5.8 (a user's fix): a pivot row can hold up to npr free variables, more than m after a symmetry reduction (POEMA tensor_mult_33: heap overflow) */
    char *seen = fx(m + 1);
    long fill0 = 0, fill1 = 0;
    for (int r = 0; r < m; r++) fill0 += R[r].n;
    /* 4.30: live column counts kept incrementally (a heap of (count, f) with lazy
     * invalidation; only the free variables of a pivot row change) - the full rescan per
     * step was quadratic (9 s for the 3485 pairs of the dual form of nn_n10_m50_K) */
    /* keys (count, f): the smallest count first, ties by the pair's index (the order of the
     * full rescan, so the pivots are the same as before) */
    const int g_ferev = getenv("BRISK_FEREV") ? atoi(getenv("BRISK_FEREV")) : 0;
    const double g_fepiv = getenv("BRISK_FEPIV") ? atof(getenv("BRISK_FEPIV")) : 0.1;   /* pivot threshold (relative to the column's largest entry) */   /* debugging: 1 reversed tie-break, 2 none */
    int64_t *hk = fx(sizeof(int64_t) * (npr + 1)); int *hf = fx(sizeof(int) * (npr + 1)); int nh = 0, hc = npr + 1;
#define HPUSH(k_, f_) do { if (nh == hc) { hc *= 2; hk = realloc(hk, sizeof(int64_t) * hc); hf = realloc(hf, sizeof(int) * hc); } \
        int i_ = nh++; hk[i_] = (int64_t)(k_) * npr + (g_ferev == 2 ? 0 : g_ferev ? npr - 1 - (f_) : (f_)); hf[i_] = (f_); \
        while (i_ > 0) { int p_ = (i_ - 1) / 2; if (hk[p_] <= hk[i_]) break; int64_t tk = hk[p_]; int tf = hf[p_]; hk[p_] = hk[i_]; hf[p_] = hf[i_]; hk[i_] = tk; hf[i_] = tf; i_ = p_; } } while (0)
#define HPOP() do { nh--; hk[0] = hk[nh]; hf[0] = hf[nh]; int i_ = 0; \
        for (;;) { int l_ = 2 * i_ + 1, r_ = l_ + 1, s_ = i_; if (l_ < nh && hk[l_] < hk[s_]) s_ = l_; if (r_ < nh && hk[r_] < hk[s_]) s_ = r_; if (s_ == i_) break; \
            int64_t tk = hk[s_]; int tf = hf[s_]; hk[s_] = hk[i_]; hf[s_] = hf[i_]; hk[i_] = tk; hf[i_] = tf; i_ = s_; } } while (0)
    /* live rows of f: not dead, f still present (the pivot row's free variables gain rows and
     * may lose them by cancellation) */
#define RECOUNT(f_) do { int f3_ = (f_), w_ = 0; \
        for (int q_ = 0; q_ < cn[f3_]; q_++) { int r_ = cl[f3_][q_]; if (dead[r_] || seen[r_]) continue; if (row_get(&R[r_], fk[f3_]) == 0) continue; seen[r_] = 1; cl[f3_][w_++] = r_; } \
        for (int q_ = 0; q_ < w_; q_++) { seen[cl[f3_][q_]] = 0; } \
        cn[f3_] = w_; HPUSH(w_, f3_); } while (0)
    for (int f = 0; f < npr; f++) RECOUNT(f);
    if (fedbg) printf("   [free-elim: rows and columns built %.2fs]\n", wtime() - tfe0);
    long naxpy = 0, laxpy = 0;
    for (;;) {
        int best = -1;
        while (nh > 0) {
            const int f0 = hf[0]; const int64_t k0 = hk[0];
            HPOP();
            if (elim[f0] || skip[f0] || k0 != (int64_t)cn[f0] * npr + (g_ferev == 2 ? 0 : g_ferev ? npr - 1 - f0 : f0)) continue;
            best = f0; break;
        }
        if (best < 0) break;
        const int f = best;
        if (cn[f] == 0) { skip[f] = 1; continue; }           /* no rows: left alone (zero column) */
        double amax = 0;
        for (int q = 0; q < cn[f]; q++) amax = fmax(amax, fabs(row_get(&R[cl[f][q]], fk[f])));
        int piv = -1; double pmk = 1e300, pa = 0;
        for (int q = 0; q < cn[f]; q++) {
            int r = cl[f][q];
            double a = row_get(&R[r], fk[f]);
            if (fabs(a) < g_fepiv * amax) continue;
            double mkz = (double)(R[r].n - 1) * (cn[f] - 1);
            if (mkz < pmk || (mkz == pmk && fabs(a) > fabs(pa))) { pmk = mkz; piv = r; pa = a; }
        }
        if (piv < 0 || pmk > maxfill) { skip[f] = 1; continue; }
        /* record the step (row and column as they are now) */
        PSFreeStep *S = &F->st[F->ns++];
        S->blk = pr[f].blk; S->ip = pr[f].ip; S->im = pr[f].im;
        S->row = piv; S->a = pa; S->b = b[piv]; S->c = row_get(&R[m], fk[f]);
        S->nr = R[piv].n - 1;
        S->rk = fx(sizeof(uint64_t) * (S->nr + 1)); S->rv = fx(sizeof(double) * (S->nr + 1));
        for (int q = 0, w = 0; q < R[piv].n; q++) if (R[piv].e[q].k != fk[f]) { S->rk[w] = R[piv].e[q].k; S->rv[w] = R[piv].e[q].v; w++; }
        S->nc = cn[f] - 1;
        S->ce = fx(sizeof(int) * (S->nc + 1)); S->cv = fx(sizeof(double) * (S->nc + 1));
        for (int q = 0, w = 0; q < cn[f]; q++) if (cl[f][q] != piv) { S->ce[w] = cl[f][q]; S->cv[w] = row_get(&R[cl[f][q]], fk[f]); w++; }
        /* other free variables of the pivot row gain the rows of f */
        int nfr = 0;
        for (int q = 0; q < R[piv].n; q++) { int g = freeof(R[piv].e[q].k); if (g >= 0 && g != f) tmp[nfr++] = g; }
        for (int q = 0; q < S->nc; q++) {
            const int e = S->ce[q];
            naxpy++; laxpy += R[e].n + R[piv].n;
            row_axpy(&R[e], S->cv[q] / pa, &R[piv], fk[f], &buf, &bcap);
            b[e] -= S->cv[q] / pa * b[piv];
            for (int z = 0; z < nfr; z++) col_add(tmp[z], e);
        }
        if (S->c != 0) {
            offset += S->c * b[piv] / pa;
            row_axpy(&R[m], S->c / pa, &R[piv], fk[f], &buf, &bcap);
        }
        dead[piv] = 1;
        elim[f] = 1;
        for (int z = 0; z < nfr; z++) RECOUNT(tmp[z]);
    }
    free(hk); free(hf);
    if (fedbg) printf("   [free-elim: elimination loop done %.2fs; %ld row updates of total length %ld; objective row %d entries]\n", wtime() - tfe0, naxpy, laxpy, R[m].n);
    int nelim = F->ns, nskip = 0;
    for (int f = 0; f < npr; f++) if (!elim[f]) nskip++;
    /* rows that became empty: dependent (dropped, dual 0) when their rhs vanished */
    int nempty = 0;
    for (int r = 0; r < m; r++) if (!dead[r] && R[r].n == 0) {
        if (fabs(b[r]) <= 1e-12 * (1.0 + fabs(P->b[r]))) { dead[r] = 2; nempty++; }
    }
    for (int r = 0; r < m; r++) if (!dead[r]) fill1 += R[r].n;
    /* 4.30 auto rule, second part: the elimination is abandoned when it fills too much or
     * leaves most of the pairs behind (butcher: 3197 of 11256 eliminated, nonzeros x7,
     * and the embedding then misjudged the problem as infeasible; the kernel forms fill
     * x2-6 and eliminate nearly all pairs) */
    /* 4.31 (ISSUES item 2, replacing the 32-SDP-block cap): the elimination pays through the
     * rows it removes from a dense Schur complement (m^3), and costs through the nonzeros
     * it adds (assembly, and the fill of a sparse Schur factor). Measured on the TSSOS
     * models (hundreds of small blocks, sparse Schur complement): nonzeros x2.7-3.1, Schur
     * factor flops x2.3-5.4, assembly x3.5-4, iterations +10-20%: case118 0.22 -> 0.82 s,
     * case1354 5.8 -> 18.3 s. On the kernel-form SOS problems (dense Schur complement) the
     * nonzeros grow x2-6 while m falls by 14-66%. Estimate: dense Schur complement (clique
     * density of the block pattern above 0.3): (m1/m0)^3 (nnz1/nnz0); sparse: (m1/m0)(nnz1/nnz0). */
    double cost_ratio = 0;
    if (automatic && !cheap) {
        const long mn_est = m - nelim - nempty;
        double cl2 = 0;
        for (int k = 0; k < nb; k++) { const double c = P->blk[k].ncon; if (P->blk[k].type == BLK_SDP) cl2 += c * c; }
        const double dens = cl2 / ((double)m * m);
        const double rm = (double)mn_est / m, rn = (double)fill1 / fmax(1.0, (double)fill0);
        cost_ratio = dens >= 0.3 ? rm * rm * rm * rn : rm * rn;
        if (fedbg) printf("   [free-elim: cost ratio %.3f (rows %d -> %ld, nonzeros %ld -> %ld, clique density %.3f)]\n", cost_ratio, m, mn_est, fill0, fill1, dens);
    }
    if (automatic && !cheap && (fill1 > 8 * fill0 || 4 * nelim < 3 * npr || cost_ratio > 1.0)) {
        if (verbose)
            printf("presolve: free elimination not used (%d of %d pairs eliminated, nonzeros %ld -> %ld, estimated cost ratio %.2f)\n", nelim, npr, fill0, fill1, cost_ratio);
        for (int r = 0; r <= m; r++) free(R[r].e);
        free(R); free(b); free(mk); free(fk); free(kf);
        for (int f = 0; f < npr; f++) free(cl[f]);
        free(cl); free(cn); free(cc); free(dead); free(elim); free(skip); free(tmp); free(seen); free(buf); free(pr);
        free_elim_free(F);
        return NULL;
    }
    /* ---- rebuild the problem: surviving rows, the skipped pairs restored as pairs */
    int mn = 0;
    int *newof = fx(sizeof(int) * (m + 1));
    for (int r = 0; r < m; r++) newof[r] = dead[r] ? -1 : mn++;
    F->row2file = fx(sizeof(int) * (mn + 1));
    for (int r = 0; r < m; r++) if (newof[r] >= 0) F->row2file[newof[r]] = P->orig ? P->orig[r] : r;
    F->mnew = mn;
    for (int s = 0; s < F->ns; s++) {            /* row ids of the record in file numbering */
        PSFreeStep *S = &F->st[s];
        S->row = P->orig ? P->orig[S->row] : S->row;
        for (int q = 0; q < S->nc; q++) S->ce[q] = P->orig ? P->orig[S->ce[q]] : S->ce[q];
    }
    size_t nt = 0, ct = 1024;
    int *tc = malloc(sizeof(int) * ct), *tb = malloc(sizeof(int) * ct), *ti = malloc(sizeof(int) * ct), *tj = malloc(sizeof(int) * ct);
    double *tv = malloc(sizeof(double) * ct);
#define tput(con_, key_, v_, neg_) do { \
        if (nt == ct) { ct *= 2; tc = realloc(tc, sizeof(int) * ct); tb = realloc(tb, sizeof(int) * ct); ti = realloc(ti, sizeof(int) * ct); tj = realloc(tj, sizeof(int) * ct); tv = realloc(tv, sizeof(double) * ct); } \
        uint64_t k_ = (key_); int bk_ = KB(k_), i_ = KI(k_), j_ = KJ(k_); double v2_ = (v_); \
        if (neg_) { int g_ = freeof(k_); i_ = j_ = pr[g_].im; v2_ = -v2_; } \
        tc[nt] = (con_); tb[nt] = bk_; ti[nt] = i_; tj[nt] = j_; tv[nt] = v2_; nt++; } while (0)
    for (int r = 0; r <= m; r++) {
        if (r < m && newof[r] < 0) continue;
        const int con = r < m ? newof[r] : -1;
        for (int q = 0; q < R[r].n; q++) {
            const uint64_t key = R[r].e[q].k;
            const int g = freeof(key);
            if (g >= 0 && elim[g]) continue;            /* cannot happen: eliminated columns are gone */
            tput(con, key, R[r].e[q].v, 0);
            if (g >= 0) tput(con, key, R[r].e[q].v, 1);   /* the skipped pair's '-' slot */
        }
    }
    int *bsz = fx(sizeof(int) * (nb + 1));
    for (int k = 0; k < nb; k++) bsz[k] = P->blk[k].type == BLK_LP ? -P->blk[k].n : P->blk[k].n;
    double *bn = fx(sizeof(double) * (mn + 1));
    for (int r = 0; r < m; r++) if (newof[r] >= 0) bn[newof[r]] = b[r];
    /* 4.30: rows left numerically dependent by the elimination (roa_acrobot_d6_K: 3 of 3112,
     * against 150 exactly empty ones) make the Schur complement singular from the first
     * iteration and put a floor of 1e-6 on the residual. They are found from the null
     * vectors of the row-normalized Gram matrix (dense eigenvalues, so m <= 4000 only) and
     * dropped one per null vector (the row with the largest weight), when their rhs is
     * consistent (v'b = 0). */
    int ndep2 = 0;
    int *back0 = fx(sizeof(int) * (mn + 1));
    for (int r = 0, w = 0; r < m; r++) if (newof[r] >= 0) back0[w++] = r;
#define bn_of(r_) (b[back0[(r_)]])
    if (mn >= 2 && mn <= 4000 && getenv("BRISK_DEPROWS")) {     /* opt-in: the eigenvalues cost O(m^3) and it did not lift roa_acrobot_d6_K's floor */
        double *G = fx(sizeof(double) * (size_t)mn * mn), *nrm = fx(sizeof(double) * (mn + 1));
        KQ *kq = fx(sizeof(KQ) * (nt + 1));
        for (size_t q = 0; q < nt; q++) { kq[q].k = KEY(tb[q], ti[q] < tj[q] ? ti[q] : tj[q], ti[q] < tj[q] ? tj[q] : ti[q]); kq[q].q = q; }
        qsort(kq, nt, sizeof(KQ), cmp_kq);
        for (size_t q = 0; q < nt; q++) if (tc[q] >= 0) { double w = (ti[q] != tj[q] && bsz[tb[q]] > 0) ? 2.0 : 1.0; nrm[tc[q]] += w * tv[q] * tv[q]; }
        for (int r = 0; r < mn; r++) nrm[r] = sqrt(nrm[r]) + 1e-300;
        for (size_t a0 = 0; a0 < nt; ) {
            size_t b0 = a0 + 1;
            while (b0 < nt && kq[b0].k == kq[a0].k) b0++;
            for (size_t x = a0; x < b0; x++) for (size_t y = a0; y < b0; y++) {
                const size_t qx = kq[x].q, qy = kq[y].q;
                if (tc[qx] < 0 || tc[qy] < 0) continue;
                const double w = (ti[qx] != tj[qx] && bsz[tb[qx]] > 0) ? 2.0 : 1.0;
                G[tc[qx] + (size_t)tc[qy] * mn] += w * tv[qx] * tv[qy] / (nrm[tc[qx]] * nrm[tc[qy]]);
            }
            a0 = b0;
        }
        double *ev = fx(sizeof(double) * (mn + 1)); int lwork = 3 * mn + 64, info = 0; double *work = fx(sizeof(double) * lwork);
        BL(dsyev_)("V", "L", &mn, G, &mn, ev, work, &lwork, &info);
        if (info == 0) {
            /* reduced rows -> file rows, for the drop */
            int *back = fx(sizeof(int) * (mn + 1));
            for (int r = 0, w = 0; r < m; r++) if (newof[r] >= 0) back[w++] = r;
            char *gone = fx(mn + 1);
            for (int e = 0; e < mn && ev[e] < 1e-11 * ev[mn - 1]; e++) {
                const double *v = G + (size_t)e * mn;
                double vb = 0, vn = 0; int best = -1; double bw = 0;
                for (int r = 0; r < mn; r++) {
                    vb += v[r] * bn_of(r); vn += fabs(v[r]);
                    if (!gone[r] && fabs(v[r]) > bw) { bw = fabs(v[r]); best = r; }
                }
                if (best < 0 || fabs(vb) > 1e-9 * (1.0 + vn)) continue;
                gone[best] = 1; dead[back[best]] = 2; ndep2++;
            }
            if (ndep2) {
                /* renumber */
                mn = 0;
                for (int r = 0; r < m; r++) newof[r] = dead[r] ? -1 : mn++;
                free(F->row2file); F->row2file = fx(sizeof(int) * (mn + 1));
                for (int r = 0; r < m; r++) if (newof[r] >= 0) F->row2file[newof[r]] = P->orig ? P->orig[r] : r;
                F->mnew = mn;
                /* rebuild the triplets with the new numbering */
                size_t w = 0;
                for (size_t q = 0; q < nt; q++) {
                    if (tc[q] >= 0) { const int fr = back[tc[q]]; if (dead[fr]) continue; tc[w] = newof[fr]; }
                    else tc[w] = -1;
                    tb[w] = tb[q]; ti[w] = ti[q]; tj[w] = tj[q]; tv[w] = tv[q]; w++;
                }
                nt = w;
            }
            free(back); free(gone);
        }
        if (verbose && (ndep2 || getenv("BRISK_FEDBG")))
            printf("presolve: %d row(s) numerically dependent after the elimination dropped (Gram eigenvalues %.1e .. %.1e)\n", ndep2, ev[0], ev[mn - 1 < 0 ? 0 : (mn - 1)]);
        free(G); free(nrm); free(kq); free(ev); free(work);
    }
    free(back0);
    /* the rhs of the surviving rows (after the dependent-row drop, maybe) */
    for (int r = 0; r < m; r++) if (newof[r] >= 0) bn[newof[r]] = b[r];
    Problem Q;
    /* 4.30: LP slots that no surviving row or the objective touches are dropped (the dual
     * form leaves 2m empty y slots behind: 30654 dummy variables on roa_vdp_inner_d12_I,
     * each costing time and, worse, entering the average complementarity) */
    if (!getenv("BRISK_NOCOMPACT")) {
        F->nblk = nb;
        F->lpoff = fx(sizeof(int) * (nb + 2));
        for (int k = 0; k < nb; k++) F->lpoff[k + 1] = F->lpoff[k] + (bsz[k] < 0 ? -bsz[k] : 0);
        F->lpmap = fx(sizeof(int) * (F->lpoff[nb] + 1));
        char *used = fx(F->lpoff[nb] + 1);
        for (size_t q = 0; q < nt; q++) if (bsz[tb[q]] < 0) used[F->lpoff[tb[q]] + ti[q]] = 1;
        int shrunk = 0;
        for (int k = 0; k < nb; k++) {
            if (bsz[k] >= 0) continue;
            const int n = -bsz[k];
            int w = 0;
            for (int l = 0; l < n; l++) F->lpmap[F->lpoff[k] + l] = used[F->lpoff[k] + l] ? w++ : -1;
            if (w == 0) { F->lpmap[F->lpoff[k]] = 0; w = 1; }
            if (w < n) { shrunk = 1; bsz[k] = -w; }
        }
        if (shrunk) {
            for (size_t q = 0; q < nt; q++) if (bsz[tb[q]] < 0) { ti[q] = tj[q] = F->lpmap[F->lpoff[tb[q]] + ti[q]]; }
        } else { free(F->lpmap); F->lpmap = NULL; }
        free(used);
    }
    problem_from_trips(&Q, mn, nb, bsz, bn, nt, tc, tb, ti, tj, tv);
    for (int k = 0; k < nb; k++) block_free_contents(&P->blk[k]);
    free(P->blk); free(P->b); free(P->b0); free(P->orig);
    P->blk = Q.blk; P->b = Q.b; P->b0 = Q.b0; P->m = mn;
    P->orig = fx(sizeof(int) * (mn + 1));
    for (int i = 0; i < mn; i++) P->orig[i] = i;
    P->morig = mn;
    F->offset = offset;
    F->cheap = cheap;
    P->obj_off += offset;
    if (fedbg) printf("   [free-elim: rebuilt %.2fs]\n", wtime() - tfe0);
    if (verbose)
        printf("presolve: %d free variable(s) (split pairs): %d eliminated%s, %d kept; %d rows removed (%d dependent); nonzeros %ld -> %ld\n",
               npr, nelim, cheap ? " (fill-free only)" : "", nskip, m - mn, nempty, fill0, fill1);
    free(Q.orig);
    free(tc); free(tb); free(ti); free(tj); free(tv); free(bsz); free(bn); free(newof);
    for (int r = 0; r <= m; r++) free(R[r].e);
    free(R); free(b); free(mk); free(fk); free(kf);
    for (int f = 0; f < npr; f++) free(cl[f]);
    free(cl); free(cn); free(cc); free(dead); free(elim); free(skip); free(tmp); free(seen); free(buf); free(pr);
    /* 4.37: nothing eliminated, but the problem may still have changed: empty rows dropped
     * or LP slots compacted. Then the record must be kept for the mapping back (y by
     * row2file, X by lpmap); until 4.36 it was freed, which shifted y by the dropped rows
     * (an empty equality next to a kept pair: err4 0.17 on the file, found through the
     * Julia interface). With nelim = 0 this is the fill-free mode (F->cheap), so no trace
     * bound follows. */
    if (nelim == 0 && mn == m && !F->lpmap) { free_elim_free(F); return NULL; }
    if (nelim == 0) F->cheap = 1;
    return F;
}

/* map a solution of the reduced problem back: y (reduced rows -> file rows, pivot rows
 * recovered) and the eliminated pairs' slots of X (Xo in the file's block layout) */
void free_elim_post(const PSFreeElim *F, const PSOrig *Ofile, double **Xo, const double *yred, double *yfile) {
    if (Xo && F->lpmap) {
        /* the compacted LP blocks back to the file layout */
        for (int k = 0; k < F->nblk; k++) {
            const int n = F->lpoff[k + 1] - F->lpoff[k];
            if (n == 0) continue;
            double *x = fx(sizeof(double) * (n + 1));
            for (int l = 0; l < n; l++) { const int c = F->lpmap[F->lpoff[k] + l]; if (c >= 0) x[l] = Xo[k][c]; }
            free(Xo[k]); Xo[k] = x;
        }
    }
    for (int i = 0; i < F->mfile; i++) yfile[i] = 0;
    for (int i = 0; i < F->mnew; i++) yfile[F->row2file[i]] = yred[i];
    for (int s = F->ns - 1; s >= 0; s--) {
        const PSFreeStep *S = &F->st[s];
        /* dual of the pivot row */
        double t = S->c;
        for (int q = 0; q < S->nc; q++) t -= S->cv[q] * yfile[S->ce[q]];
        yfile[S->row] = t / S->a;
        if (!Xo) continue;
        /* the free value from its pivot row */
        double v = S->b;
        for (int q = 0; q < S->nr; q++) {
            const uint64_t k = S->rk[q];
            const int bk = KB(k), i = KI(k), j = KJ(k);
            double x;
            if (Ofile->bs[bk] < 0) {
                x = Xo[bk][i];
                /* a free variable (by its '+' slot): its value is + minus - */
                for (int g = 0; g < F->np; g++)
                    if (F->pb[g] == bk && F->pp[g] == i) { x = Xo[bk][i] - Xo[bk][F->pm[g]]; break; }
            } else {
                const int n = Ofile->bs[bk];
                x = (i == j) ? Xo[bk][i + (size_t)i * n] : 2.0 * Xo[bk][i + (size_t)j * n];
            }
            v -= S->rv[q] * x;
        }
        v /= S->a;
        Xo[S->blk][S->ip] = v > 0 ? v : 0.0;
        Xo[S->blk][S->im] = v < 0 ? -v : 0.0;
    }
}

/* ---- 4.30: trace bound. On kernel-form SOS problems the Gram matrices (the primal X)
 * are unbounded at the optimum (the moment side has no interior), the embedding's tau
 * collapses and the primal residual blows up near the end. One dense row
 *     sum_k tr(X_k) + s = R,  s >= 0 (a new LP block of size 1),
 * with R far above any solution of interest keeps tau ~ 1 without changing the problem
 * unless it binds (then the slack's dual w != 0 and the caller re-solves with a larger R).
 * The row is the last one and the block the last block; tracebound_strip drops them. */
void tracebound_add(Problem *P, double R) {
    const int m = P->m, nb = P->nblk;
    size_t nt = 0, ct = 1024;
    int *tc = malloc(sizeof(int) * ct), *tb = malloc(sizeof(int) * ct), *ti = malloc(sizeof(int) * ct), *tj = malloc(sizeof(int) * ct);
    double *tv = malloc(sizeof(double) * ct);
#define tput2(con_, bk_, i_, j_, v_) do { \
        if (nt == ct) { ct *= 2; tc = realloc(tc, sizeof(int) * ct); tb = realloc(tb, sizeof(int) * ct); ti = realloc(ti, sizeof(int) * ct); tj = realloc(tj, sizeof(int) * ct); tv = realloc(tv, sizeof(double) * ct); } \
        tc[nt] = (con_); tb[nt] = (bk_); ti[nt] = (i_); tj[nt] = (j_); tv[nt] = (v_); nt++; } while (0)
    for (int k = 0; k < nb; k++) {
        const Block *B = &P->blk[k];
        const int lp = B->type == BLK_LP;
        for (int t = -1; t < B->ncon; t++) {
            const SpSym *S = t < 0 ? &B->C : &B->A[t];
            const int con = t < 0 ? -1 : B->con[t];
            for (int q = 0; q < S->nnz; q++) tput2(con, k, S->row[q], lp ? S->row[q] : S->col[q], S->val[q]);
        }
        if (!lp) for (int i = 0; i < B->n; i++) tput2(m, k, i, i, 1.0);
    }
    tput2(m, nb, 0, 0, 1.0);
    int *bsz = malloc(sizeof(int) * (nb + 2));
    for (int k = 0; k < nb; k++) bsz[k] = P->blk[k].type == BLK_LP ? -P->blk[k].n : P->blk[k].n;
    bsz[nb] = -1;
    double *bn = malloc(sizeof(double) * (m + 2));
    memcpy(bn, P->b, sizeof(double) * m);
    bn[m] = R;
    Problem Q;
    problem_from_trips(&Q, m + 1, nb + 1, bsz, bn, nt, tc, tb, ti, tj, tv);
    for (int k = 0; k < nb; k++) block_free_contents(&P->blk[k]);
    free(P->blk); free(P->b); free(P->b0); free(P->orig);
    P->blk = Q.blk; P->b = Q.b; P->b0 = Q.b0; P->m = m + 1; P->nblk = nb + 1;
    P->orig = malloc(sizeof(int) * (m + 2));
    for (int i = 0; i <= m; i++) P->orig[i] = i;
    P->morig = m + 1;
    P->tbR = R;
    free(Q.orig);
    free(tc); free(tb); free(ti); free(tj); free(tv); free(bsz); free(bn);
}

/* 4.30: after the back-substitution, the free values are refined by least squares on the
 * file problem: with the cone part fixed, x_f <- x_f + argmin ||r - A_f dx||, r = b - A(X).
 * The chain of eliminations (roa_acrobot_d6_K: 742 pairs, multipliers up to 10) amplifies
 * the reduced residual (1e-13) to 3.5e-6 on the file rows; one least-squares pass on the
 * nf x nf normal equations (dense Cholesky) removes that. */
void free_elim_refine(const PSFreeElim *F, const PSOrig *O, double **Xo, int verbose) {
    const int m = O->m, nf = F->np;
    if (!Xo || nf <= 0 || nf > 6000) return;
    /* free index of an LP slot (block, index) of a '+' half */
    int *slot = malloc(sizeof(int) * (nf + 1)), *sblk = malloc(sizeof(int) * (nf + 1));
    for (int g = 0; g < nf; g++) { slot[g] = F->pp[g]; sblk[g] = F->pb[g]; }
    double *r = malloc(sizeof(double) * (m + 1));
    for (int i = 0; i < m; i++) r[i] = O->b[i];
    /* A_f as column lists: (row, value) per free variable, from the '+' slots */
    int *cnt = calloc(nf + 1, sizeof(int));
    for (size_t q = 0; q < O->nnz; q++) {
        const int c = O->con[q], bk = O->blk[q], i = O->ii[q], j = O->jj[q];
        if (c < 0) continue;
        double x;
        if (O->bs[bk] < 0) {
            x = Xo[bk][i];
            for (int g = 0; g < nf; g++) if (sblk[g] == bk && slot[g] == i) cnt[g]++;
        } else {
            const int n = O->bs[bk];
            x = (i == j) ? Xo[bk][i + (size_t)i * n] : 2.0 * Xo[bk][i + (size_t)j * n];
        }
        r[c] -= O->v[q] * x;
    }
    double r0 = 0; for (int i = 0; i < m; i++) r0 += r[i] * r[i];
    if (getenv("BRISK_FEDBG")) {
        char *ispiv = calloc(m + 1, 1), *hasf = calloc(m + 1, 1);
        for (int s2 = 0; s2 < F->ns; s2++) { ispiv[F->st[s2].row] = 1; for (int q = 0; q < F->st[s2].nc; q++) hasf[F->st[s2].ce[q]] = 1; }
        double mp = 0, mf = 0, mo = 0; int ip = -1, jf = -1, ko = -1;
        for (int i = 0; i < m; i++) {
            const double a = fabs(r[i]);
            if (ispiv[i]) { if (a > mp) { mp = a; ip = i; } }
            else if (hasf[i]) { if (a > mf) { mf = a; jf = i; } }
            else if (a > mo) { mo = a; ko = i; }
        }
        printf("   [free-elim residuals on the file rows: pivot rows max %.2e (row %d), rows with free vars %.2e (row %d), others %.2e (row %d)]\n", mp, ip, mf, jf, mo, ko);
        free(ispiv); free(hasf);
    }
    int *off = malloc(sizeof(int) * (nf + 2)); off[0] = 0;
    for (int g = 0; g < nf; g++) off[g + 1] = off[g] + cnt[g];
    int *crow = malloc(sizeof(int) * (off[nf] + 1)); double *cval = malloc(sizeof(double) * (off[nf] + 1));
    int *fill = calloc(nf + 1, sizeof(int));
    /* map (blk, slot) -> g by a small hash over LP slots */
    int maxlp = 0; for (int k = 0; k < O->nblk; k++) if (O->bs[k] < 0 && -O->bs[k] > maxlp) maxlp = -O->bs[k];
    const size_t ngof = (size_t)(maxlp + 1) * (size_t)O->nblk;
    int *gof = malloc(sizeof(int) * (ngof + 1));
    for (size_t q = 0; q < ngof; q++) gof[q] = -1;
    for (int g = 0; g < nf; g++) gof[slot[g] + (size_t)(maxlp + 1) * sblk[g]] = g;
    for (size_t q = 0; q < O->nnz; q++) {
        const int c = O->con[q], bk = O->blk[q];
        if (c < 0 || O->bs[bk] >= 0) continue;
        const int g = gof[O->ii[q] + (size_t)(maxlp + 1) * bk];
        if (g < 0) continue;
        crow[off[g] + fill[g]] = c; cval[off[g] + fill[g]] = O->v[q]; fill[g]++;
    }
    double *rhs = calloc(nf + 1, sizeof(double));
    int solved = 0;
    if (nf > 1500) {
        /* 4.30: CGLS on the sparse A_f (the dual form of nn_n10_m50_K has 3485 free variables:
         * the dense normal equations cost 0.5 s, more than the solve) */
        double *rr = malloc(sizeof(double) * (m + 1)), *q = malloc(sizeof(double) * (m + 1));
        double *sv = malloc(sizeof(double) * (nf + 1)), *pv = malloc(sizeof(double) * (nf + 1));
        memcpy(rr, r, sizeof(double) * m);
#define ATR(out_) do { for (int g_ = 0; g_ < nf; g_++) { double t_ = 0; for (int e_ = off[g_]; e_ < off[g_ + 1]; e_++) t_ += cval[e_] * rr[crow[e_]]; (out_)[g_] = t_; } } while (0)
        ATR(sv);
        double gam = 0; for (int g = 0; g < nf; g++) gam += sv[g] * sv[g];
        const double gam0 = gam;
        memcpy(pv, sv, sizeof(double) * nf);
        for (int it = 0; it < 500 && gam > 1e-28 * gam0; it++) {
            memset(q, 0, sizeof(double) * m);
            for (int g = 0; g < nf; g++) for (int e = off[g]; e < off[g + 1]; e++) q[crow[e]] += cval[e] * pv[g];
            double qq = 0; for (int i = 0; i < m; i++) qq += q[i] * q[i];
            if (!(qq > 0)) break;
            const double al = gam / qq;
            for (int g = 0; g < nf; g++) rhs[g] += al * pv[g];
            for (int i = 0; i < m; i++) rr[i] -= al * q[i];
            ATR(sv);
            double g2 = 0; for (int g = 0; g < nf; g++) g2 += sv[g] * sv[g];
            const double be = g2 / gam; gam = g2;
            for (int g = 0; g < nf; g++) pv[g] = sv[g] + be * pv[g];
        }
#undef ATR
        free(rr); free(q); free(sv); free(pv);
        solved = 1;
    }
    int *rowg = malloc(sizeof(int) * (m + 1)); double *rowv = malloc(sizeof(double) * (m + 1));
    for (int i = 0; i <= m; i++) rowg[i] = -1;
    /* rows -> (g, v) lists: build by scanning columns; a row may hold several free vars */
    int *rcnt = calloc(m + 1, sizeof(int));
    for (int g = 0; g < nf; g++) for (int q = off[g]; q < off[g + 1]; q++) rcnt[crow[q]]++;
    int *roff = malloc(sizeof(int) * (m + 2)); roff[0] = 0;
    for (int i = 0; i < m; i++) roff[i + 1] = roff[i] + rcnt[i];
    int *rg = malloc(sizeof(int) * (roff[m] + 1)); double *rv = malloc(sizeof(double) * (roff[m] + 1));
    int *rfill = calloc(m + 1, sizeof(int));
    for (int g = 0; g < nf; g++) for (int q = off[g]; q < off[g + 1]; q++) { const int i = crow[q]; rg[roff[i] + rfill[i]] = g; rv[roff[i] + rfill[i]] = cval[q]; rfill[i]++; }
    double *G = NULL;
    if (!solved) {
    /* normal equations G = A_f' A_f (dense, lower), rhs = A_f' r */
    G = calloc((size_t)nf * nf, sizeof(double));
    for (int i = 0; i < m; i++)
        for (int a = roff[i]; a < roff[i + 1]; a++) {
            rhs[rg[a]] += rv[a] * r[i];
            for (int c2 = roff[i]; c2 < roff[i + 1]; c2++)
                if (rg[c2] <= rg[a]) G[rg[a] + (size_t)rg[c2] * nf] += rv[a] * rv[c2];
        }
    for (int g = 0; g < nf; g++) G[g + (size_t)g * nf] += 1e-14 * (1.0 + G[g + (size_t)g * nf]);
    int info = 0, one = 1;
    BL(dpotrf_)("L", &nf, G, &nf, &info);
    if (info == 0) { BL(dpotrs_)("L", &nf, &one, G, &nf, rhs, &nf, &info); solved = 1; }
    }
    if (solved) {
        for (int g = 0; g < nf; g++) {
            const int bk = sblk[g];
            const double x = Xo[bk][slot[g]] - Xo[bk][F->pm[g]] + rhs[g];
            Xo[bk][slot[g]] = x > 0 ? x : 0.0;
            Xo[bk][F->pm[g]] = x < 0 ? -x : 0.0;
        }
        for (int i = 0; i < m; i++) for (int a = roff[i]; a < roff[i + 1]; a++) r[i] -= rv[a] * rhs[rg[a]];
        double r1 = 0; for (int i = 0; i < m; i++) r1 += r[i] * r[i];
        if (verbose > 0) printf("postsolve: free values refined by least squares: residual %.2e -> %.2e\n", sqrt(r0), sqrt(r1));
    }
    free(slot); free(sblk); free(r); free(cnt); free(off); free(crow); free(cval); free(fill); free(gof); free(G); free(rhs);
    free(rowg); free(rowv); free(rcnt); free(roff); free(rg); free(rv); free(rfill);
}
