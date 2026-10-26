#include "brisk.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <sys/stat.h>
#include <ctype.h>

typedef struct { int con, blk, i, j; double v; } Trip;

static void *xmalloc(size_t n) {
    void *p = malloc(n ? n : 1);
    if (!p) { fprintf(stderr, "brisk: out of memory (%zu bytes)\n", n); exit(1); }
    return p;
}
static void *xcalloc(size_t n, size_t s) {
    void *p = calloc(n ? n : 1, s);
    if (!p) { fprintf(stderr, "brisk: out of memory\n"); exit(1); }
    return p;
}

/* 5.12: decimal numbers whose digits fit 2^53 with a power of ten of at most 22 converted exactly
 * by one multiplication or division (Clinger's fast path: both operands exact, one correctly
 * rounded operation), the rest by strtod; 0 when the token is not of that form. The same doubles
 * as strtod's, several times faster on the large moment relaxations (83 MB: 1.29 s to read). */
static int fast_num(const char *p, char **endp, double *out) {
    static const double p10[23] = { 1e0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9, 1e10, 1e11, 1e12, 1e13, 1e14, 1e15,
                                    1e16, 1e17, 1e18, 1e19, 1e20, 1e21, 1e22 };
    const char *q = p;
    int neg = 0;
    if (*q == '-' || *q == '+') { neg = *q == '-'; q++; }
    unsigned long long mnt = 0; int nd = 0, e10 = 0, any = 0;
    while (*q >= '0' && *q <= '9') { if (nd < 19) { mnt = mnt * 10 + (unsigned)(*q - '0'); if (mnt) nd++; } else e10++; q++; any = 1; }
    if (*q == '.') {
        q++;
        while (*q >= '0' && *q <= '9') { if (nd < 19) { mnt = mnt * 10 + (unsigned)(*q - '0'); if (mnt) nd++; e10--; } q++; any = 1; }
    }
    if (!any) return 0;
    if (*q == 'e' || *q == 'E') {
        const char *r = q + 1; int eneg = 0, ev = 0, dig = 0;
        if (*r == '-' || *r == '+') { eneg = *r == '-'; r++; }
        while (*r >= '0' && *r <= '9') { if (ev < 10000) ev = ev * 10 + (*r - '0'); r++; dig = 1; }
        if (!dig) return 0;
        e10 += eneg ? -ev : ev; q = r;
    }
    if (*q && !isspace((unsigned char)*q)) return 0;
    if (nd >= 19 || mnt > (1ULL << 53) || e10 < -22 || e10 > 22) return 0;
    double v = (double)mnt;
    if (e10 > 0) v *= p10[e10]; else if (e10 < 0) v /= p10[-e10];
    *out = neg ? -v : v; *endp = (char *)q;
    return 1;
}
/* Parse next number; skips non-numeric tokens (counted). Returns 0 at end of buffer. */
static long g_skipped_words = 0;
static int next_num(char **pp, double *out) {
    char *p = *pp;
    for (;;) {
        while (*p && isspace((unsigned char)*p)) p++;
        if (!*p) { *pp = p; return 0; }
        char *e;
        { static int fn = -1; if (fn < 0) fn = !getenv("BRISK_NOFASTNUM");
          if (fn && fast_num(p, &e, out)) { *pp = e; return 1; } }
        double v = strtod(p, &e);
        if (e != p && (*e == '\0' || isspace((unsigned char)*e))) { *out = v; *pp = e; return 1; }
        if (e != p) { *out = v; *pp = e; g_skipped_words++; return 1; }   /* number glued to text */
        while (*p && !isspace((unsigned char)*p)) p++;   /* skip a word */
        g_skipped_words++;
    }
}

/* first number on the next non-comment line */
static int line_num(char **pp, double *out) {
    char *p = *pp;
    while (*p) {
        char *eol = strchr(p, '\n');
        char *next = eol ? eol + 1 : p + strlen(p);
        if (eol) *eol = '\0';
        char *q = p;
        while (*q && isspace((unsigned char)*q)) q++;
        int ok = 0;
        if (*q && *q != '"' && *q != '*') ok = next_num(&q, out);
        if (eol) *eol = ' ';
        p = next;
        if (ok) { *pp = p; return 1; }
    }
    *pp = p;
    return 0;
}

static int cmp_trip(const void *a, const void *b) {
    const Trip *x = a, *y = b;
    if (x->blk != y->blk) return x->blk - y->blk;
    if (x->con != y->con) return x->con - y->con;
    if (x->i != y->i) return x->i - y->i;
    return x->j - y->j;
}

/* 4.40: the largest SDP block the reader accepts: n^2 must fit an int for the dense paths;
 * -lralm (lralm.c) never forms n x n arrays and lifts it */
int g_read_maxn = 46340;

static int cmp_int_rows(const void *x, const void *y) { const int a = *(const int *)x, b = *(const int *)y; return a < b ? -1 : a > b; }
void spsym_finish(SpSym *S, int n, int is_lp) {
    int k, nf = 0;
    S->fr = xmalloc(sizeof(int) * (2 * S->nnz));
    S->fc = xmalloc(sizeof(int) * (2 * S->nnz));
    S->fv = xmalloc(sizeof(double) * (2 * S->nnz));
    for (k = 0; k < S->nnz; k++) {
        S->fr[nf] = S->row[k]; S->fc[nf] = S->col[k]; S->fv[nf] = S->val[k]; nf++;
        if (!is_lp && S->row[k] != S->col[k]) {
            S->fr[nf] = S->col[k]; S->fc[nf] = S->row[k]; S->fv[nf] = S->val[k]; nf++;
        }
    }
    S->ef = nf;
    if ((size_t)nf * 16 < (size_t)n) {
        /* 4.41: few entries in a large block (an LP block of 60 000 with 30 000 rows after
         * the algebra reduction): sort the rows instead of a mark array of n per matrix */
        int *r = xmalloc(sizeof(int) * (nf + 1));
        for (k = 0; k < nf; k++) r[k] = S->fr[k];
        if (nf <= 32) { for (int a = 1; a < nf; a++) { const int v = r[a]; int b = a - 1; while (b >= 0 && r[b] > v) { r[b + 1] = r[b]; b--; } r[b + 1] = v; } }
        else qsort(r, nf, sizeof(int), cmp_int_rows);
        int c = 0;
        for (k = 0; k < nf; k++) if (c == 0 || r[k] != r[c - 1]) r[c++] = r[k];
        S->nr = c;
        S->rows = xmalloc(sizeof(int) * (c + 1));
        memcpy(S->rows, r, sizeof(int) * c);
        free(r);
        return;
    }
    char *mark = xcalloc(n, 1);
    S->nr = 0;
    for (k = 0; k < nf; k++) if (!mark[S->fr[k]]) { mark[S->fr[k]] = 1; S->nr++; }
    S->rows = xmalloc(sizeof(int) * S->nr);
    int c = 0;
    for (k = 0; k < n; k++) if (mark[k]) S->rows[c++] = k;
    free(mark);
}

/* blocks from (sorted or not) triplets; duplicates summed, zeros dropped */
/* 23 October 2026: the triplets in the order (blk, con, i, j) of cmp_trip by a stable LSD radix sort, 11 bits a pass,
 * the passes no value needs skipped; nothing when they are in order already. glibc's qsort is a merge sort (stable) at
 * these sizes, so the order, ties included (duplicates are summed in it), is the same. A D-optimal design with 9.9
 * million entries, rebuilt after the free elimination: 3.8 s in qsort's comparisons, 23 a triplet. BRISK_TRIPQSORT=1:
 * qsort. */
/* the order of a problem written row by row: (constraint, block, i, j), the objective's row (constraint -1) first or,
 * with last = 1, after the constraints */
static int cmp_rows(const Trip *x, const Trip *y, int last) {
    if (x->con != y->con) {
        if (last && (x->con < 0 || y->con < 0)) return x->con < 0 ? 1 : -1;
        return x->con < y->con ? -1 : 1;
    }
    if (x->blk != y->blk) return x->blk < y->blk ? -1 : 1;
    if (x->i != y->i) return x->i < y->i ? -1 : 1;
    return (x->j > y->j) - (x->j < y->j);
}
static void sort_trips(Trip *T, size_t nt) {
    if (nt < 2) return;
    size_t q = 1;
    while (q < nt && cmp_trip(&T[q - 1], &T[q]) <= 0) q++;
    if (q == nt) return;
    Trip *U = (nt >= 4096 && !getenv("BRISK_TRIPQSORT")) ? malloc(sizeof(Trip) * nt) : NULL;
    if (!U) { qsort(T, nt, sizeof(Trip), cmp_trip); return; }
    /* 23 October 2026: a problem written row by row (the free elimination's rebuild: the rows' entries in order, the
     * objective last): one stable counting pass by (block, objective or constraint) gives the order (block, constraint,
     * i, j), ties in their order; the radix sort's seven passes over 9.9 million triplets took 2 s on a D-optimal design */
    for (int last = 0; last <= 1; last++) {
        size_t r = 1;
        while (r < nt && cmp_rows(&T[r - 1], &T[r], last) <= 0) r++;
        if (r < nt) continue;
        int nbk = 0; for (size_t e = 0; e < nt; e++) if (T[e].blk + 1 > nbk) nbk = T[e].blk + 1;
        size_t *c2 = calloc(2 * (size_t)nbk + 1, sizeof(size_t));
        if (!c2) break;
        for (size_t e = 0; e < nt; e++) c2[2 * (size_t)T[e].blk + (T[e].con >= 0)]++;
        size_t o = 0; for (size_t d = 0; d < 2 * (size_t)nbk; d++) { const size_t c = c2[d]; c2[d] = o; o += c; }
        for (size_t e = 0; e < nt; e++) U[c2[2 * (size_t)T[e].blk + (T[e].con >= 0)]++] = T[e];
        memcpy(T, U, sizeof(Trip) * nt);
        free(c2); free(U);
        return;
    }
    unsigned mx[4] = { 0, 0, 0, 0 };          /* the fields from the least significant: j, i, con + 1, blk */
    for (size_t r = 0; r < nt; r++) {
        if ((unsigned)T[r].j > mx[0]) mx[0] = (unsigned)T[r].j;
        if ((unsigned)T[r].i > mx[1]) mx[1] = (unsigned)T[r].i;
        if ((unsigned)(T[r].con + 1) > mx[2]) mx[2] = (unsigned)(T[r].con + 1);
        if ((unsigned)T[r].blk > mx[3]) mx[3] = (unsigned)T[r].blk;
    }
    enum { RB = 11, RN = 1 << RB };
    size_t *cnt = malloc(sizeof(size_t) * RN);
    if (!cnt) { free(U); qsort(T, nt, sizeof(Trip), cmp_trip); return; }
    Trip *src = T, *dst = U;
    for (int f = 0; f < 4; f++)
        for (int sh = 0; sh < 32 && (mx[f] >> sh) != 0; sh += RB) {
            memset(cnt, 0, sizeof(size_t) * RN);
            for (size_t r = 0; r < nt; r++) {
                const unsigned v = f == 0 ? (unsigned)src[r].j : f == 1 ? (unsigned)src[r].i : f == 2 ? (unsigned)(src[r].con + 1) : (unsigned)src[r].blk;
                cnt[(v >> sh) & (RN - 1)]++;
            }
            size_t o = 0; for (int d = 0; d < RN; d++) { const size_t c = cnt[d]; cnt[d] = o; o += c; }
            for (size_t r = 0; r < nt; r++) {
                const unsigned v = f == 0 ? (unsigned)src[r].j : f == 1 ? (unsigned)src[r].i : f == 2 ? (unsigned)(src[r].con + 1) : (unsigned)src[r].blk;
                dst[cnt[(v >> sh) & (RN - 1)]++] = src[r];
            }
            Trip *t = src; src = dst; dst = t;
        }
    if (src != T) memcpy(T, src, sizeof(Trip) * nt);
    free(cnt); free(U);
}

static void build_blocks(Problem *P, Trip *T, size_t nt, long lower_entries) {
    const int nblk = P->nblk;
    static int bbdbg = -1; if (bbdbg < 0) bbdbg = getenv("BRISK_FEDBG") != NULL;
    const double tbb0 = bbdbg ? wtime() : 0;
    sort_trips(T, nt);
    if (bbdbg && nt > 1000000) printf("   [build_blocks: %zu triplets sorted %.2fs]\n", nt, wtime() - tbb0);
    /* merge duplicates (summed, as SDPA does), then drop entries that cancelled */
    size_t w = 0, ndup = 0;
    for (size_t r = 0; r < nt; r++) {
        if (w > 0 && T[w-1].blk == T[r].blk && T[w-1].con == T[r].con &&
            T[w-1].i == T[r].i && T[w-1].j == T[r].j) { T[w-1].v += T[r].v; ndup++; }
        else T[w++] = T[r];
    }
    nt = w;
    w = 0;
    for (size_t r = 0; r < nt; r++) if (T[r].v != 0.0) T[w++] = T[r];
    nt = w;
    if (ndup)
        fprintf(stderr, "brisk: warning: %zu duplicate entr%s summed%s\n", ndup, ndup == 1 ? "y" : "ies",
                lower_entries ? " (the file also has lower-triangle entries: if it lists both triangles, "
                                "off-diagonals are counted twice)" : "");
    if (bbdbg && nt > 1000000) printf("   [build_blocks: duplicates merged %.2fs]\n", wtime() - tbb0);
    size_t r = 0;
    double tfin = 0;
    for (int k = 0; k < nblk; k++) {
        Block *B = &P->blk[k];
        size_t s = r;
        const double tk0 = bbdbg ? wtime() : 0;
        while (r < nt && T[r].blk == k) r++;
        /* count constraints in [s,r) (con = -1 is C) */
        int ncon = 0;
        for (size_t t = s; t < r; t++)
            if (T[t].con >= 0 && (t == s || T[t].con != T[t-1].con)) ncon++;
        B->ncon = ncon;
        B->con = xmalloc(sizeof(int) * ncon);
        B->A = xcalloc(ncon, sizeof(SpSym));
        int c = -1, lastcon = -2;
        size_t t = s;
        while (t < r) {
            size_t u = t;
            while (u < r && T[u].con == T[t].con) u++;
            SpSym *S;
            if (T[t].con < 0) S = &B->C;
            else {
                if (T[t].con != lastcon) { c++; lastcon = T[t].con; }
                B->con[c] = T[t].con;
                S = &B->A[c];
            }
            S->nnz = (int)(u - t);
            S->row = xmalloc(sizeof(int) * S->nnz);
            S->col = xmalloc(sizeof(int) * S->nnz);
            S->val = xmalloc(sizeof(double) * S->nnz);
            for (size_t q = t; q < u; q++) {
                S->row[q-t] = T[q].i; S->col[q-t] = T[q].j; S->val[q-t] = T[q].v;
            }
            t = u;
        }
        if (!B->C.row) {        /* empty C */
            B->C.nnz = 0;
            B->C.row = xmalloc(sizeof(int)); B->C.col = xmalloc(sizeof(int));
            B->C.val = xmalloc(sizeof(double));
        }
        const double tf0 = bbdbg ? wtime() : 0;
        spsym_finish(&B->C, B->n, B->type == BLK_LP);
        for (int q = 0; q < ncon; q++) spsym_finish(&B->A[q], B->n, B->type == BLK_LP);
        if (bbdbg) { tfin += wtime() - tf0; if (r - s > 1000000) printf("   [build_blocks: block %d (%s %d, %d constraints, %zu entries) %.2fs, of which spsym_finish %.2fs]\n", k + 1, B->type == BLK_LP ? "LP" : "SDP", B->n, ncon, r - s, wtime() - tk0, wtime() - tf0); }
    }
    if (bbdbg && nt > 1000000) printf("   [build_blocks: blocks built %.2fs (spsym_finish %.2fs)]\n", wtime() - tbb0, tfin);
}

/* A problem built in memory (used by the moment form of the chordal conversion). T is
 * consumed. Entries (row <= col) as in the file, con = -1 for C (C itself, not -F0). */
void problem_from_trips(Problem *P, int m, int nblk, const int *bsz, const double *b,
                        size_t nt, const int *con, const int *blk, const int *ii, const int *jj, const double *v) {
    memset(P, 0, sizeof(*P));
    P->m = m; P->nblk = nblk;
    P->blk = xcalloc(nblk, sizeof(Block));
    for (int k = 0; k < nblk; k++) { P->blk[k].type = bsz[k] < 0 ? BLK_LP : BLK_SDP; P->blk[k].n = abs(bsz[k]); }
    P->b = xmalloc(sizeof(double) * (m + 1));
    P->b0 = xmalloc(sizeof(double) * (m + 1));
    memcpy(P->b, b, sizeof(double) * m);
    memcpy(P->b0, b, sizeof(double) * m);
    P->morig = m;
    P->orig = xmalloc(sizeof(int) * (m + 1));
    for (int i = 0; i < m; i++) P->orig[i] = -1;
    Trip *T = xmalloc(sizeof(Trip) * (nt + 1));
    for (size_t q = 0; q < nt; q++) {
        int a = ii[q], c2 = jj[q];
        if (a > c2) { int t = a; a = c2; c2 = t; }
        T[q].con = con[q]; T[q].blk = blk[q]; T[q].i = a; T[q].j = c2; T[q].v = v[q];
    }
    build_blocks(P, T, nt, 0);
    free(T);
}

/* 4.37: the problem from the numbers of an SDPA file held in memory (the library interface,
 * brisk_run_data). The same checks and the same result as read_sdpa on the file those
 * numbers would make: C = -F0, entries below the diagonal swapped, zeros skipped. */
int problem_from_sdpa_data(const BriskData *d, Problem *P) {
    memset(P, 0, sizeof(*P));
    const int m = d->m, nblk = d->nblk;
    if (m < 0 || m > 100000000 || nblk < 1 || !d->bs || (m > 0 && !d->c) || (d->nnz > 0 && (!d->mat || !d->blk || !d->i || !d->j || !d->v))) {
        fprintf(stderr, "brisk: invalid problem data (m = %d, %d blocks)\n", m, nblk);
        return -1;
    }
    P->m = m;
    P->nblk = nblk;
    P->blk = xcalloc(nblk, sizeof(Block));
    for (int k = 0; k < nblk; k++) {
        const int s = d->bs[k];
        if (s == 0 || s < -2000000000 || s > 2000000000) {
            fprintf(stderr, "brisk: invalid size %d of block %d\n", s, k + 1);
            goto bad;
        }
        P->blk[k].type = s < 0 ? BLK_LP : BLK_SDP;
        P->blk[k].n = abs(s);
        if (P->blk[k].type == BLK_SDP && P->blk[k].n > g_read_maxn) {
            fprintf(stderr, "brisk: SDP block %d of size %d exceeds the supported maximum (46340; -lralm has no such limit)\n", k + 1, P->blk[k].n);
            goto bad;
        }
    }
    P->b = xmalloc(sizeof(double) * (m ? m : 1));
    P->b0 = xmalloc(sizeof(double) * (m ? m : 1));
    for (int i = 0; i < m; i++) {
        if (!isfinite(d->c[i])) { fprintf(stderr, "brisk: non-finite value c[%d]\n", i + 1); goto bad; }
        P->b[i] = P->b0[i] = d->c[i];
    }
    P->morig = m;
    P->orig = xmalloc(sizeof(int) * (m ? m : 1));
    for (int i = 0; i < m; i++) P->orig[i] = i;
    long lower_entries = 0;
    size_t nt = 0;
    Trip *T = xmalloc(sizeof(Trip) * (d->nnz + 1));
    for (size_t q = 0; q < d->nnz; q++) {
        const int mat = d->mat[q], bk = d->blk[q] - 1;
        int i = d->i[q] - 1, j = d->j[q] - 1;
        const double v = d->v[q];
        if (!isfinite(v)) {
            fprintf(stderr, "brisk: non-finite value in entry %d %d %d %d\n", mat, bk + 1, i + 1, j + 1);
            free(T); goto bad;
        }
        if (v == 0.0) continue;
        if (mat < 0 || mat > m || bk < 0 || bk >= nblk || i < 0 || j < 0 ||
            i >= P->blk[bk].n || j >= P->blk[bk].n) {
            fprintf(stderr, "brisk: bad entry %d %d %d %d\n", mat, bk + 1, i + 1, j + 1);
            free(T); goto bad;
        }
        if (P->blk[bk].type == BLK_LP && i != j) {
            fprintf(stderr, "brisk: off-diagonal entry in LP block\n");
            free(T); goto bad;
        }
        if (i > j) { int t = i; i = j; j = t; lower_entries++; }
        T[nt].con = mat - 1; T[nt].blk = bk; T[nt].i = i; T[nt].j = j;
        T[nt].v = (mat == 0) ? -v : v;      /* C = -F0 */
        nt++;
    }
    build_blocks(P, T, nt, lower_entries);
    free(T);
    return 0;
bad:
    fprintf(stderr, "brisk: invalid problem data\n");
    free(P->blk); free(P->b); free(P->b0); free(P->orig);   /* nothing else is built yet */
    memset(P, 0, sizeof(*P));
    return -1;
}

int read_sdpa(const char *fname, Problem *P) {
    FILE *f = fopen(fname, "rb");
    if (!f) { fprintf(stderr, "brisk: cannot open %s\n", fname); return -1; }
    { struct stat st_;       /* 4.42: a directory opens without error and then "needs" 2^63 bytes */
      if (fstat(fileno(f), &st_) == 0 && S_ISDIR(st_.st_mode)) { fprintf(stderr, "brisk: %s is a directory, not an SDPA file\n", fname); fclose(f); return -1; } }
    fseek(f, 0, SEEK_END);
    long len = ftell(f);
    fseek(f, 0, SEEK_SET);
    char *buf = xmalloc(len + 1);
    if (fread(buf, 1, len, f) != (size_t)len) { fclose(f); free(buf); return -1; }
    buf[len] = '\0';
    fclose(f);
    for (long i = 0; i < len; i++) {
        char ch = buf[i];
        if (ch == ',' || ch == '{' || ch == '}' || ch == '(' || ch == ')' || ch == '=') buf[i] = ' ';
        /* Fortran-style exponents: 1.5D+03 -> 1.5e+03 */
        else if ((ch == 'D' || ch == 'd') && i > 0 && i + 1 < len &&
                 (isdigit((unsigned char)buf[i - 1]) || buf[i - 1] == '.') &&
                 (isdigit((unsigned char)buf[i + 1]) ||
                  ((buf[i + 1] == '+' || buf[i + 1] == '-') && i + 2 < len && isdigit((unsigned char)buf[i + 2]))))
            buf[i] = 'e';
    }
    char *p = buf;
    double v;
    if (!line_num(&p, &v)) goto bad;
    int m = (int)v;
    if (!line_num(&p, &v)) goto bad;
    int nblk = (int)v;
    /* block structure: remaining tokens, comment lines stripped from here on */
    {
        char *q = p;
        while (*q) {
            char *s = q;
            while (*s == ' ' || *s == '\t' || *s == '\r') s++;
            char *eol = strchr(q, '\n');
            if (*s == '"' || *s == '*') {
                char *e = eol ? eol : q + strlen(q);
                for (char *t = q; t < e; t++) *t = ' ';
            }
            if (!eol) break;
            q = eol + 1;
        }
    }
    if (m < 0 || nblk < 1 || v != floor(v) || m > 100000000) goto bad;
    P->m = m;
    P->nblk = nblk;
    P->blk = xcalloc(nblk, sizeof(Block));
    for (int k = 0; k < nblk; k++) {
        if (!next_num(&p, &v)) goto bad;
        if (!(fabs(v) >= 1 && fabs(v) <= 2e9) || v != floor(v)) {
            fprintf(stderr, "brisk: invalid size %g of block %d\n", v, k + 1);
            goto bad;
        }
        int s = (int)v;
        P->blk[k].type = s < 0 ? BLK_LP : BLK_SDP;
        P->blk[k].n = abs(s);
        if (P->blk[k].type == BLK_SDP && P->blk[k].n > g_read_maxn) {
            fprintf(stderr, "brisk: SDP block %d of size %d exceeds the supported maximum (46340; -lralm has no such limit)\n", k + 1, P->blk[k].n);
            goto bad;
        }
    }
    P->b = xmalloc(sizeof(double) * m);
    P->b0 = xmalloc(sizeof(double) * m);
    for (int i = 0; i < m; i++) {
        if (!next_num(&p, &v) || !isfinite(v)) goto bad;
        P->b[i] = v;
        P->b0[i] = v;
    }
    P->morig = m;
    P->orig = xmalloc(sizeof(int) * (m ? m : 1));
    for (int i = 0; i < m; i++) P->orig[i] = i;
    g_skipped_words = 0;
    long lower_entries = 0;
    size_t cap = 1024, nt = 0;
    Trip *T = xmalloc(sizeof(Trip) * cap);
    double e[5];
    for (;;) {
        int got = 0;
        {   /* 5.18: text after the last entry (Mittelmann's neosfbr25 ends with a BEGIN.COMMENT ... END.COMMENT block,
             * which 1.5's strict reading of ISSUES 101 rejected as malformed): at an entry boundary, a token that does
             * not start a number, with no line of numbers after it, ends the data with a note; text inside an entry
             * or before more entries stays an error */
            char *q = p;
            while (*q && isspace((unsigned char)*q)) q++;
            if (*q && !(isdigit((unsigned char)*q) || *q == '-' || *q == '+' || *q == '.')) {
                int more = 0; long nl = 0;
                for (char *r = q; *r && !more; ) {
                    char *eol = strchr(r, '\n');
                    const size_t len = eol ? (size_t)(eol - r) : strlen(r);
                    nl++;
                    char *t = r; int nnum = 0;
                    while (t < r + len) {
                        while (t < r + len && isspace((unsigned char)*t)) t++;
                        if (t >= r + len) break;
                        char *e2; (void)strtod(t, &e2);
                        if (e2 == t) break;
                        nnum++; t = e2;
                    }
                    if (nnum >= 5) more = 1;
                    r = eol ? eol + 1 : r + len;
                }
                if (!more) {
                    fprintf(stderr, "brisk: note: %ld line(s) of text after the last entry of %s ignored\n", nl, fname);
                    break;
                }
            }
        }
        for (got = 0; got < 5; got++) if (!next_num(&p, &e[got])) break;
        if (got < 5) {
            if (got > 0) fprintf(stderr, "brisk: warning: incomplete last entry (%d of 5 numbers) ignored\n", got);
            break;
        }
        for (int q = 0; q < 4; q++)
            if (e[q] != floor(e[q]) || fabs(e[q]) > 2e9) {
                fprintf(stderr, "brisk: non-integer index in entry %g %g %g %g %g\n", e[0], e[1], e[2], e[3], e[4]);
                free(T); goto bad;
            }
        if (!isfinite(e[4])) {
            fprintf(stderr, "brisk: non-finite value in entry %g %g %g %g\n", e[0], e[1], e[2], e[3]);
            free(T); goto bad;
        }
        int mat = (int)e[0], bk = (int)e[1] - 1, i = (int)e[2] - 1, j = (int)e[3] - 1;
        if (e[4] == 0.0) continue;
        if (mat < 0 || mat > m || bk < 0 || bk >= nblk || i < 0 || j < 0 ||
            i >= P->blk[bk].n || j >= P->blk[bk].n) {
            fprintf(stderr, "brisk: bad entry %d %d %d %d\n", mat, bk + 1, i + 1, j + 1);
            free(T); goto bad;
        }
        if (P->blk[bk].type == BLK_LP && i != j) {
            fprintf(stderr, "brisk: off-diagonal entry in LP block\n");
            free(T); goto bad;
        }
        if (i > j) { int t = i; i = j; j = t; lower_entries++; }
        if (nt == cap) { cap *= 2; T = realloc(T, sizeof(Trip) * cap); if (!T) exit(1); }
        T[nt].con = mat - 1; T[nt].blk = bk; T[nt].i = i; T[nt].j = j;
        T[nt].v = (mat == 0) ? -e[4] : e[4];      /* C = -F0 */
        nt++;
    }
    /* 15 October 2026 (the owner's benchmark, ISSUES 101): a non-numeric token in the data section is an error, not a
     * warning (a file whose entries read np.float64(0.75) was solved as a garbled problem with status OPTIMAL) */
    if (g_skipped_words) {
        fprintf(stderr, "brisk: %ld non-numeric token(s) in the data section of %s (every entry is \"matrix block i j value\", numbers only)\n", g_skipped_words, fname);
        g_skipped_words = 0;
        free(T); goto bad;
    }
    free(buf);
    build_blocks(P, T, nt, lower_entries);
    free(T);
    return 0;
bad:
    fprintf(stderr, "brisk: malformed SDPA file %s\n", fname);
    free(buf);
    return -1;
}

static void spsym_scale(SpSym *S, double s) {
    for (int k = 0; k < S->nnz; k++) S->val[k] *= s;
    for (int k = 0; k < S->ef; k++) S->fv[k] *= s;
}


/* ------------------------------------------------------------------------ */
/* Low-rank factorization A = W diag(sig) W' of one constraint block part.
 * Small support: eigendecomposition of the principal submatrix.
 * Otherwise: randomized range finder with k = 8, 16, 32 and a randomized
 * residual check. Returns the rank, or -1 if not (cheaply) low-rank.       */
#define LR_SEED 88172645463325252UL
static unsigned long lr_rng = LR_SEED;
/* 5.15: back to the seed at the start of each solve (brisk_main): the state ran on from solve to
 * solve, so in a library host (Julia, Python, MATLAB) the second solve of the same problem got
 * other random vectors, other low-rank factors and other bits (gpp124-1: 11 of 16 solves in one
 * process differed from the first, once in the status) */
void lowrank_rng_reset(void) { lr_rng = LR_SEED; }
static double lr_rand(void) {
    lr_rng ^= lr_rng << 13; lr_rng ^= lr_rng >> 7; lr_rng ^= lr_rng << 17;
    return (double)(lr_rng >> 11) / 9007199254740992.0 - 0.5;
}

static void sp_mulvec(const SpSym *S, const double *x, double *y, int n) {   /* y = S x */
    memset(y, 0, sizeof(double) * n);
    for (int q = 0; q < S->ef; q++) y[S->fr[q]] += S->fv[q] * x[S->fc[q]];
}

/* (22 October 2026: Wout NULL asks for the rank alone, from the eigenvalues without vectors and with no verification) */
static int lowrank_factor(const SpSym *S, int n, int kmax, double **Wout, double **sigout) {
    const int one = 1, want = Wout != NULL;
    double anrm = 0;
    for (int q = 0; q < S->ef; q++) anrm += S->fv[q] * S->fv[q];
    anrm = sqrt(anrm);
    if (anrm == 0) { if (want) { *Wout = NULL; *sigout = NULL; } return 0; }
    int r = S->nr;
    double *T = NULL, *ev = NULL, *Q = NULL, *W = NULL, *sig = NULL;
    int k = 0, rank = -1;
    if (r <= kmax) {
        /* dense principal submatrix on the support rows */
        k = r;
        T = xcalloc((size_t)k * k, sizeof(double));
        int *pos = xmalloc(sizeof(int) * n);
        for (int i = 0; i < n; i++) pos[i] = -1;
        for (int i = 0; i < r; i++) pos[S->rows[i]] = i;
        for (int q = 0; q < S->ef; q++) T[pos[S->fr[q]] + (size_t)pos[S->fc[q]] * k] = S->fv[q];
        Q = xcalloc((size_t)n * k, sizeof(double));
        for (int i = 0; i < r; i++) Q[S->rows[i] + (size_t)i * n] = 1.0;
        free(pos);
    } else {
        for (k = 8; k <= kmax && k < n; k *= 2) {
            Q = xmalloc(sizeof(double) * (size_t)n * k);
            double *om = xmalloc(sizeof(double) * n);
            for (int j = 0; j < k; j++) {                 /* Y = A * Omega */
                for (int i = 0; i < n; i++) om[i] = lr_rand();
                sp_mulvec(S, om, Q + (size_t)j * n, n);
            }
            free(om);
            /* orthonormalize (modified Gram-Schmidt, twice) and drop null directions */
            int kk = 0;
            for (int j = 0; j < k; j++) {
                double *qj = Q + (size_t)j * n;
                double n0 = BL(dnrm2_)(&n, qj, &one);
                for (int pass = 0; pass < 2; pass++)
                    for (int i = 0; i < kk; i++) {
                        double c = -BL(ddot_)(&n, Q + (size_t)i * n, &one, qj, &one);
                        BL(daxpy_)(&n, &c, Q + (size_t)i * n, &one, qj, &one);
                    }
                double nj = BL(dnrm2_)(&n, qj, &one);
                if (nj <= 1e-10 * (n0 + 1e-300)) continue;
                double inv = 1.0 / nj;
                double *dst = Q + (size_t)kk * n;
                for (int i = 0; i < n; i++) dst[i] = qj[i] * inv;
                kk++;
            }
            if (kk < k) break;          /* range captured (fewer directions than samples) */
            free(Q); Q = NULL;          /* full rank sample: try a bigger k */
            if (k * 2 > kmax) break;
            continue;
        }
        if (!Q) return -1;
        /* count kept columns again (those were compacted to the front) */
        int kk = 0;
        for (int j = 0; j < k; j++) {
            double nj = BL(dnrm2_)(&n, Q + (size_t)j * n, &one);
            if (fabs(nj - 1.0) < 1e-8) kk++; else break;
        }
        k = kk;
        if (k == 0) { free(Q); return -1; }
        /* T = Q' A Q */
        double *AQ = xmalloc(sizeof(double) * (size_t)n * k);
        for (int j = 0; j < k; j++) sp_mulvec(S, Q + (size_t)j * n, AQ + (size_t)j * n, n);
        T = xmalloc(sizeof(double) * (size_t)k * k);
        const double d1 = 1.0, d0 = 0.0;
        BL(dgemm_)("T", "N", &k, &k, &n, &d1, Q, &n, AQ, &n, &d0, T, &k);
        for (int j = 0; j < k; j++)
            for (int i = 0; i < j; i++) { double v = 0.5 * (T[i + (size_t)j * k] + T[j + (size_t)i * k]); T[i + (size_t)j * k] = T[j + (size_t)i * k] = v; }
        free(AQ);
    }
    /* eigendecomposition of the small matrix */
    ev = xmalloc(sizeof(double) * k);
    int lwork = -1, info;
    double wq;
    BL(dsyev_)(want ? "V" : "N", "U", &k, T, &k, ev, &wq, &lwork, &info);
    lwork = (int)wq + 1;
    double *work = xmalloc(sizeof(double) * lwork);
    BL(dsyev_)(want ? "V" : "N", "U", &k, T, &k, ev, work, &lwork, &info);
    free(work);
    if (info != 0) goto fail;
    double emax = 0;
    for (int j = 0; j < k; j++) emax = fmax(emax, fabs(ev[j]));
    rank = 0;
    for (int j = 0; j < k; j++) if (fabs(ev[j]) > 1e-14 * emax) rank++;
    if (!want) {    /* (the verification's 2 n random draws taken all the same: the random stream, hence every later factor, as before) */
        for (int i = 0; i < 2 * n; i++) (void)lr_rand();
        free(T); free(ev); free(Q); return rank;
    }
    W = xmalloc(sizeof(double) * (size_t)n * (rank ? rank : 1));
    sig = xmalloc(sizeof(double) * (rank ? rank : 1));
    {
        int c = 0;
        const double d1 = 1.0, d0 = 0.0;
        for (int j = 0; j < k; j++) {
            if (fabs(ev[j]) <= 1e-14 * emax) continue;
            BL(dgemv_)("N", &n, &k, &d1, Q, &n, T + (size_t)j * k, &one, &d0, W + (size_t)c * n, &one);
            sig[c++] = ev[j];
        }
    }
    /* randomized residual check: || A z - W S W' z || <= 1e-10 ||A|| ||z|| */
    {
        double *z = xmalloc(sizeof(double) * n), *az = xmalloc(sizeof(double) * n);
        double *c = xmalloc(sizeof(double) * (rank ? rank : 1));
        for (int trial = 0; trial < 2 && rank >= 0; trial++) {
            for (int i = 0; i < n; i++) z[i] = lr_rand();
            sp_mulvec(S, z, az, n);
            for (int j = 0; j < rank; j++) c[j] = -sig[j] * BL(ddot_)(&n, W + (size_t)j * n, &one, z, &one);
            for (int j = 0; j < rank; j++) BL(daxpy_)(&n, &c[j], W + (size_t)j * n, &one, az, &one);
            double res = BL(dnrm2_)(&n, az, &one), zn = BL(dnrm2_)(&n, z, &one);
            if (res > 1e-10 * anrm * zn) rank = -1;
        }
        free(z); free(az); free(c);
    }
    if (rank < 0) goto fail;
    free(T); free(ev); free(Q);
    *Wout = W; *sigout = sig;
    return rank;
fail:
    free(T); free(ev); free(Q); free(W); free(sig);
    return -1;
}

/* Decide whether a block should use the low-rank Schur route and build it. */
static void lowrank_block(Block *B, const Params *par, double cost_now) {
    const int n = B->n, nc = B->ncon;
    const double cs = par->c_sparse, cb = par->c_blas;
    double nd = n;
    B->lowrank = 0;
    if (nc == 0 || par->lowrank == 0) return;
    /* optimistic bound: every constraint has rank >= 1 */
    double lb = cb * (nd * nd * nc + nd * (double)nc * nc) + 0.3 * cs * (double)nc * nc;
    if (par->lowrank < 0 && lb >= cost_now) return;
    int kmax = n < 32 ? n : 32;
    double **Ws = xcalloc(nc, sizeof(double *)), **Ss = xcalloc(nc, sizeof(double *));
    int *rk = xmalloc(sizeof(int) * nc);
    long R = 0;
    int ok = 1;
    int small = n < 64 ? n : 64;
    /* 22 October 2026: the decision from the ranks alone (eigenvalues without vectors, several times cheaper), the
     * factors with their vectors and the verification only when the route is taken, the random numbers put back
     * between the two passes so that a taken route gets the same factors as before (delay_indep_n32, a control LMI:
     * 1,056 eigendecompositions with vectors, 0.56 s, for a route that is not taken) */
    const unsigned long rng0 = lr_rng;
    for (int t = 0; t < nc && ok; t++) {
        rk[t] = lowrank_factor(&B->A[t], n, B->A[t].nr <= small ? small : kmax, NULL, NULL);
        if (rk[t] < 0) ok = 0; else R += rk[t];
        if (ok && par->lowrank < 0) {
            double lbt = cb * (nd * nd * (R + (nc - t - 1)) + nd * (double)(R + nc - t - 1) * (R + nc - t - 1));
            if (lbt >= cost_now) ok = 0;
        }
    }
    double cost_lr = cb * (nd * nd * R + nd * (double)R * R) + 0.3 * cs * (double)R * R;
    double mem = 8.0 * (double)R * R;         /* single-buffer mode */
    if (getenv("BRISK_LRDBG")) printf("   [low-rank block: n %d, %d constraints, rank %ld (ok %d): cost %.3g against the routes' %.3g, memory %.3g of %.3g]\n", n, nc, R, ok, cost_lr, cost_now, mem, par->dense_mem);
    if (ok && (par->lowrank > 0 || (cost_lr < cost_now && mem <= par->dense_mem))) {
        lr_rng = rng0;
        long R2 = 0;
        for (int t = 0; t < nc && ok; t++) {
            const int r2 = lowrank_factor(&B->A[t], n, B->A[t].nr <= small ? small : kmax, &Ws[t], &Ss[t]);
            if (r2 < 0) ok = 0; else { rk[t] = r2; R2 += r2; }
        }
        if (ok && R2 != R) {     /* (the vectors' pass found another rank somewhere: decide again on it) */
            R = R2; cost_lr = cb * (nd * nd * R + nd * (double)R * R) + 0.3 * cs * (double)R * R; mem = 8.0 * (double)R * R;
            if (!(par->lowrank > 0 || (cost_lr < cost_now && mem <= par->dense_mem))) ok = 0;
        }
    }
    if (ok && (par->lowrank > 0 || (cost_lr < cost_now && mem <= par->dense_mem))) {
        B->lowrank = 1;
        B->lrR = (int)R;
        B->lr_off = xmalloc(sizeof(int) * (nc + 1));
        B->lr_W = xmalloc(sizeof(double) * (size_t)n * (R ? R : 1));
        B->lr_sig = xmalloc(sizeof(double) * (R ? R : 1));
        long c = 0;
        for (int t = 0; t < nc; t++) {
            B->lr_off[t] = (int)c;
            if (rk[t] > 0) {
                memcpy(B->lr_W + (size_t)c * n, Ws[t], sizeof(double) * (size_t)n * rk[t]);
                memcpy(B->lr_sig + c, Ss[t], sizeof(double) * rk[t]);
            }
            c += rk[t];
        }
        B->lr_off[nc] = (int)c;
    }
    for (int t = 0; t < nc; t++) { free(Ws[t]); free(Ss[t]); }
    free(Ws); free(Ss); free(rk);
}

/* 5.18: a rank-one factor from the data's pivot column, w = A e_p / sqrt|A_pp| with s = sign A_pp (p the largest
 * diagonal entry): w w' s reproduces the column p exactly and the rest to the rounding of the data, where the
 * eigendecomposition of lowrank_factor carries about n u (an occupation-measure dual form ended a decade less
 * accurate through the dictionary built on it). Kept when it reproduces the entries better. */
static void rank1_exact(const SpSym *S, int n, double *W, double *sg) {
    int p = -1; double best = 0;
    for (int q = 0; q < S->ef; q++) if (S->fr[q] == S->fc[q] && fabs(S->fv[q]) > best) { best = fabs(S->fv[q]); p = S->fr[q]; }
    if (p < 0) return;
    double *w = xcalloc(n, sizeof(double));
    double dpp = 0;
    for (int q = 0; q < S->ef; q++) if (S->fr[q] == p && S->fc[q] == p) dpp = S->fv[q];
    const double s = dpp > 0 ? 1.0 : -1.0, rt = sqrt(fabs(dpp));
    for (int q = 0; q < S->ef; q++) if (S->fc[q] == p) w[S->fr[q]] = S->fv[q] / (s * rt);
    double e_new = 0, e_old = 0;
    for (int q = 0; q < S->ef; q++) {
        const int i = S->fr[q], j = S->fc[q];
        e_new = fmax(e_new, fabs(s * w[i] * w[j] - S->fv[q]));
        e_old = fmax(e_old, fabs(sg[0] * W[i] * W[j] - S->fv[q]));
    }
    if (e_new <= e_old) { memcpy(W, w, sizeof(double) * n); sg[0] = s; }
    free(w);
}
double analyze_sdp_block(Block *B, const Params *par);
int brisk_block_as_cone(const Params *par, const Block *B);
typedef struct { int r, c; double v; } LDTrip;
static int cmp_ldtrip(const void *a, const void *b) {
    const LDTrip *x = a, *y = b;
    if (x->r != y->r) return x->r < y->r ? -1 : 1;
    return (x->c > y->c) - (x->c < y->c);
}
/* 5.18 (the owner's benchmark, ISSUES 97: A-optimal design, one LMI of order 180 with 2,000 dense rank-one
 * constraints v v' and 4,095 sparse ones): the dictionary route with the factors of the constraints of low rank as
 * its shared columns. A constraint whose factor is smaller than its entries (A_t = sum_k s_k w_k w_k', rank r <= 8,
 * 4 n r <= its entries) contributes the columns w_k with S_t = diag(s_k) on them; every other constraint keeps its
 * entries on unit columns (exact). Then tr(A_t X A_u Zi) = tr(S_t P S_u Q) with P = D'XD, Q = D'Zi D, the virtual
 * block holds 1 to 2 entries a constraint, and the pairs of two rank-one constraints are P_tu Q_tu. The dense route
 * took 2 n^3 per such constraint and m_d^2 n^2 for their pairs (2.7 s an assembly); the block-level low-rank route
 * counts every sparse constraint's factor too (R = 10,100 against 2,000). BRISK_LRDICT=0 off. */
static double lrdict_build(Block *B, const Params *par) {
    static int on = -1; if (on < 0) { const char *e = getenv("BRISK_LRDICT"); on = e ? atoi(e) : 1; }
    const int n = B->n, nc = B->ncon;
    const double n2 = (double)n * n;
    if (!on || par->dict == 0 || par->lowrank == 0 || nc < 2 || n < 16) return 1e300;
    int *rk = xcalloc(nc, sizeof(int)); double **Ws = xcalloc(nc, sizeof(double *)), **Ss = xcalloc(nc, sizeof(double *));
    int nl = 0; long nv = 0;
    for (int t = 0; t < nc; t++) {
        const SpSym *S = &B->A[t];
        if (S->ef < 0.25 * n2) continue;
        double *W = NULL, *sg = NULL;
        int r = lowrank_factor(S, n, 8, &W, &sg);
        if (r == 1) rank1_exact(S, n, W, sg);
        if (r > 0 && r <= 8 && 4.0 * n * r <= S->ef) { rk[t] = r; Ws[t] = W; Ss[t] = sg; nl++; nv += r; }
        else { free(W); free(sg); }
    }
    if (nl < 2) { for (int t = 0; t < nc; t++) { free(Ws[t]); free(Ss[t]); } free(rk); free(Ws); free(Ss); return 1e300; }
    double *V = xmalloc(sizeof(double) * (size_t)n * nv);
    SpSym *SS = xcalloc(nc, sizeof(SpSym));
    int *ucol = xcalloc(n + 1, sizeof(int)), nu = 0;
    for (int t = 0; t < nc; t++) if (!rk[t]) {
        const SpSym *A = &B->A[t];
        for (int q = 0; q < A->nnz; q++) {
            if (!ucol[A->row[q]]) ucol[A->row[q]] = ++nu;
            if (!ucol[A->col[q]]) ucol[A->col[q]] = ++nu;
        }
    }
    long c = 0;
    for (int t = 0; t < nc; t++) {
        SpSym *S = &SS[t];
        const SpSym *A = &B->A[t];
        const int ne = rk[t] ? rk[t] : A->nnz;
        LDTrip *T = xmalloc(sizeof(LDTrip) * (ne + 1));
        if (rk[t]) {
            for (int k = 0; k < rk[t]; k++) {
                memcpy(V + (size_t)(c + k) * n, Ws[t] + (size_t)k * n, sizeof(double) * n);
                T[k] = (LDTrip){ (int)(nu + c + k), (int)(nu + c + k), Ss[t][k] };
            }
            c += rk[t];
        } else for (int q = 0; q < A->nnz; q++) T[q] = (LDTrip){ ucol[A->row[q]] - 1, ucol[A->col[q]] - 1, A->val[q] };
        qsort(T, ne, sizeof(LDTrip), cmp_ldtrip);
        S->nnz = ne;
        S->row = xmalloc(sizeof(int) * (ne + 1)); S->col = xmalloc(sizeof(int) * (ne + 1)); S->val = xmalloc(sizeof(double) * (ne + 1));
        for (int q = 0; q < ne; q++) { S->row[q] = T[q].r; S->col[q] = T[q].c; S->val[q] = T[q].v; }
        free(T);
    }
    const int qd = nu + (int)nv;
    for (int t = 0; t < nc; t++) spsym_finish(&SS[t], qd, 0);
    B->dunit = xmalloc(sizeof(int) * (nu + 1));
    for (int p = 0; p < n; p++) if (ucol[p]) B->dunit[ucol[p] - 1] = p;
    free(ucol);
    for (int t = 0; t < nc; t++) { free(Ws[t]); free(Ss[t]); }
    free(rk); free(Ws); free(Ss);
    Block *vb = xcalloc(1, sizeof(Block));
    vb->type = BLK_SDP; vb->n = qd; vb->ncon = nc;
    vb->con = xmalloc(sizeof(int) * (nc + 1));
    memcpy(vb->con, B->con, sizeof(int) * nc);
    vb->A = SS;
    vb->C.nnz = 0;
    vb->C.row = xmalloc(sizeof(int)); vb->C.col = xmalloc(sizeof(int)); vb->C.val = xmalloc(sizeof(double));
    spsym_finish(&vb->C, qd, 0);
    Params pv = *par;
    pv.lowrank = 0; pv.dict = 0; pv.cls = 0;               /* P is singular: no factor-based route inside */
    const double cost_vb = analyze_sdp_block(vb, &pv);
    B->vb = vb; B->dV = V; B->dnu = nu; B->dnv = (int)nv; B->dstruct = nl; B->dlr = 1;
    const double nd = n;
    const double form = par->c_blas * 2.0 * (nd * nd * nv + nd * (double)nv * nv) + par->c_sparse * (double)qd * qd;
    if (getenv("BRISK_LRDBG")) printf("   [low-rank dictionary: n %d, %d of %d constraints by %ld factor columns, %d unit columns: cost %.3g + %.3g]\n", n, nl, nc, nv, nu, cost_vb, form);
    return cost_vb + form;
}

static double g_fact_ref = 0;   /* the work of a dense factorization of the Schur complement (problem_prepare) */
static double g_symg_min = -1;  /* the gather share from which the symmetric gather is used (BRISK_SYMGMIN) */
typedef struct { double ef; int t; } EfIdx;
static int cmp_ef_desc(const void *a, const void *b) {
    const EfIdx *x = a, *y = b;
    return (x->ef < y->ef) - (x->ef > y->ef);
}

/* Structural analysis of one SDP block. Cost model (flop units):
 *   cs = scalar random-access flop, cb = BLAS-3 flop.
 *   dense route      (t): cb*(2n^3 + nd*n^2) + cs*S_sparse
 *   row-product  (t): cb*nr_t*n^2 + cs*(ef_t*n + suffix_sparse)
 *   sparse-sparse(t): cs*ef_t*suffix_sparse                              */
double analyze_sdp_block(Block *B, const Params *par) {
    const int n = B->n, nc = B->ncon;
    const double n2 = (double)n * n, n3 = n2 * n;
    const double cs = par->c_sparse, cb = par->c_blas;

    /* union row set and union full pattern */
    unsigned char *pat = xcalloc((size_t)n * n, 1);
    char *mark = xcalloc(n, 1);
    double Stot = 0;
    for (int t = 0; t < nc; t++) {
        const SpSym *S = &B->A[t];
        Stot += S->ef;
        for (int q = 0; q < S->nr; q++) mark[S->rows[q]] = 1;
        for (int q = 0; q < S->ef; q++) pat[S->fr[q] + (size_t)S->fc[q] * n] = 1;
    }
    B->unr = 0;
    for (int p = 0; p < n; p++) B->unr += mark[p];
    B->urows = xmalloc(sizeof(int) * (B->unr ? B->unr : 1));
    for (int p = 0, c = 0; p < n; p++) if (mark[p]) B->urows[c++] = p;
    free(mark);
    size_t unf = 0;
    for (size_t k = 0; k < (size_t)n * n; k++) unf += pat[k];
    B->unf = (int)unf;
    B->ufr = xmalloc(sizeof(int) * (unf ? unf : 1));
    B->ufc = xmalloc(sizeof(int) * (unf ? unf : 1));
    {
        size_t c = 0;
        for (int j = 0; j < n; j++)
            for (int i = 0; i < n; i++)
                if (pat[i + (size_t)j * n]) { B->ufr[c] = i; B->ufc[c] = j; c++; }
    }
    free(pat);

    /* dense classification */
    B->route = xmalloc(nc ? nc : 1);
    B->dpos = xmalloc(sizeof(int) * (nc ? nc : 1));
    EfIdx *cand = xmalloc(sizeof(EfIdx) * (nc ? nc : 1));
    int ncand = 0;
    for (int t = 0; t < nc; t++) {
        B->dpos[t] = -1;
        if (B->A[t].ef >= 0.02 * n2 && n >= 8) { cand[ncand].ef = B->A[t].ef; cand[ncand].t = t; ncand++; }
    }
    qsort(cand, ncand, sizeof(EfIdx), cmp_ef_desc);
    double budget = par->dense_mem;
    int nd = 0;
    double cost_now = 0;
    /* 15 October 2026 (the owner's benchmark, ISSUES 96): a dense constraint of low rank (the row <J, X> = 0 of a
     * min-bisection relaxation, J = 11' written out with n(n+1)/2 entries) is a dense row whose product X A Zi is
     * sum_k sig_k (X w_k)(Zi w_k)': 3 n^2 r instead of 2 n^3 (G48_mb, n = 3000: the row cost 165 s of 313 on the
     * owner's machine; the block-level low-rank route is not taken there, every constraint of the block counted) */
    int *drk = xcalloc(ncand ? ncand : 1, sizeof(int)); double **drW = xcalloc(ncand ? ncand : 1, sizeof(double *)), **drS = xcalloc(ncand ? ncand : 1, sizeof(double *));
    for (int c = 0; c < ncand; c++) {
        const SpSym *S = &B->A[cand[c].t];
        double dense_cost = cb * (2 * n3 + ncand * n2) + cs * fmax(0.0, Stot - cand[c].ef * ncand);
        double row_cost = cb * S->nr * n2 + cs * (S->ef * n + 0.5 * Stot);
        int r = 0; double *W = NULL, *sg = NULL;
        /* (where the dense product costs 2e9 flops or more, n >= 1000: below that it takes a tenth of a second an
         * iteration at most, and the old path keeps the rounding of the small files: gpp500-1, n = 500, ended
         * at 1.5e-8 instead of 2.5e-9 by the other order of the operations) */
        if (par->lowrank != 0 && S->ef >= 0.25 * n2 && 2.0 * n3 >= 2e9 && !getenv("BRISK_NODRLR")) {
            r = lowrank_factor(S, n, 8, &W, &sg);     /* (the randomized range finder samples 8 directions at least) */
            if (r <= 0 || r > 8) { free(W); free(sg); W = NULL; sg = NULL; r = 0; }
        }
        if (r > 0) dense_cost = cb * 3.0 * n2 * r + cs * fmax(0.0, Stot - cand[c].ef * ncand);
        if ((r > 0 || dense_cost < row_cost) && (nd + 1) * n2 * 8.0 <= budget) {
            if (r > 0) { drk[nd] = r; drW[nd] = W; drS[nd] = sg; W = NULL; sg = NULL; }
            B->dpos[cand[c].t] = nd++; cost_now += dense_cost;
        }
        free(W); free(sg);
    }
    free(cand);
    /* dry run of the sparse-route costs, then the low-rank decision (before any dense copies) */
    {
        double suf = 0;
        for (int t = nc - 1; t >= 0; t--) {
            if (B->dpos[t] >= 0) continue;
            const SpSym *S = &B->A[t];
            suf += S->ef;
            double c3 = cs * S->ef * suf;
            double cr = cb * S->nr * n2 + cs * (S->ef * n + suf);
            cost_now += (cr < c3) ? cr : c3;
        }
    }
    /* dictionary route first (linear-time detection with gates), then low-rank;
     * the cheapest of {current routes, dictionary, low-rank} wins */
    /* 22 October 2026: the analysis's parts timed with BRISK_SETUPT (delay_indep_n32, a control LMI: 0.57 s of 0.86 s here) */
    static int setupt = -1; if (setupt < 0) setupt = getenv("BRISK_SETUPT") != NULL;
    double ts0 = setupt ? wtime() : 0, ts_dict = 0, ts_lrd = 0, ts_lr = 0, ts_cls = 0;
    double cost_dict = (par->dict != 0) ? dict_build(B, par) : 1e300;
    if (setupt) { ts_dict = wtime() - ts0; ts0 = wtime(); }
    if (!B->vb) {
        /* 5.18: the factors of low-rank constraints as the shared columns, with the gate of the entry-class route
         * below: cheaper by a fifth by the model, on a block whose assembly matters, and a saving of at least 4 % of a
         * dense factorization of the Schur complement (where it saves less it only moves the rounding: an
         * occupation-measure dual form, nn_n20_m100_K, OPTIMAL 1.0e-11 -> 1.0e-8 for no time) */
        const double c_lrd = lrdict_build(B, par);
        if (B->vb && !(par->dict > 0 || (c_lrd < 0.8 * cost_now && cost_now > 1e7 && cost_now - c_lrd >= 0.04 * g_fact_ref))) {
            if (getenv("BRISK_LRDBG")) printf("   [low-rank dictionary not taken: %.3g against %.3g, the factorization %.3g]\n", c_lrd, cost_now, g_fact_ref);
            dict_free(B); B->dlr = 0; cost_dict = 1e300;
        } else cost_dict = c_lrd;
    }
    if (setupt) { ts_lrd = wtime() - ts0; ts0 = wtime(); }
    if (!(B->vb && par->dict > 0)) lowrank_block(B, par, fmin(cost_now, cost_dict));
    if (setupt) { ts_lr = wtime() - ts0; ts0 = wtime(); }
    if (B->lowrank) dict_free(B);
    else if (B->vb && (par->dict > 0 || cost_dict < cost_now)) { B->dict = 1; cost_now = cost_dict; }
    else dict_free(B);
    /* the entry-class route (dictroute.c) where it is cheaper than the block's routes */
    class_free(B);
    if (!B->lowrank && !B->dict && par->cls != 0) {
        const double cost_cls = class_build(B, par);
        /* auto: cheaper by a fifth by the model (it overestimates the route: box (4,8) 0.55 by the
         * model, 0.65 measured), on a block whose assembly matters (above about ten milliseconds an
         * iteration), and a saving of at least 4 % of a dense factorization of the Schur complement:
         * where it saves less, the gain is lost in the noise while the other rounding moves the
         * ill-posed problems (moment relaxations of a user's benchmark: 8-240 %, all faster; on the
         * regression sets 0-1.2 % where it lost or tied, e_moment_quadknap 26 -> 70 iterations) */
        const int take = B->cb && (par->cls > 0 || (cost_cls < 0.8 * cost_now && cost_now > 1e7 && cost_now - cost_cls >= 0.04 * g_fact_ref));
        if (B->cb && getenv("BRISK_CLSDBG")) printf("   [block n %d: entry-class route %.3g against %.3g: %s]\n", n, cost_cls, cost_now, take ? "taken" : "not taken");
        if (take) { B->cls = 1; cost_now = cost_cls; }
        else class_free(B);
    }
    if (setupt) { ts_cls = wtime() - ts0; printf("   [setup: block n %d, %d constraints: dictionary %.3fs, low-rank dictionary %.3fs, low-rank %.3fs, entry classes %.3fs]\n", n, nc, ts_dict, ts_lrd, ts_lr, ts_cls); }
    if (B->lowrank || B->cls) {                /* all constraints treated as sparse */
        for (int t = 0; t < nc; t++) B->dpos[t] = -1;
        for (int a = 0; a < nd; a++) { free(drW[a]); free(drS[a]); }
        nd = 0;
    }
    /* (the dense indices are renumbered in con order below: the low-rank data follow their constraint) */
    B->drk = xcalloc(nd ? nd : 1, sizeof(int)); B->drW = xcalloc(nd ? nd : 1, sizeof(double *)); B->drS = xcalloc(nd ? nd : 1, sizeof(double *));
    B->nd = nd;
    B->ns = nc - nd;
    B->dlist = xmalloc(sizeof(int) * (nd ? nd : 1));
    B->slist = xmalloc(sizeof(int) * (B->ns ? B->ns : 1));
    B->Ad = NULL;
    B->dfac = 0;
    if (nd && B->dict && B->dlr && !getenv("BRISK_NODFAC")) {
        /* 5.18: with the low-rank dictionary the assembly reads no dense copy; the operators then take each dense
         * constraint through its factor (A_t(D) = sum s_k w_k' D w_k, A*(y) = W diag(s y) W': two products of n x n x R
         * instead of passes over nd n^2 copies; A-optimal design: 520 MB read a dozen times an iteration), when every
         * dense constraint has a factor that gives its entries back to 32 u of the largest */
        int *rk = xcalloc(nc, sizeof(int)); double **Ws = xcalloc(nc, sizeof(double *)), **Ss = xcalloc(nc, sizeof(double *));
        int okf = 1; long Rd = 0;
        for (int t = 0; t < nc && okf; t++) {
            if (B->dpos[t] < 0) continue;
            const SpSym *S = &B->A[t];
            const int r = lowrank_factor(S, n, 8, &Ws[t], &Ss[t]);
            if (r <= 0 || r > 8) { okf = 0; break; }
            if (r == 1) rank1_exact(S, n, Ws[t], Ss[t]);
            double amax = 0, emax = 0;
            for (int q = 0; q < S->ef; q++) {
                double v = 0;
                for (int k = 0; k < r; k++) v += Ss[t][k] * Ws[t][S->fr[q] + (size_t)k * n] * Ws[t][S->fc[q] + (size_t)k * n];
                amax = fmax(amax, fabs(S->fv[q])); emax = fmax(emax, fabs(v - S->fv[q]));
            }
            /* (entries outside the pattern: the factor's own values there) */
            /* the operators define the problem the solver sees, so the factor must give the entries back to the
             * rounding of data written with 15 to 16 digits (32 u of the largest: 2.3e-15 measured on such data): the rank-one factor from the pivot
             * column does; an eigendecomposition's n u does not (an occupation-measure dual form, factors of rank 4,
             * n = 120: OPTIMAL 1e-11 -> 1e-8 through them) */
            const double ftol = 32.0 * 1.1102230246251565e-16;
            if (getenv("BRISK_LRDBG") && emax > ftol * amax) printf("   [dense constraint %d: factor of rank %d gives its entries to %.2e of the largest]\n", t, r, emax / amax);
            if (emax > ftol * amax) okf = 0;
            rk[t] = r; Rd += r;
        }
        if (getenv("BRISK_LRDBG")) printf("   [low-rank dictionary: the operators through the factors: %s (%d dense constraints)]\n", okf ? "yes" : "no", nd);
        if (okf) {
            B->dfac = 1;
            B->dfoff = xmalloc(sizeof(int) * (nd + 1));
            B->dfW = xmalloc(sizeof(double) * (size_t)n * (Rd ? Rd : 1));
            B->dfS = xmalloc(sizeof(double) * (Rd ? Rd : 1));
            long c = 0; int a = 0;
            for (int t = 0; t < nc; t++) {
                if (B->dpos[t] < 0) continue;
                B->dfoff[a++] = (int)c;
                memcpy(B->dfW + (size_t)c * n, Ws[t], sizeof(double) * (size_t)n * rk[t]);
                memcpy(B->dfS + c, Ss[t], sizeof(double) * rk[t]);
                c += rk[t];
            }
            B->dfoff[a] = (int)c;
        }
        for (int t = 0; t < nc; t++) { free(Ws[t]); free(Ss[t]); }
        free(rk); free(Ws); free(Ss);
    }
    B->Adl = NULL;
    if (nd && !B->dfac) {
        if (posix_memalign((void **)&B->Ad, 64, (size_t)nd * n * n * sizeof(double))) exit(1);
        memset(B->Ad, 0, (size_t)nd * n * n * sizeof(double));
    }
    {
        int di = 0, si = 0;
        for (int t = 0; t < nc; t++) {
            if (B->dpos[t] >= 0) {
                /* renumber dense ids in con order */
                { const int a0 = B->dpos[t]; B->drk[di] = drk[a0]; B->drW[di] = drW[a0]; B->drS[di] = drS[a0]; }
                B->dpos[t] = di;
                B->dlist[di] = t;
                if (B->Ad) {
                    double *D = B->Ad + (size_t)di * n * n;
                    const SpSym *S = &B->A[t];
                    for (int q = 0; q < S->ef; q++) D[S->fr[q] + (size_t)S->fc[q] * n] = S->fv[q];
                }
                B->route[t] = 2;
                di++;
            } else B->slist[si++] = t;
        }
        free(drk); free(drW); free(drS);
        /* 17 October 2026: small blocks: the dense constraints' lower triangles packed by columns, so that A(D) and
         * A*(y) are one pass over n(n+1)/2 entries each, without a BLAS call per block (mater-5: 2,440 blocks of order
         * 11, 1.2 million calls of a 121 x 16 dgemv, 2.6 us each). BRISK_NOADL=1: the dgemv on the full copies. */
        if (B->Ad && n <= ADL_MAXN && !getenv("BRISK_NOADL")) {
            const int nl = n * (n + 1) / 2;
            B->Adl = xmalloc(sizeof(double) * (size_t)nd * nl);
            for (int a = 0; a < nd; a++) {
                const double *D = B->Ad + (size_t)a * n * n; double *o = B->Adl + (size_t)a * nl;
                int k = 0;
                for (int j = 0; j < n; j++) for (int i = j; i < n; i++) o[k++] = D[i + (size_t)j * n];
            }
        }
        if (getenv("BRISK_DRLRDBG")) for (int a = 0; a < nd; a++) if (B->drk[a]) printf("   [block n %d: dense constraint %d of rank %d: the low-rank row route]\n", n, B->con[B->dlist[a]], B->drk[a]);
    }

    /* flattened full entries of sparse constraints */
    size_t fe = 0;
    for (int a = 0; a < B->ns; a++) fe += B->A[B->slist[a]].ef;
    B->foff = xmalloc(sizeof(int) * (B->ns + 1));
    B->ffr = xmalloc(sizeof(int) * (fe ? fe : 1));
    B->ffc = xmalloc(sizeof(int) * (fe ? fe : 1));
    B->ffv = xmalloc(sizeof(double) * (fe ? fe : 1));
    fe = 0;
    for (int a = 0; a < B->ns; a++) {
        const SpSym *S = &B->A[B->slist[a]];
        B->foff[a] = (int)fe;
        memcpy(B->ffr + fe, S->fr, sizeof(int) * S->ef);
        memcpy(B->ffc + fe, S->fc, sizeof(int) * S->ef);
        memcpy(B->ffv + fe, S->fv, sizeof(double) * S->ef);
        fe += S->ef;
    }
    B->foff[B->ns] = (int)fe;
    B->ffp = (double)n * n < 2147483648.0 ? xmalloc(sizeof(int) * (fe ? fe : 1)) : NULL;
    if (B->ffp) for (size_t k = 0; k < fe; k++) B->ffp[k] = B->ffr[k] + B->ffc[k] * n;
    /* 4.23: the stored entries flattened the same way (blk_Aop: contiguous loads, and the
     * summation order of the per-constraint loop, so its rounding is unchanged) */
    size_t le = 0;
    for (int a = 0; a < B->ns; a++) le += B->A[B->slist[a]].nnz;
    B->lfoff = xmalloc(sizeof(int) * (B->ns + 1));
    B->lfr = xmalloc(sizeof(int) * (le ? le : 1));
    B->lfc = xmalloc(sizeof(int) * (le ? le : 1));
    B->lfv = xmalloc(sizeof(double) * (le ? le : 1));
    le = 0;
    for (int a = 0; a < B->ns; a++) {
        const SpSym *S = &B->A[B->slist[a]];
        B->lfoff[a] = (int)le;
        memcpy(B->lfr + le, S->row, sizeof(int) * S->nnz);
        memcpy(B->lfc + le, S->col, sizeof(int) * S->nnz);
        memcpy(B->lfv + le, S->val, sizeof(double) * S->nnz);
        le += S->nnz;
    }
    B->lfoff[B->ns] = (int)le;

    /* sparse-sparse vs row-product vs row-product on the union pattern (route 3: X A_t Zi
     * evaluated only where later constraints can read it, |U| nr_t instead of nr_t n^2;
     * pays off when the union pattern is sparse, e.g. moment matrices of AC-OPF) */
    double suf = 0;
    const int allow3 = !ENV_ON("BRISK_NOROUTE3");
    for (int a = B->ns - 1; a >= 0; a--) {
        const SpSym *S = &B->A[B->slist[a]];
        suf += S->ef;
        double c3 = cs * S->ef * suf;
        double cr = cb * S->nr * n2 + cs * (S->ef * n + suf);
        double cu = allow3 ? cs * (S->ef * n + 0.5 * (double)B->unf * S->nr + suf) : 1e300;
        B->route[B->slist[a]] = (c3 <= cr && c3 <= cu) ? 0 : (cr <= cu ? 1 : 3);
    }
    /* 16 October 2026 (the owner's benchmark, ISSUES 96a): a block whose constraints are all monomial matrices (one
     * entry per row and per column, at least half the rows: the Pauli-string constraints of quantum state
     * tomography, R(P) a signed permutation of order 2^(q+1)) takes the batched route: A_a Zi is a signed gather of
     * the rows of Zi, the products X (A_a Zi) of a chunk of constraints are one BLAS-3 call, and M_ab = <A_b, X A_a Zi>
     * is a gather of n entries per pair. The row products ran one small product per constraint at 7 GF/s. */
    /* (measured on qst_7q_15, n = 256, 2,458 constraints, two threads: the row-product route 2.3 s an assembly, the
     * batched route 2.3 s as well, its product at 71 GF/s in 1.15 s but the gather of the pairs 0.8 s and the row
     * gathers and the interleaving 0.4 s; both routes are bound by the gather of n entries per pair, m^2 n / 2 random
     * reads an assembly; an experiment, BRISK_MONO=1, until the gather is made to use the structure of the permutations) */
    B->allmono = 0;
    if (!B->lowrank && !B->dict && !B->cls && B->nd == 0 && B->ns >= 16 && n >= 32 && ENV_ON("BRISK_MONO")) {
        int ok = 1; char *mk = xcalloc(n, 1);
        for (int a = 0; a < B->ns && ok; a++) {
            const SpSym *S = &B->A[B->slist[a]];
            if (S->ef != S->nr || 2 * S->ef < n) { ok = 0; break; }
            memset(mk, 0, n);
            for (int e = 0; e < S->ef; e++) { if (mk[S->fc[e]]) { ok = 0; break; } mk[S->fc[e]] = 1; }
        }
        free(mk);
        if (ok) { B->allmono = 1; for (int a = 0; a < B->ns; a++) B->route[B->slist[a]] = 5; }
    }
    /* 5.18 (ISSUES 96a): Weyl operators. When n = 2^p and every constraint is A[c ^ x, c] = k (-1)^(z . c) for all c
     * (the Pauli strings of quantum state tomography in their real form: a translation of Z_2^p and a character), the
     * pair <A_b, X A_a Zi> = k_b sum_c (-1)^(z_b . c) G_a[c ^ x_b, c] is entry z_b of the Walsh-Hadamard transform of
     * the x_b-th XOR diagonal of G_a: n^2 log n an a for all b instead of a gather of n entries a pair (m^2 n / 2
     * random reads an assembly). BRISK_WEYL=0 off. */
    free(B->wx); free(B->wz); free(B->wk); B->wx = B->wz = NULL; B->wk = NULL; B->allweyl = 0; weylpd_free(B);
    if (getenv("BRISK_WEYLDBG")) printf("   [Weyl check: n %d ns %d nd %d lowrank %d dict %d cls %d]\n", n, B->ns, B->nd, B->lowrank, B->dict, B->cls);
    if (!B->lowrank && !B->dict && !B->cls && B->nd == 0 && B->ns >= 2 * n && n >= 32 && (n & (n - 1)) == 0
        && !(getenv("BRISK_WEYL") && atoi(getenv("BRISK_WEYL")) == 0)) {
        int ok = 1;
        int *wx = xmalloc(sizeof(int) * B->ns), *wz = xmalloc(sizeof(int) * B->ns); double *wk = xmalloc(sizeof(double) * B->ns);
        double *vc = xmalloc(sizeof(double) * n); char *have = xmalloc(n);
        for (int a = 0; a < B->ns && ok; a++) {
            const SpSym *S = &B->A[B->slist[a]];
            if (S->ef != n) { ok = 0; break; }
            memset(have, 0, n);
            const int x = S->fr[0] ^ S->fc[0];
            for (int e = 0; e < S->ef; e++) {
                if ((S->fr[e] ^ S->fc[e]) != x || have[S->fc[e]]) { ok = 0; break; }
                have[S->fc[e]] = 1; vc[S->fc[e]] = S->fv[e];
            }
            if (!ok) break;
            const double k = vc[0];
            if (k == 0) { ok = 0; break; }
            int z = 0;
            for (int bit = 1; bit < n; bit <<= 1) if (vc[bit] != k) { if (vc[bit] == -k) z |= bit; else { ok = 0; break; } }
            for (int c = 0; c < n && ok; c++) {
                const double want = __builtin_popcount(z & c) & 1 ? -k : k;
                if (vc[c] != want) ok = 0;
            }
            wx[a] = x; wz[a] = z; wk[a] = k;
        }
        free(vc); free(have);
        if (getenv("BRISK_WEYLDBG")) printf("   [Weyl operators: block n %d, %d constraints: %s]\n", n, B->ns, ok ? "all" : "not all");
        if (ok) { B->allweyl = 1; B->wx = wx; B->wz = wz; B->wk = wk; B->allmono = 1; for (int a = 0; a < B->ns; a++) B->route[B->slist[a]] = 5;
                  weylpd_setup(B); }   /* (23 October 2026: the pairs in the Pauli basis when that is cheaper, weylpd.c) */
        else { free(wx); free(wz); free(wk); }
    }

    /* 5.5: the live region of the row products. X A_a Zi is read only at the entries of the
     * constraints b >= a. An index whose last constraint is before a is dead for a: with the
     * indices in the order of their death (the longest-lived first) the live ones are a prefix,
     * and the rows that the entries of a column still need are a prefix too (lv_ch). The product
     * is then formed panel by panel on that staircase (solver.c). Kept when the staircases of
     * the row-product constraints hold at most 2/3 of nr n^2 (moment matrices in the order of
     * the moments: about 55 %) and the block has order 128 or more: measured gains at n = 204 ... 441
     * with shares of 53-58 %, losses of 8 % at a share of 74 % (n = 135) and of 15 % on blocks of
     * 55-66 (small products run at a lower rate, the gather is not reduced). */
    B->live = 0; B->lv_perm = B->lv_rank = B->lv_u = B->lv_cp = B->lv_ca = B->lv_cr = B->lv_ch = NULL; B->lv_cv = NULL;
    B->lu_cp = B->lu_ca = B->lu_cr = B->lu_ch = NULL; B->lu_cv = NULL; B->symg = 0; B->s2k = 0;
    {
        int n1 = 0;
        for (int a = 0; a < B->ns; a++) n1 += B->route[B->slist[a]] == 1;
        if (n1 > 0 && n >= 128 && fe > 0 && fe < ((size_t)1 << 28) && !B->cls && !ENV_ON("BRISK_NOLIVE")) {
            const int ns = B->ns;
            int *death = xmalloc(sizeof(int) * n), *cnt = xcalloc((size_t)ns + 2, sizeof(int));
            for (int i = 0; i < n; i++) death[i] = -1;
            for (int a = 0; a < ns; a++) { const SpSym *S = &B->A[B->slist[a]]; for (int k = 0; k < S->nr; k++) death[S->rows[k]] = a; }
            /* the order: by death, the last first (a counting sort; never-used indices at the end) */
            B->lv_perm = xmalloc(sizeof(int) * n); B->lv_rank = xmalloc(sizeof(int) * n); B->lv_u = xmalloc(sizeof(int) * (ns + 1));
            for (int i = 0; i < n; i++) cnt[death[i] + 1]++;             /* cnt[d + 1]: indices with death d */
            { int live = n - cnt[0];                                     /* indices used at all */
              for (int a = 0; a < ns; a++) { B->lv_u[a] = live; live -= cnt[a + 1]; } }
            { int *start = xmalloc(sizeof(int) * (ns + 2)); int o = 0;
              for (int d = ns - 1; d >= -1; d--) { start[d + 1] = o; o += cnt[d + 1]; }
              for (int i = 0; i < n; i++) { const int q = start[death[i] + 1]++; B->lv_perm[q] = i; B->lv_rank[i] = q; }
              free(start); }
            B->lv_cp = xcalloc((size_t)n + 2, sizeof(int));
            B->lv_ca = xmalloc(sizeof(int) * fe); B->lv_cr = xmalloc(sizeof(int) * fe); B->lv_ch = xmalloc(sizeof(int) * fe); B->lv_cv = xmalloc(sizeof(double) * fe);
            for (size_t k = 0; k < fe; k++) B->lv_cp[B->lv_rank[B->ffc[k]] + 2]++;
            for (int q = 0; q < n; q++) B->lv_cp[q + 2] += B->lv_cp[q + 1];
            for (int a = 0; a < ns; a++)
                for (int k = B->foff[a]; k < B->foff[a + 1]; k++) {
                    const int t = B->lv_cp[B->lv_rank[B->ffc[k]] + 1]++;
                    B->lv_ca[t] = a; B->lv_cr[t] = B->lv_rank[B->ffr[k]]; B->lv_cv[t] = B->ffv[k];
                }
            for (int q = 0; q < n; q++) {
                int hmax = 0;
                for (int t = B->lv_cp[q + 1] - 1; t >= B->lv_cp[q]; t--) { if (B->lv_cr[t] + 1 > hmax) hmax = B->lv_cr[t] + 1; B->lv_ch[t] = hmax; }
            }
            /* the staircases against the full products; the gather's entries (for the symmetric gather) */
            double full = 0, area = 0, gath = 0;
            {   double sf = 0;
                for (int a = ns - 1; a >= 0; a--) { sf += B->A[B->slist[a]].ef; if (B->route[B->slist[a]] == 1) gath += sf; } }
            /* (22 October 2026: the staircase of constraint a over the live prefix, sum over q < lv_u[a] of the
             * height of column q's entries of the constraints >= a, by one sweep from the last constraint that keeps
             * each column's height and the prefix's sum: linear in the entries, the same integers. A binary search per
             * constraint and column cost ns n log: fap25, DIMACS, an SDP with 2.2 million constraints on a block of
             * order 2,118, spent minutes here under a time limit of 900 s) */
            {   int *H = xcalloc(n, sizeof(int)); double *ar = xmalloc(sizeof(double) * (ns ? ns : 1)); double S = 0; int P = 0;
                for (int a = ns - 1; a >= 0; a--) {
                    for (int k = B->foff[a]; k < B->foff[a + 1]; k++) {
                        const int q = B->lv_rank[B->ffc[k]], h = B->lv_rank[B->ffr[k]] + 1;
                        if (h > H[q]) { if (q < P) S += h - H[q]; H[q] = h; }
                    }
                    while (P < B->lv_u[a]) S += H[P++];
                    ar[a] = S;
                }
                for (int a = 0; a < ns; a++) {
                    if (B->route[B->slist[a]] != 1) continue;
                    const SpSym *Sa = &B->A[B->slist[a]];
                    full += (double)Sa->nr * n2; area += (double)Sa->nr * ar[a];
                }
                free(H); free(ar); }
            /* 5.9 (a user's suggestion, refitted on his 24 relaxations with the flat gather of
             * 5.7): the ratio at which the live region stops paying grows with the block
             * order (measured, not derived). 0.67 up to n = 420 as before, then
             * 0.3026 ln n - 1.1578: 0.75 at n = 550, 0.86 at 777, 1 from n = 1250.
             * Measured (Schur formation, live against full): n = 777 at a ratio of 0.674
             * 20 % faster, 580 at 0.709 5-7 %, 550 at 0.735 2-5 %; 437 at 0.720 equal, 418
             * at 0.726 5-8 % slower, 352 at 0.697 20 % slower, 300-370 at 0.74-0.77 5-29 %
             * slower; his results on six threads: 860 at 0.68 and two blocks of 1296 and 2208
             * at 1.00 faster. (370 at 0.679 is 20 % faster too, next to 352 at 0.697: the
             * ratio is not the only determinant.) His fit on 1.2.1 was 0.215 ln n - 0.465
             * (BRISK_LIVE=2). */
            double thr = 0.3026 * log((double)n) - 1.1578;
            if (thr < 0.67) thr = 0.67;
            if (thr > 1.0) thr = 1.0;
            B->live = area <= thr * full;
            { const char *e = getenv("BRISK_LIVE");      /* test: 0/1 forces it; 2: his fit on 1.2.1; 3: the fixed 0.67 of 4.x-5.8 */
              if (e) { const int v = atoi(e); B->live = v == 2 ? area <= (0.215 * log((double)n) - 0.465) * full : v == 3 ? area <= 0.67 * full : v; } }
            /* 5.12: the symmetric gather halves the gather and doubles the inner dimension of the
             * products (their flops about the same, at a lower rate, and twice the copies): it pays
             * where the gather is a good share of the work. Measured: moment matrices with 15-27
             * rows a constraint, gather 0.4-0.8 of the products' cost units, 8-10 % faster; sphere
             * relaxations with 134 rows a constraint, 0.2, 12 % slower. */
            const double gshare = gath / fmax(2.0 * cb / cs * area, 1.0);
            if (ENV_ON("BRISK_ROUTEDBG")) printf("   [block n %d: live region of the row products: %.1f %% of nr n^2 (%s); gather %.3g entries, products %.3g flops, share %.2f]\n", n, 100.0 * area / fmax(full, 1.0), B->live ? "used" : "not used", gath, 2.0 * area, gshare);
            free(death); free(cnt);
            if (!B->live) { free(B->lv_perm); free(B->lv_rank); free(B->lv_u); free(B->lv_cp); free(B->lv_ca); free(B->lv_cr); free(B->lv_ch); free(B->lv_cv);
                            B->lv_perm = B->lv_rank = B->lv_u = B->lv_cp = B->lv_ca = B->lv_cr = B->lv_ch = NULL; B->lv_cv = NULL;
              if (!ENV_ON("BRISK_NOS2K")) {
                /* 18 October 2026: without the live region (its staircase saves nothing: random sparse data, every
                 * index read to the last constraint), the upper entries of the block's constraints by column in the
                 * original order, the diagonal halved, each column's entries by constraint: route 1 then forms
                 * H = X A_a Zi + Zi A_a X by one dsyr2k (its upper triangle, the flops of the full X A_a Zi) and
                 * reads it at the later constraints' upper entries column by column, half the entries of the full
                 * gather and in the order of the memory (the owner's benchmark, random sparse SDPs: the gather was
                 * half of the assembly). BRISK_NOS2K=1: the full product and gather as before */
                size_t fu = 0;
                for (size_t k = 0; k < fe; k++) fu += B->ffr[k] <= B->ffc[k];
                B->lu_cp = xcalloc((size_t)n + 2, sizeof(int));
                B->lu_ca = xmalloc(sizeof(int) * (fu + 1)); B->lu_cr = xmalloc(sizeof(int) * (fu + 1)); B->lu_cv = xmalloc(sizeof(double) * (fu + 1));
                for (size_t k = 0; k < fe; k++) if (B->ffr[k] <= B->ffc[k]) B->lu_cp[B->ffc[k] + 1]++;
                for (int q = 0; q < n; q++) B->lu_cp[q + 1] += B->lu_cp[q];
                int *pos = xmalloc(sizeof(int) * (n + 1));
                memcpy(pos, B->lu_cp, sizeof(int) * (n + 1));
                for (int a = 0; a < ns; a++)
                    for (int k = B->foff[a]; k < B->foff[a + 1]; k++) {
                        if (B->ffr[k] > B->ffc[k]) continue;
                        const int t = pos[B->ffc[k]]++;
                        B->lu_ca[t] = a; B->lu_cr[t] = B->ffr[k];
                        B->lu_cv[t] = B->ffr[k] == B->ffc[k] ? 0.5 * B->ffv[k] : B->ffv[k];
                    }
                free(pos);
                B->s2k = 1;
              }
            }
            else if (!ENV_ON("BRISK_NOSYMG")) {
                /* 5.12, the symmetric gather (solver.c): the upper triangle in the permuted order, in
                 * the same column-and-constraint order. BRISK_NOSYMG=1: the full entries as before.
                 * Built for every live block: the single-precision row products of the early
                 * iterations always take it (their products cost half, so halving the gather pays:
                 * box (4,8) of a user's set, assembly 13.3 -> 10.5 s); in double precision only from
                 * the gather share g_symg_min (symg) */
                B->symg = gshare >= g_symg_min || ENV_ON("BRISK_SYMG");
                size_t fu = 0;
                for (int q = 0; q < n; q++) for (int t = B->lv_cp[q]; t < B->lv_cp[q + 1]; t++) fu += B->lv_cr[t] <= q;
                B->lu_cp = xcalloc((size_t)n + 1, sizeof(int));
                B->lu_ca = xmalloc(sizeof(int) * (fu + 1)); B->lu_cr = xmalloc(sizeof(int) * (fu + 1));
                B->lu_ch = xmalloc(sizeof(int) * (fu + 1)); B->lu_cv = xmalloc(sizeof(double) * (fu + 1));
                size_t w = 0;
                for (int q = 0; q < n; q++) {
                    B->lu_cp[q] = (int)w;
                    for (int t = B->lv_cp[q]; t < B->lv_cp[q + 1]; t++) {
                        if (B->lv_cr[t] > q) continue;
                        B->lu_ca[w] = B->lv_ca[t]; B->lu_cr[w] = B->lv_cr[t];
                        B->lu_cv[w] = B->lv_cr[t] == q ? 0.5 * B->lv_cv[t] : B->lv_cv[t];
                        w++;
                    }
                }
                B->lu_cp[n] = (int)w;
                for (int q = 0; q < n; q++) {
                    int hmax = 0;
                    for (int t = B->lu_cp[q + 1] - 1; t >= B->lu_cp[q]; t--) { if (B->lu_cr[t] + 1 > hmax) hmax = B->lu_cr[t] + 1; B->lu_ch[t] = hmax; }
                }
            }
        }
    }

    if (ENV_ON("BRISK_ROUTEDBG")) {
        double sef = 0, snr = 0; int n1 = 0;
        for (int a = 0; a < B->ns; a++) { const SpSym *S = &B->A[B->slist[a]]; sef += S->ef; snr += S->nr; n1 += B->route[B->slist[a]] != 0; }
        printf("   [block n %d: %d sparse constraints (%d row-product), union pattern %d (%.1f per row), avg ef %.1f avg rows %.1f]\n",
               n, B->ns, n1, B->unf, (double)B->unf / n, sef / fmax(1, B->ns), snr / fmax(1, B->ns));
    }
    /* product route for Zi * (A'y) * R */
    double c_row = cb * 2.0 * B->unr * n2;
    double c_sl = 0.2 * cs * (double)B->unf * n + cb * n3;
    double c_dn = cb * 2.0 * n3;
    if (c_row <= c_sl && c_row <= c_dn) B->prod_route = 0;
    else if (c_sl <= c_dn) B->prod_route = 1;
    else B->prod_route = 2;
    B->asm_cost = B->lowrank ? 0.0 : cost_now / cb;
    return cost_now;
}

/* Scaling, cost-model routing and pattern analysis. */
void problem_prepare(Problem *P, const Params *par) {
    int m = P->m;
    double *nrm2 = xcalloc(m, sizeof(double));
    double c2 = 0, c1 = 0;
    for (int k = 0; k < P->nblk; k++) {
        Block *B = &P->blk[k];
        for (int t = 0; t < B->ncon; t++) {
            SpSym *S = &B->A[t];
            for (int q = 0; q < S->ef; q++) nrm2[B->con[t]] += S->fv[q] * S->fv[q];
        }
        for (int q = 0; q < B->C.ef; q++) { c2 += B->C.fv[q] * B->C.fv[q]; c1 += fabs(B->C.fv[q]); }
    }
    P->normC2 = sqrt(c2);
    P->normC1 = c1;
    P->normb1 = P->normb2 = 0;
    for (int i = 0; i < m; i++) { if (P->tbR != 0 && P->b0[i] == P->tbR) continue; P->normb1 += fabs(P->b0[i]); P->normb2 += P->b0[i] * P->b0[i]; }
    P->normb2 = sqrt(P->normb2);

    /* row equilibration of the constraint operator */
    P->d = xmalloc(sizeof(double) * m);
    for (int i = 0; i < m; i++) {
        P->d[i] = nrm2[i] > 0 ? 1.0 / sqrt(nrm2[i]) : 1.0;
        if (nrm2[i] == 0 && P->b[i] != 0)
            fprintf(stderr, "brisk: warning: constraint %d is empty but b != 0 (infeasible)\n", i + 1);
    }
    free(nrm2);
    double nb = 0;
    for (int i = 0; i < m; i++) if (!(P->tbR != 0 && P->b0[i] == P->tbR)) nb += P->d[i] * P->b[i] * P->d[i] * P->b[i];
    nb = sqrt(nb);
    P->bs = nb > 1 ? nb : 1.0;
    /* 4.30: the trace-bound row is scaled so that its rhs is 1 in the scaled problem (its
     * entries are then bs/R, tiny): with the equilibrated row the rhs R would dominate every
     * measure and the starting point */
    if (P->tbR != 0) for (int i = 0; i < m; i++) if (P->b0[i] == P->tbR) P->d[i] = P->bs / P->tbR;
    for (int k = 0; k < P->nblk; k++) {
        Block *B = &P->blk[k];
        for (int t = 0; t < B->ncon; t++) spsym_scale(&B->A[t], P->d[B->con[t]]);
    }
    for (int i = 0; i < m; i++) P->b[i] *= P->d[i];
    P->du = xmalloc(sizeof(double) * (m + 1));
    for (int i = 0; i < m; i++) P->du[i] = P->bs / P->d[i] * ((P->tbR != 0 && P->b0[i] == P->tbR) ? (1.0 + P->normb2) / P->tbR : 1.0);
    P->cs = P->normC2 > 1 ? P->normC2 : 1.0;
    for (int i = 0; i < m; i++) P->b[i] /= P->bs;
    for (int k = 0; k < P->nblk; k++) spsym_scale(&P->blk[k].C, 1.0 / P->cs);
    P->scaled = 1;

    g_fact_ref = par->c_blas * (double)m * m * m / 3.0;
    if (g_symg_min < 0) { const char *e = getenv("BRISK_SYMGMIN"); g_symg_min = e ? atof(e) : 0.3; }
    for (int k = 0; k < P->nblk; k++) {
        Block *B = &P->blk[k];
        int n = B->n;
        if (B->type == BLK_LP) {
            int *cnt = xcalloc(n + 1, sizeof(int));
            int tot = 0;
            for (int t = 0; t < B->ncon; t++)
                for (int q = 0; q < B->A[t].nnz; q++) { cnt[B->A[t].row[q] + 1]++; tot++; }
            for (int p = 0; p < n; p++) cnt[p+1] += cnt[p];
            B->lp_ptr = xmalloc(sizeof(int) * (n + 1));
            memcpy(B->lp_ptr, cnt, sizeof(int) * (n + 1));
            B->lp_con = xmalloc(sizeof(int) * (tot ? tot : 1));
            B->lp_val = xmalloc(sizeof(double) * (tot ? tot : 1));
            for (int t = 0; t < B->ncon; t++)      /* con ascending => lists sorted */
                for (int q = 0; q < B->A[t].nnz; q++) {
                    int p = B->A[t].row[q];
                    B->lp_con[cnt[p]] = B->con[t];
                    B->lp_val[cnt[p]] = B->A[t].val[q];
                    cnt[p]++;
                }
            free(cnt);
            continue;
        }
        if (brisk_block_as_cone(par, B)) { Params pc = *par; pc.lowrank = 0; pc.dict = 0; analyze_sdp_block(B, &pc); }
        else analyze_sdp_block(B, par);
    }
}

static void spsym_free(SpSym *S) {
    free(S->row); free(S->col); free(S->val);
    free(S->fr); free(S->fc); free(S->fv); free(S->rows);
}

void block_free_contents(Block *B) {
    for (int t = 0; t < B->ncon; t++) spsym_free(&B->A[t]);
    free(B->A); free(B->con);
    spsym_free(&B->C);
    free(B->route); free(B->urows); free(B->ufr); free(B->ufc);
    free(B->dpos); free(B->dlist); free(B->slist); free(B->Ad); free(B->Adl);
    free(B->dfoff); free(B->dfW); free(B->dfS); B->dfoff = NULL; B->dfW = NULL; B->dfS = NULL; B->dfac = 0;
    free(B->wx); free(B->wz); free(B->wk); B->wx = B->wz = NULL; B->wk = NULL; B->allweyl = 0; weylpd_free(B);
    free(B->ltr_off); free(B->ltr_t); free(B->ltr_v); B->ltr_off = B->ltr_t = NULL; B->ltr_v = NULL;
    if (B->drk) { for (int a = 0; a < B->nd; a++) { free(B->drW[a]); free(B->drS[a]); } }
    free(B->drk); free(B->drW); free(B->drS);
    free(B->foff); free(B->ffr); free(B->ffc); free(B->ffv); free(B->ffp);
    free(B->lv_perm); free(B->lv_rank); free(B->lv_u); free(B->lv_cp); free(B->lv_ca); free(B->lv_cr); free(B->lv_ch); free(B->lv_cv);
    free(B->lu_cp); free(B->lu_ca); free(B->lu_cr); free(B->lu_ch); free(B->lu_cv);
    free(B->lfoff); free(B->lfr); free(B->lfc); free(B->lfv);
    free(B->lr_off); free(B->lr_W); free(B->lr_sig);
    free(B->lp_ptr); free(B->lp_con); free(B->lp_val);
    dict_free(B);
    class_free(B);
    memset(B, 0, sizeof(*B));
}

void problem_free(Problem *P) {
    for (int k = 0; k < P->nblk; k++) block_free_contents(&P->blk[k]);
    free(P->blk); free(P->b); free(P->b0); free(P->d); free(P->du); free(P->orig);
    memset(P, 0, sizeof(*P));
}

/* 4.30: an exact copy of an unprepared problem (data as triplets, rebuilt), for undoing a
 * presolve step that is not kept (facial reduction rejected by the depth rule: the
 * pipeline used to start over from the file, repeating the free elimination) */
/* 23 October 2026: the copy by arrays, when the blocks are in the form the rebuild from triplets gives (each matrix's
 * entries strictly increasing in (row, col), none zero, row <= col, an LP's col equal to its row, every constraint of a
 * block with an entry, the constraints ascending): the same problem without the sort and the rebuild (2.4 s for the
 * 18.8 million entries of a D-optimal design, copied before the facial reduction, which then removed nothing) */
static int spsym_canonical(const SpSym *S, int lp) {
    for (int q = 0; q < S->nnz; q++) {
        if (S->val[q] == 0.0 || S->row[q] > S->col[q] || (lp && S->col[q] != S->row[q])) return 0;
        if (q > 0 && (S->row[q - 1] > S->row[q] || (S->row[q - 1] == S->row[q] && S->col[q - 1] >= S->col[q]))) return 0;
    }
    return 1;
}
/* (the derived arrays recomputed as the rebuild does, from the entries: never a stale copy) */
static void spsym_copy(SpSym *D, const SpSym *S, int n, int lp) {
    const size_t z = S->nnz;
    D->nnz = S->nnz;
    D->row = xmalloc(sizeof(int) * (z ? z : 1)); D->col = xmalloc(sizeof(int) * (z ? z : 1)); D->val = xmalloc(sizeof(double) * (z ? z : 1));
    if (z) { memcpy(D->row, S->row, sizeof(int) * z); memcpy(D->col, S->col, sizeof(int) * z); memcpy(D->val, S->val, sizeof(double) * z); }
    spsym_finish(D, n, lp);
}
static int problem_clone_arrays(const Problem *P, Problem *Q) {
    const int m = P->m, nb = P->nblk;
    if (getenv("BRISK_CLONETRIP")) return 0;
    for (int k = 0; k < nb; k++) {
        const Block *B = &P->blk[k]; const int lp = B->type == BLK_LP;
        if (!B->C.row || !spsym_canonical(&B->C, lp)) return 0;
        for (int t = 0; t < B->ncon; t++) {
            const SpSym *S = &B->A[t];
            if (S->nnz == 0 || (t > 0 && B->con[t - 1] >= B->con[t]) || !spsym_canonical(S, lp)) return 0;
        }
    }
    memset(Q, 0, sizeof(*Q));
    Q->m = m; Q->nblk = nb;
    Q->blk = xcalloc(nb, sizeof(Block));
    for (int k = 0; k < nb; k++) {
        const Block *B = &P->blk[k]; Block *C = &Q->blk[k];
        C->type = B->type; C->n = B->n; C->ncon = B->ncon;
        C->con = xmalloc(sizeof(int) * (B->ncon ? B->ncon : 1));
        if (B->ncon) memcpy(C->con, B->con, sizeof(int) * B->ncon);
        C->A = xcalloc(B->ncon ? B->ncon : 1, sizeof(SpSym));
        for (int t = 0; t < B->ncon; t++) spsym_copy(&C->A[t], &B->A[t], B->n, B->type == BLK_LP);
        spsym_copy(&C->C, &B->C, B->n, B->type == BLK_LP);
    }
    Q->b = xmalloc(sizeof(double) * (m + 1)); Q->b0 = xmalloc(sizeof(double) * (m + 1)); Q->orig = xmalloc(sizeof(int) * (m + 1));
    memcpy(Q->b, P->b, sizeof(double) * m); memcpy(Q->b0, P->b0, sizeof(double) * m); memcpy(Q->orig, P->orig, sizeof(int) * m);
    Q->morig = P->morig; Q->tbR = P->tbR; Q->obj_off = P->obj_off;
    return 1;
}
void problem_clone(const Problem *P, Problem *Q) {
    if (problem_clone_arrays(P, Q)) return;
    const int m = P->m, nb = P->nblk;
    size_t nt = 0, ct = 1024;
    /* (23 October 2026: sized by a count first; the doubling copied the five arrays of 9.9 million entries of a D-optimal
     * design about twice over) */
    for (int k = 0; k < nb; k++) { const Block *B = &P->blk[k]; ct += B->C.nnz; for (int t = 0; t < B->ncon; t++) ct += B->A[t].nnz; }
    int *tc = xmalloc(sizeof(int) * ct), *tb = xmalloc(sizeof(int) * ct), *ti = xmalloc(sizeof(int) * ct), *tj = xmalloc(sizeof(int) * ct);
    double *tv = xmalloc(sizeof(double) * ct);
    for (int k = 0; k < nb; k++) {
        const Block *B = &P->blk[k];
        const int lp = B->type == BLK_LP;
        for (int t = -1; t < B->ncon; t++) {
            const SpSym *S = t < 0 ? &B->C : &B->A[t];
            const int con = t < 0 ? -1 : B->con[t];
            for (int q = 0; q < S->nnz; q++) {
                if (nt == ct) { ct *= 2; tc = realloc(tc, sizeof(int) * ct); tb = realloc(tb, sizeof(int) * ct); ti = realloc(ti, sizeof(int) * ct); tj = realloc(tj, sizeof(int) * ct); tv = realloc(tv, sizeof(double) * ct); }
                tc[nt] = con; tb[nt] = k; ti[nt] = S->row[q]; tj[nt] = lp ? S->row[q] : S->col[q]; tv[nt] = S->val[q]; nt++;
            }
        }
    }
    int *bsz = xmalloc(sizeof(int) * (nb + 1));
    for (int k = 0; k < nb; k++) bsz[k] = P->blk[k].type == BLK_LP ? -P->blk[k].n : P->blk[k].n;
    problem_from_trips(Q, m, nb, bsz, P->b, nt, tc, tb, ti, tj, tv);
    memcpy(Q->b0, P->b0, sizeof(double) * m);
    memcpy(Q->orig, P->orig, sizeof(int) * m);
    Q->morig = P->morig;
    Q->tbR = P->tbR;
    Q->obj_off = P->obj_off;
    free(tc); free(tb); free(ti); free(tj); free(tv); free(bsz);
}
