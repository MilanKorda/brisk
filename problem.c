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

/* Parse next number; skips non-numeric tokens (counted). Returns 0 at end of buffer. */
static long g_skipped_words = 0;
static int next_num(char **pp, double *out) {
    char *p = *pp;
    for (;;) {
        while (*p && isspace((unsigned char)*p)) p++;
        if (!*p) { *pp = p; return 0; }
        char *e;
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
static void build_blocks(Problem *P, Trip *T, size_t nt, long lower_entries) {
    const int nblk = P->nblk;
    qsort(T, nt, sizeof(Trip), cmp_trip);
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
    size_t r = 0;
    for (int k = 0; k < nblk; k++) {
        Block *B = &P->blk[k];
        size_t s = r;
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
        spsym_finish(&B->C, B->n, B->type == BLK_LP);
        for (int q = 0; q < ncon; q++) spsym_finish(&B->A[q], B->n, B->type == BLK_LP);
    }
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
    free(buf);
    if (g_skipped_words)
        fprintf(stderr, "brisk: warning: %ld non-numeric token(s) skipped in the data section\n", g_skipped_words);
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
static unsigned long lr_rng = 88172645463325252UL;
static double lr_rand(void) {
    lr_rng ^= lr_rng << 13; lr_rng ^= lr_rng >> 7; lr_rng ^= lr_rng << 17;
    return (double)(lr_rng >> 11) / 9007199254740992.0 - 0.5;
}

static void sp_mulvec(const SpSym *S, const double *x, double *y, int n) {   /* y = S x */
    memset(y, 0, sizeof(double) * n);
    for (int q = 0; q < S->ef; q++) y[S->fr[q]] += S->fv[q] * x[S->fc[q]];
}

static int lowrank_factor(const SpSym *S, int n, int kmax, double **Wout, double **sigout) {
    const int one = 1;
    double anrm = 0;
    for (int q = 0; q < S->ef; q++) anrm += S->fv[q] * S->fv[q];
    anrm = sqrt(anrm);
    if (anrm == 0) { *Wout = NULL; *sigout = NULL; return 0; }
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
    BL(dsyev_)("V", "U", &k, T, &k, ev, &wq, &lwork, &info);
    lwork = (int)wq + 1;
    double *work = xmalloc(sizeof(double) * lwork);
    BL(dsyev_)("V", "U", &k, T, &k, ev, work, &lwork, &info);
    free(work);
    if (info != 0) goto fail;
    double emax = 0;
    for (int j = 0; j < k; j++) emax = fmax(emax, fabs(ev[j]));
    rank = 0;
    for (int j = 0; j < k; j++) if (fabs(ev[j]) > 1e-14 * emax) rank++;
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
    for (int t = 0; t < nc && ok; t++) {
        rk[t] = lowrank_factor(&B->A[t], n, B->A[t].nr <= small ? small : kmax, &Ws[t], &Ss[t]);
        if (rk[t] < 0) ok = 0; else R += rk[t];
        if (ok && par->lowrank < 0) {
            double lbt = cb * (nd * nd * (R + (nc - t - 1)) + nd * (double)(R + nc - t - 1) * (R + nc - t - 1));
            if (lbt >= cost_now) ok = 0;
        }
    }
    double cost_lr = cb * (nd * nd * R + nd * (double)R * R) + 0.3 * cs * (double)R * R;
    double mem = 8.0 * (double)R * R;         /* single-buffer mode */
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
    for (int c = 0; c < ncand; c++) {
        const SpSym *S = &B->A[cand[c].t];
        double dense_cost = cb * (2 * n3 + ncand * n2) + cs * fmax(0.0, Stot - cand[c].ef * ncand);
        double row_cost = cb * S->nr * n2 + cs * (S->ef * n + 0.5 * Stot);
        if (dense_cost < row_cost && (nd + 1) * n2 * 8.0 <= budget) { B->dpos[cand[c].t] = nd++; cost_now += dense_cost; }
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
    double cost_dict = (par->dict != 0) ? dict_build(B, par) : 1e300;
    if (!(B->vb && par->dict > 0)) lowrank_block(B, par, fmin(cost_now, cost_dict));
    if (B->lowrank) dict_free(B);
    else if (B->vb && (par->dict > 0 || cost_dict < cost_now)) { B->dict = 1; cost_now = cost_dict; }
    else dict_free(B);
    if (B->lowrank) {                          /* all constraints treated as sparse */
        for (int t = 0; t < nc; t++) B->dpos[t] = -1;
        nd = 0;
    }
    B->nd = nd;
    B->ns = nc - nd;
    B->dlist = xmalloc(sizeof(int) * (nd ? nd : 1));
    B->slist = xmalloc(sizeof(int) * (B->ns ? B->ns : 1));
    B->Ad = NULL;
    if (nd) {
        if (posix_memalign((void **)&B->Ad, 64, (size_t)nd * n * n * sizeof(double))) exit(1);
        memset(B->Ad, 0, (size_t)nd * n * n * sizeof(double));
    }
    {
        int di = 0, si = 0;
        for (int t = 0; t < nc; t++) {
            if (B->dpos[t] >= 0) {
                /* renumber dense ids in con order */
                B->dpos[t] = di;
                B->dlist[di] = t;
                double *D = B->Ad + (size_t)di * n * n;
                const SpSym *S = &B->A[t];
                for (int q = 0; q < S->ef; q++) D[S->fr[q] + (size_t)S->fc[q] * n] = S->fv[q];
                B->route[t] = 2;
                di++;
            } else B->slist[si++] = t;
        }
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
        analyze_sdp_block(B, par);
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
    free(B->dpos); free(B->dlist); free(B->slist); free(B->Ad);
    free(B->foff); free(B->ffr); free(B->ffc); free(B->ffv);
    free(B->lfoff); free(B->lfr); free(B->lfc); free(B->lfv);
    free(B->lr_off); free(B->lr_W); free(B->lr_sig);
    free(B->lp_ptr); free(B->lp_con); free(B->lp_val);
    dict_free(B);
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
void problem_clone(const Problem *P, Problem *Q) {
    const int m = P->m, nb = P->nblk;
    size_t nt = 0, ct = 1024;
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
