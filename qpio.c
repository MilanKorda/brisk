/* qpio.c - QPS files for the QP solver (the SOCP developer, 22 October 2026): MPS with a quadratic objective section.
 * Sections: NAME, OBJSENSE (MAX / MAXIMIZE, also inline), ROWS, COLUMNS (MARKER lines skipped: integers read as
 * continuous), RHS (on the objective row: minus the constant), RANGES, BOUNDS (UP LO FX FR MI PL BV LI UI), QUADOBJ (the
 * lower triangle of H), QMATRIX and QSECTION (H in full: both triangles; only the lower is kept). Free format (fields
 * separated by blanks); a ROWS line with more than two fields switches to the fixed format (names with blanks).
 * The objective is 1/2 x'Hx + c'x + c0, the QPS convention. An UP bound below zero on a column without a lower bound
 * makes the lower bound -infinity (the MPS rule). The first N row is the objective, other N rows are dropped. */
#include "qp.h"
#include "fastnum.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <ctype.h>
#include <stdint.h>

static void *xm(size_t s) { void *p = malloc(s ? s : 1); if (!p) { fprintf(stderr, "brisk_qp: out of memory\n"); exit(2); } return p; }
static void *xz(size_t s) { void *p = calloc(1, s ? s : 1); if (!p) { fprintf(stderr, "brisk_qp: out of memory\n"); exit(2); } return p; }
static void *xr(void *p, size_t s) { p = realloc(p, s ? s : 1); if (!p) { fprintf(stderr, "brisk_qp: out of memory\n"); exit(2); } return p; }
static char *xdup(const char *s) { const size_t l = strlen(s); char *d = xm(l + 1); memcpy(d, s, l + 1); return d; }

/* names to indices: open addressing */
typedef struct { char **key; int *val; size_t cap, n; } Hash;
static uint64_t hstr(const char *s) { uint64_t h = 1469598103934665603ULL; while (*s) { h ^= (unsigned char)*s++; h *= 1099511628211ULL; } return h; }
static void h_init(Hash *H, size_t cap) { H->cap = cap; H->n = 0; H->key = xz(sizeof(char *) * cap); H->val = xm(sizeof(int) * cap); }
static void h_free(Hash *H) { free(H->key); free(H->val); }
static int h_get(const Hash *H, const char *k) {
    size_t i = hstr(k) & (H->cap - 1);
    while (H->key[i]) { if (!strcmp(H->key[i], k)) return H->val[i]; i = (i + 1) & (H->cap - 1); }
    return -1;
}
static void h_put(Hash *H, char *k, int v) {
    if (2 * (H->n + 1) > H->cap) {
        Hash N; h_init(&N, 2 * H->cap);
        for (size_t i = 0; i < H->cap; i++) if (H->key[i]) h_put(&N, H->key[i], H->val[i]);
        h_free(H); *H = N;
    }
    size_t i = hstr(k) & (H->cap - 1);
    while (H->key[i]) { if (!strcmp(H->key[i], k)) { H->val[i] = v; return; } i = (i + 1) & (H->cap - 1); }
    H->key[i] = k; H->val[i] = v; H->n++;
}

static int split(char *s, char **tok, int maxt) {
    int n = 0;
    while (*s && n < maxt) {
        while (*s == ' ' || *s == '\t' || *s == '\r' || *s == '\n') s++;
        if (!*s) break;
        tok[n++] = s;
        while (*s && *s != ' ' && *s != '\t' && *s != '\r' && *s != '\n') s++;
        if (*s) *s++ = 0;
    }
    return n;
}
/* fixed MPS fields: 2-3, 5-12, 15-22, 25-36, 40-47, 50-61 (1-based); trailing blanks of names removed */
static int split_fixed(char *s, char **tok, int nfield_names) {
    static char buf[6][64];
    const int st[6] = { 1, 4, 14, 24, 39, 49 }, ln[6] = { 2, 8, 8, 12, 8, 12 };
    const int L = (int)strlen(s);
    int n = 0;
    for (int f = 0; f < 6; f++) {
        if (st[f] >= L) break;
        int a = st[f], b = st[f] + ln[f]; if (b > L) b = L;
        int k = 0; for (int i = a; i < b && s[i] != '\n' && s[i] != '\r'; i++) buf[f][k++] = s[i];
        buf[f][k] = 0;
        while (k > 0 && (buf[f][k - 1] == ' ' || buf[f][k - 1] == '\t')) buf[f][--k] = 0;
        char *p = buf[f]; while (*p == ' ') p++;
        if (f == 0) { if (*p) tok[n++] = p; continue; }
        (void)nfield_names;
        if (*p) tok[n++] = p;
    }
    return n;
}

int qp_is_qps(const char *fname) {
    FILE *f = fopen(fname, "r"); if (!f) return 0;
    char line[512]; int q = 0, rows = 0, k = 0;
    while (fgets(line, sizeof line, f) && k++ < 2000000) {
        if (!strncmp(line, "ROWS", 4)) rows = 1;
        if (!strncmp(line, "QUADOBJ", 7) || !strncmp(line, "QMATRIX", 7) || !strncmp(line, "QSECTION", 8) || !strncmp(line, "QCMATRIX", 8)) { q = 1; break; }
    }
    fclose(f);
    return rows && q;
}

typedef struct { int i, j; double v; } Trip;
typedef struct { int r, i, j; double v; } QTrip;      /* a quadratic constraint's entry: row r, Q's (i, j), i >= j */
static int qtrip_cmp(const void *a, const void *b) {
    const QTrip *x = a, *y = b;
    if (x->r != y->r) return x->r < y->r ? -1 : 1;
    if (x->j != y->j) return x->j < y->j ? -1 : 1;
    return (x->i > y->i) - (x->i < y->i);
}

static int read_(const char *fname, QpProb *P, char *msg, size_t nmsg, int fixed) {
    memset(P, 0, sizeof(*P));
    FILE *f = fopen(fname, "r");
    if (!f) { snprintf(msg, nmsg, "cannot open %s", fname); return 1; }
    enum { S_NONE, S_ROWS, S_COLS, S_RHS, S_RANGES, S_BOUNDS, S_QUAD, S_QC, S_OBJSENSE, S_SKIP } sec = S_NONE;
    long ccap = 1 << 10, nc = 0; QTrip *C = xm(sizeof(QTrip) * ccap); int qcr = -1;     /* QCMATRIX entries */
    int qfull = 0;                                   /* QMATRIX / QSECTION: both triangles given */
    int mcap = 1024, mall = 0, objrow = -1;
    char *rtype = xm(mcap); char **rn = xm(sizeof(char *) * mcap);
    Hash HR, HC; h_init(&HR, 1024); h_init(&HC, 1024);
    int ncap = 1024, n = 0; char **cn = xm(sizeof(char *) * ncap);
    long zcap = 1 << 16, nz = 0; Trip *Z = xm(sizeof(Trip) * zcap);     /* A's entries and the objective's (i = -1) */
    long qcap = 1 << 12, nq = 0; Trip *Q = xm(sizeof(Trip) * qcap);
    double *rhs = NULL, *rng = NULL, *lo = NULL, *up = NULL; char *hasrng = NULL, *loset = NULL;
    double c0 = 0;
    char *line = NULL; size_t lcap = 0; long lineno = 0;
    char *tok[8];
    int rc = 0, cols_done = 0, ended = 0;
    char lastcol[256] = ""; int lastj = -1;
    while (getline(&line, &lcap, f) >= 0) {
        lineno++;
        if (line[0] == '*' || line[0] == '\n' || line[0] == '\r' || line[0] == 0) continue;
        if (line[0] != ' ' && line[0] != '\t') {
            const int nt = split(line, tok, 4);
            if (nt == 0) continue;
            if (!cols_done && sec == S_COLS && strcmp(tok[0], "COLUMNS")) {
                cols_done = 1;
                rhs = xz(sizeof(double) * (mall + 1)); rng = xz(sizeof(double) * (mall + 1)); hasrng = xz(mall + 1);
                lo = xz(sizeof(double) * (n + 1)); up = xm(sizeof(double) * (n + 1)); loset = xz(n + 1);
                for (int j = 0; j < n; j++) up[j] = QP_INF;
            }
            if (!strcmp(tok[0], "NAME")) { if (nt > 1) snprintf(P->name, sizeof P->name, "%s", tok[1]); sec = S_NONE; }
            else if (!strcmp(tok[0], "ROWS")) sec = S_ROWS;
            else if (!strcmp(tok[0], "COLUMNS")) sec = S_COLS;
            else if (!strcmp(tok[0], "RHS")) sec = S_RHS;
            else if (!strcmp(tok[0], "RANGES")) sec = S_RANGES;
            else if (!strcmp(tok[0], "BOUNDS")) sec = S_BOUNDS;
            else if (!strcmp(tok[0], "QUADOBJ")) { sec = S_QUAD; qfull = 0; }
            else if (!strcmp(tok[0], "QMATRIX") || !strcmp(tok[0], "QSECTION")) { sec = S_QUAD; qfull = 1; }
            else if (!strcmp(tok[0], "OBJSENSE")) { sec = S_OBJSENSE; if (nt > 1 && (tok[1][0] == 'M' || tok[1][0] == 'm') && (tok[1][1] == 'A' || tok[1][1] == 'a')) P->maxim = 1; }
            else if (!strcmp(tok[0], "ENDATA")) { ended = 1; break; }
            else if (!strcmp(tok[0], "QCMATRIX")) {
                if (nt < 2 || (qcr = h_get(&HR, tok[1])) < 0) { snprintf(msg, nmsg, "%s:%ld: QCMATRIX needs a known row", fname, lineno); rc = 1; break; }
                sec = S_QC;
            }
            else sec = S_SKIP;
            if (sec != S_COLS && sec != S_ROWS && sec != S_NONE && sec != S_OBJSENSE && sec != S_SKIP && !cols_done) {
                /* a file without COLUMNS lines (no variables): allocate now */
                cols_done = 1;
                rhs = xz(sizeof(double) * (mall + 1)); rng = xz(sizeof(double) * (mall + 1)); hasrng = xz(mall + 1);
                lo = xz(sizeof(double) * (n + 1)); up = xm(sizeof(double) * (n + 1)); loset = xz(n + 1);
                for (int j = 0; j < n; j++) up[j] = QP_INF;
            }
            continue;
        }
        int nt = (fixed && (sec == S_ROWS || sec == S_COLS || sec == S_RHS || sec == S_RANGES || sec == S_BOUNDS || sec == S_QUAD))
                 ? split_fixed(line, tok, 0) : split(line, tok, 7);
        /* (28 October 2026) a name beginning with '$' after the line's first field begins a comment (the MPS convention; as
         * the MPS reader, lpio.c: a MIPLIB 2017 file with "C0063540  LB_full  0  $ empty column" was refused) */
        if (sec == S_COLS || sec == S_RHS || sec == S_RANGES || sec == S_BOUNDS || sec == S_QUAD)
            for (int t = 1; t < nt; t++) if (tok[t][0] == '$') { nt = t; break; }
        if (nt == 0) continue;
        if (sec == S_OBJSENSE) { if ((tok[0][0] == 'M' || tok[0][0] == 'm') && (tok[0][1] == 'A' || tok[0][1] == 'a')) P->maxim = 1; continue; }
        if (sec == S_ROWS) {
            if (!fixed && nt > 2) { rc = 3; break; }
            if (nt < 2) { snprintf(msg, nmsg, "%s:%ld: a ROWS line needs a type and a name", fname, lineno); rc = 1; break; }
            const char t = (char)toupper((unsigned char)tok[0][0]);
            if (t != 'N' && t != 'E' && t != 'L' && t != 'G') { snprintf(msg, nmsg, "%s:%ld: row type %s", fname, lineno, tok[0]); rc = 1; break; }
            if (mall == mcap) { mcap *= 2; rtype = xr(rtype, mcap); rn = xr(rn, sizeof(char *) * mcap); }
            rtype[mall] = t; rn[mall] = xdup(tok[1]);
            if (t == 'N' && objrow < 0) objrow = mall;
            h_put(&HR, rn[mall], mall); mall++;
        } else if (sec == S_COLS) {
            if (nt >= 2 && !strcmp(tok[1], "'MARKER'")) continue;
            if (nt < 3) { snprintf(msg, nmsg, "%s:%ld: a COLUMNS line needs a column, a row and a value", fname, lineno); rc = 1; break; }
            int j;
            if (lastj >= 0 && !strcmp(lastcol, tok[0])) j = lastj;
            else {
                j = h_get(&HC, tok[0]);
                if (j < 0) {
                    if (n == ncap) { ncap *= 2; cn = xr(cn, sizeof(char *) * ncap); }
                    cn[n] = xdup(tok[0]); h_put(&HC, cn[n], n); j = n++;
                }
                snprintf(lastcol, sizeof lastcol, "%s", tok[0]); lastj = j;
            }
            for (int k = 1; k + 1 < nt; k += 2) {
                const int r = h_get(&HR, tok[k]);
                if (r < 0) { snprintf(msg, nmsg, "%s:%ld: unknown row %s", fname, lineno, tok[k]); rc = 1; break; }
                const double v = fast_strtod(tok[k + 1], NULL);
                if (rtype[r] == 'N' && r != objrow) continue;
                if (nz == zcap) { zcap *= 2; Z = xr(Z, sizeof(Trip) * zcap); }
                Z[nz].i = r == objrow ? -1 : r; Z[nz].j = j; Z[nz].v = v; nz++;
            }
            if (rc) break;
        } else if (sec == S_RHS || sec == S_RANGES) {
            /* the first field may be the set's name (absent in some files): pairs from the end */
            const int k0 = (nt % 2 == 1) ? 1 : 0;
            for (int k = k0; k + 1 < nt; k += 2) {
                const int r = h_get(&HR, tok[k]);
                if (r < 0) { snprintf(msg, nmsg, "%s:%ld: unknown row %s", fname, lineno, tok[k]); rc = 1; break; }
                const double v = fast_strtod(tok[k + 1], NULL);
                if (sec == S_RHS) { if (r == objrow) c0 = -v; else rhs[r] = v; }
                else { rng[r] = v; hasrng[r] = 1; }
            }
            if (rc) break;
        } else if (sec == S_BOUNDS) {
            if (nt < 2) continue;
            char *type = tok[0]; const char *col; double v = 0;
            if (!strcmp(type, "FR") || !strcmp(type, "MI") || !strcmp(type, "PL") || !strcmp(type, "BV")) {
                col = nt >= 3 ? tok[2] : tok[1];
                if (nt == 2) col = tok[1];
            } else {
                if (nt < 3) { snprintf(msg, nmsg, "%s:%ld: a bound needs a value", fname, lineno); rc = 1; break; }
                col = nt >= 4 ? tok[2] : tok[1]; v = fast_strtod(nt >= 4 ? tok[3] : tok[2], NULL);
            }
            const int j = h_get(&HC, col);
            if (j < 0) { snprintf(msg, nmsg, "%s:%ld: unknown column %s", fname, lineno, col); rc = 1; break; }
            if (!strcmp(type, "UP") || !strcmp(type, "UI")) { up[j] = v; if (v < 0 && !loset[j] && lo[j] == 0) lo[j] = -QP_INF; }
            else if (!strcmp(type, "LO") || !strcmp(type, "LI")) { lo[j] = v; loset[j] = 1; }
            else if (!strcmp(type, "FX")) { lo[j] = up[j] = v; loset[j] = 1; }
            else if (!strcmp(type, "FR")) { lo[j] = -QP_INF; up[j] = QP_INF; loset[j] = 1; }
            else if (!strcmp(type, "MI")) { lo[j] = -QP_INF; loset[j] = 1; }
            else if (!strcmp(type, "PL")) up[j] = QP_INF;
            else if (!strcmp(type, "BV")) { lo[j] = 0; up[j] = 1; loset[j] = 1; }
            else { snprintf(msg, nmsg, "%s:%ld: bound type %s", fname, lineno, type); rc = 1; break; }
        } else if (sec == S_QUAD) {
            if (nt < 3) continue;
            const int a = h_get(&HC, tok[0]), b = h_get(&HC, tok[1]);
            if (a < 0 || b < 0) { snprintf(msg, nmsg, "%s:%ld: unknown column in the quadratic section", fname, lineno); rc = 1; break; }
            const double v = fast_strtod(tok[2], NULL);
            int i = a, j = b; if (i < j) { const int t = i; i = j; j = t; }       /* lower triangle: row i >= column j */
            if (qfull && a < b) continue;                  /* both triangles given: the upper one dropped */
            if (nq == qcap) { qcap *= 2; Q = xr(Q, sizeof(Trip) * qcap); }
            Q[nq].i = i; Q[nq].j = j; Q[nq].v = v; nq++;
        } else if (sec == S_QC) {
            /* M listed in full (both triangles): the lower one kept, Q_k = 2 M */
            if (nt < 3) continue;
            const int a = h_get(&HC, tok[0]), b = h_get(&HC, tok[1]);
            if (a < 0 || b < 0) { snprintf(msg, nmsg, "%s:%ld: unknown column in QCMATRIX", fname, lineno); rc = 1; break; }
            if (a < b) continue;
            if (nc == ccap) { ccap *= 2; C = xr(C, sizeof(QTrip) * ccap); }
            C[nc].r = qcr; C[nc].i = a; C[nc].j = b; C[nc].v = 2 * fast_strtod(tok[2], NULL); nc++;
        }
    }
    free(line); fclose(f);
    /* (24 October 2026) a file that ends before ENDATA is an error: cut at a line boundary it would be read as a smaller
     * problem (a CBLIB file joined while the disk was full was solved as another problem) */
    if (!rc && !ended) { snprintf(msg, nmsg, "%s: the file ends before ENDATA (cut short?)", fname); rc = 1; }
    if (!rc && !cols_done) {
        cols_done = 1;
        rhs = xz(sizeof(double) * (mall + 1)); rng = xz(sizeof(double) * (mall + 1)); hasrng = xz(mall + 1);
        lo = xz(sizeof(double) * (n + 1)); up = xm(sizeof(double) * (n + 1)); loset = xz(n + 1);
        for (int j = 0; j < n; j++) up[j] = QP_INF;
    }
    if (!rc) {
        /* rows: the constraint rows in file order (N rows dropped) */
        int *rmap = xm(sizeof(int) * (mall + 1)); int m = 0;
        for (int r = 0; r < mall; r++) rmap[r] = rtype[r] == 'N' ? -1 : m++;
        P->n = n; P->m = m; P->c0 = c0;
        P->c = xz(sizeof(double) * (n + 1)); P->rl = xm(sizeof(double) * (m + 1)); P->ru = xm(sizeof(double) * (m + 1));
        P->xl = xm(sizeof(double) * (n + 1)); P->xu = xm(sizeof(double) * (n + 1));
        P->rname = xm(sizeof(char *) * (m + 1)); P->cname = cn;
        for (int r = 0; r < mall; r++) {
            const int i = rmap[r]; if (i < 0) continue;
            P->rname[i] = rn[r]; rn[r] = NULL;
            const double b = rhs[r], R = rng[r];
            switch (rtype[r]) {
                case 'E':
                    if (!hasrng[r]) P->rl[i] = P->ru[i] = b;
                    else if (R >= 0) { P->rl[i] = b; P->ru[i] = b + R; }
                    else { P->rl[i] = b + R; P->ru[i] = b; }
                    break;
                case 'L': P->ru[i] = b; P->rl[i] = hasrng[r] ? b - fabs(R) : -QP_INF; break;
                case 'G': P->rl[i] = b; P->ru[i] = hasrng[r] ? b + fabs(R) : QP_INF; break;
            }
        }
        for (int j = 0; j < n; j++) { P->xl[j] = lo[j]; P->xu[j] = up[j]; }
        /* A by columns, duplicates summed */
        long na = 0; for (long q = 0; q < nz; q++) { if (Z[q].i < 0) P->c[Z[q].j] += Z[q].v; else na++; }
        P->Ap = xz(sizeof(int) * (n + 1)); P->Ai = xm(sizeof(int) * (na + 1)); P->Ax = xm(sizeof(double) * (na + 1));
        for (long q = 0; q < nz; q++) if (Z[q].i >= 0) P->Ap[Z[q].j + 1]++;
        for (int j = 0; j < n; j++) P->Ap[j + 1] += P->Ap[j];
        int *pos = xm(sizeof(int) * (n + 1)); memcpy(pos, P->Ap, sizeof(int) * n);
        for (long q = 0; q < nz; q++) if (Z[q].i >= 0) { const int p = pos[Z[q].j]++; P->Ai[p] = rmap[Z[q].i]; P->Ax[p] = Z[q].v; }
        /* sort each column by row, sum duplicates */
        int *mark = xm(sizeof(int) * (m + 1)); for (int i = 0; i < m; i++) mark[i] = -1;
        int w = 0;
        for (int j = 0; j < n; j++) {
            const int a = P->Ap[j], b = P->Ap[j + 1]; P->Ap[j] = w; const int w0 = w;
            for (int p = a; p < b; p++) {
                const int i = P->Ai[p];
                if (mark[i] >= w0) P->Ax[mark[i]] += P->Ax[p];
                else { mark[i] = w; P->Ai[w] = i; P->Ax[w] = P->Ax[p]; w++; }
            }
            /* insertion sort by row (columns are short) */
            for (int p = w0 + 1; p < w; p++) { const int ki = P->Ai[p]; const double kv = P->Ax[p]; int q = p - 1; while (q >= w0 && P->Ai[q] > ki) { P->Ai[q + 1] = P->Ai[q]; P->Ax[q + 1] = P->Ax[q]; q--; } P->Ai[q + 1] = ki; P->Ax[q + 1] = kv; }
        }
        P->Ap[n] = w;
        free(mark); free(pos);
        /* H (lower triangle) by columns, duplicates summed, sorted */
        if (nq > 0) {
            P->Hp = xz(sizeof(int) * (n + 1)); P->Hi = xm(sizeof(int) * (nq + 1)); P->Hx = xm(sizeof(double) * (nq + 1));
            for (long q = 0; q < nq; q++) P->Hp[Q[q].j + 1]++;
            for (int j = 0; j < n; j++) P->Hp[j + 1] += P->Hp[j];
            int *ps = xm(sizeof(int) * (n + 1)); memcpy(ps, P->Hp, sizeof(int) * n);
            for (long q = 0; q < nq; q++) { const int p = ps[Q[q].j]++; P->Hi[p] = Q[q].i; P->Hx[p] = Q[q].v; }
            int *mk = xm(sizeof(int) * (n + 1)); for (int i = 0; i < n; i++) mk[i] = -1;
            int ww = 0;
            for (int j = 0; j < n; j++) {
                const int a = P->Hp[j], b = P->Hp[j + 1]; P->Hp[j] = ww; const int w0 = ww;
                for (int p = a; p < b; p++) {
                    const int i = P->Hi[p];
                    if (mk[i] >= w0) P->Hx[mk[i]] += P->Hx[p];
                    else { mk[i] = ww; P->Hi[ww] = i; P->Hx[ww] = P->Hx[p]; ww++; }
                }
                for (int p = w0 + 1; p < ww; p++) { const int ki = P->Hi[p]; const double kv = P->Hx[p]; int q = p - 1; while (q >= w0 && P->Hi[q] > ki) { P->Hi[q + 1] = P->Hi[q]; P->Hx[q + 1] = P->Hx[q]; q--; } P->Hi[q + 1] = ki; P->Hx[q + 1] = kv; }
            }
            P->Hp[n] = ww;
            free(ps); free(mk);
        }
        /* the quadratic constraints: entries sorted by (row, column, row index of Q), duplicates summed */
        if (nc > 0) {
            for (long q = 0; q < nc; q++) C[q].r = rmap[C[q].r];
            long w = 0; for (long q = 0; q < nc; q++) if (C[q].r >= 0) C[w++] = C[q];      /* (an N row's QCMATRIX: dropped) */
            nc = w;
            qsort(C, nc, sizeof(QTrip), qtrip_cmp);
            int nk = 0; for (long q = 0; q < nc; q++) if (q == 0 || C[q].r != C[q - 1].r) nk++;
            P->nqc = nk; P->qrow = xm(sizeof(int) * (nk + 1)); P->qs = xm(sizeof(int) * (nk + 1));
            P->qi = xm(sizeof(int) * (nc + 1)); P->qj = xm(sizeof(int) * (nc + 1)); P->qv = xm(sizeof(double) * (nc + 1));
            int k = -1, e = 0;
            for (long q = 0; q < nc; q++) {
                if (q == 0 || C[q].r != C[q - 1].r) { k++; P->qrow[k] = C[q].r; P->qs[k] = e; }
                else if (C[q].i == C[q - 1].i && C[q].j == C[q - 1].j) { P->qv[e - 1] += C[q].v; continue; }
                P->qi[e] = C[q].i; P->qj[e] = C[q].j; P->qv[e] = C[q].v; e++;
            }
            P->qs[nk] = e;
        }
        if (P->maxim) {      /* max f = - min (-f): the objective negated (H too: a concave maximization is a convex minimization) */
            for (int j = 0; j < n; j++) P->c[j] = -P->c[j];
            P->c0 = -P->c0;
            if (P->Hp) for (int p = 0; p < P->Hp[n]; p++) P->Hx[p] = -P->Hx[p];
        }
        free(rmap);
        cn = NULL;
    }
    for (int r = 0; r < mall; r++) free(rn[r]);
    free(rn); free(rtype); h_free(&HR); h_free(&HC);
    if (cn) { for (int j = 0; j < n; j++) free(cn[j]); free(cn); }
    free(Z); free(Q); free(C); free(rhs); free(rng); free(hasrng); free(lo); free(up); free(loset);
    if (rc) qp_prob_free(P);
    return rc;
}

int qp_read_qps(const char *fname, QpProb *P, char *msg, size_t nmsg) {
    msg[0] = 0;
    const int rc = read_(fname, P, msg, nmsg, 0);
    if (rc == 3) return read_(fname, P, msg, nmsg, 1);      /* names with blanks: the fixed format */
    return rc;
}

/* the problem written as a free-format QPS file with plain names (R<i>, C<j>): for readers that fail on the fixed
 * format's names with blanks (the comparison harness reads every file through this form). A maximization is
 * written as one, the data negated back. Returns 0 on success. */
int qp_write_qps(const QpProb *P, const char *fname) {
    FILE *f = fopen(fname, "w");
    if (!f) return 1;
    const double sg = P->maxim ? -1.0 : 1.0;
    fprintf(f, "NAME %s\n", P->name[0] ? P->name : "QP");
    if (P->maxim) fprintf(f, "OBJSENSE\n    MAX\n");
    fprintf(f, "ROWS\n N OBJ\n");
    for (int i = 0; i < P->m; i++) {
        const double l = P->rl[i], u = P->ru[i];
        fprintf(f, " %s R%d\n", l == u ? "E" : l > -QP_INF ? "G" : u < QP_INF ? "L" : "N", i);
    }
    fprintf(f, "COLUMNS\n");
    for (int j = 0; j < P->n; j++) {
        if (P->c[j] != 0) fprintf(f, " C%d OBJ %.17g\n", j, sg * P->c[j]);
        for (int p = P->Ap[j]; p < P->Ap[j + 1]; p++) fprintf(f, " C%d R%d %.17g\n", j, P->Ai[p], P->Ax[p]);
        if (P->c[j] == 0 && P->Ap[j] == P->Ap[j + 1]) fprintf(f, " C%d OBJ 0\n", j);
    }
    fprintf(f, "RHS\n");
    if (P->c0 != 0) fprintf(f, " RHS OBJ %.17g\n", -sg * P->c0);
    for (int i = 0; i < P->m; i++) {
        const double l = P->rl[i], u = P->ru[i];
        const double r = l == u ? l : l > -QP_INF ? l : u < QP_INF ? u : 0;
        if (r != 0) fprintf(f, " RHS R%d %.17g\n", i, r);
    }
    fprintf(f, "RANGES\n");
    for (int i = 0; i < P->m; i++) if (P->rl[i] > -QP_INF && P->ru[i] < QP_INF && P->rl[i] != P->ru[i]) fprintf(f, " RNG R%d %.17g\n", i, P->ru[i] - P->rl[i]);
    fprintf(f, "BOUNDS\n");
    for (int j = 0; j < P->n; j++) {
        const double l = P->xl[j], u = P->xu[j];
        if (l == u) { fprintf(f, " FX BND C%d %.17g\n", j, l); continue; }
        if (l <= -QP_INF && u >= QP_INF) { fprintf(f, " FR BND C%d\n", j); continue; }
        if (l <= -QP_INF) fprintf(f, " MI BND C%d\n", j); else if (l != 0 || u < 0) fprintf(f, " LO BND C%d %.17g\n", j, l);
        if (u < QP_INF) fprintf(f, " UP BND C%d %.17g\n", j, u);
    }
    if (P->Hp && P->Hp[P->n] > 0) {
        fprintf(f, "QUADOBJ\n");
        for (int j = 0; j < P->n; j++) for (int p = P->Hp[j]; p < P->Hp[j + 1]; p++) fprintf(f, " C%d C%d %.17g\n", j, P->Hi[p], sg * P->Hx[p]);
    }
    for (int k = 0; k < P->nqc; k++) {             /* M = Q_k / 2 in full */
        fprintf(f, "QCMATRIX R%d\n", P->qrow[k]);
        for (int e = P->qs[k]; e < P->qs[k + 1]; e++) {
            fprintf(f, " C%d C%d %.17g\n", P->qi[e], P->qj[e], 0.5 * P->qv[e]);
            if (P->qi[e] != P->qj[e]) fprintf(f, " C%d C%d %.17g\n", P->qj[e], P->qi[e], 0.5 * P->qv[e]);
        }
    }
    fprintf(f, "ENDATA\n");
    return fclose(f) != 0;
}

/* the problem as read, in plain text for other programs (the comparison harness: every solver gets exactly the problem
 * BRISK read): "n m nnzH nnzA c0 maxim", then n lines c_j xl_j xu_j, m lines rl_i ru_i (+-1e300 for infinite), nnzH
 * lines "i j v" (H's lower triangle, 1/2 x'Hx), nnzA lines "i j v"; a maximization written as read (negated back). */
int qp_write_text(const QpProb *P, const char *fname) {
    FILE *f = fopen(fname, "w");
    if (!f) return 1;
    const double sg = P->maxim ? -1.0 : 1.0;
    const int nh = P->Hp ? P->Hp[P->n] : 0;
    fprintf(f, "%d %d %d %d %.17g %d\n", P->n, P->m, nh, P->Ap[P->n], sg * P->c0, P->maxim);
    for (int j = 0; j < P->n; j++) fprintf(f, "%.17g %.17g %.17g\n", sg * P->c[j], P->xl[j] <= -QP_INF ? -1e300 : P->xl[j], P->xu[j] >= QP_INF ? 1e300 : P->xu[j]);
    for (int i = 0; i < P->m; i++) fprintf(f, "%.17g %.17g\n", P->rl[i] <= -QP_INF ? -1e300 : P->rl[i], P->ru[i] >= QP_INF ? 1e300 : P->ru[i]);
    for (int j = 0; j < P->n && nh; j++) for (int p = P->Hp[j]; p < P->Hp[j + 1]; p++) fprintf(f, "%d %d %.17g\n", P->Hi[p], j, sg * P->Hx[p]);
    for (int j = 0; j < P->n; j++) for (int p = P->Ap[j]; p < P->Ap[j + 1]; p++) fprintf(f, "%d %d %.17g\n", P->Ai[p], j, P->Ax[p]);
    return fclose(f) != 0;
}
