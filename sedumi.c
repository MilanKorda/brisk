/* sedumi.c - problems in SeDuMi form (5.4):   min c'x  s.t.  A x = b,  x in K
 *   K = R^f x R^l_+ x Q^{q_1} x ... x Qr^{r_1} x ... x S^{s_1}_+ x ...   (K.f, K.l, K.q, K.r, K.s)
 *
 *  - without semidefinite blocks the problem goes to the cone solver (socp.c);
 *  - with them it is written in SDPA form for the semidefinite solver: free variables as split
 *    pairs, a second-order cone of dimension d as a d x d block with arrow-shaped data (the dual
 *    form z = c - A'y in Q  <=>  Arw(z) >= 0, exact, no extra constraint; x0 = tr X, x_t = 2 X_0t);
 *  - a reader for MATLAB MAT-files (level 5, as written by save -v6 / -v7: plain or compressed
 *    elements), with its own inflate: variables A or At, b, c, K.
 */
#include "brisk.h"
#include "socp.h"
#include "sedumi.h"
#include <string.h>
#ifdef _OPENMP
#include <omp.h>
#endif
#include <math.h>

/* ------------------------------------------------------------------ inflate (RFC 1951) */
typedef struct { const unsigned char *in; size_t inlen, incnt; unsigned char *out; size_t outlen, outcnt; int bitbuf, bitcnt; int err; } Infl;
static int ibits(Infl *s, int need) {
    long val = s->bitbuf;
    while (s->bitcnt < need) {
        if (s->incnt == s->inlen) { s->err = 1; return 0; }
        val |= (long)(s->in[s->incnt++]) << s->bitcnt;
        s->bitcnt += 8;
    }
    s->bitbuf = (int)(val >> need);
    s->bitcnt -= need;
    return (int)(val & ((1L << need) - 1));
}
static int iout(Infl *s, size_t extra) {
    if (s->outcnt + extra > s->outlen) {
        size_t nl = s->outlen * 2 + extra + 65536;
        unsigned char *p = (unsigned char *)realloc(s->out, nl);
        if (!p) { s->err = 2; return 1; }
        s->out = p; s->outlen = nl;
    }
    return 0;
}
typedef struct { short count[16]; short symbol[288]; } Huff;
static int idecode(Infl *s, const Huff *h) {
    int code = 0, first = 0, index = 0;
    for (int len = 1; len <= 15; len++) {
        code |= ibits(s, 1);
        if (s->err) return -1;
        const int count = h->count[len];
        if (code - count < first) return h->symbol[index + (code - first)];
        index += count; first += count; first <<= 1; code <<= 1;
    }
    s->err = 1;
    return -1;
}
static int iconstruct(Huff *h, const short *length, int n) {
    short offs[16];
    for (int len = 0; len <= 15; len++) h->count[len] = 0;
    for (int sym = 0; sym < n; sym++) h->count[length[sym]]++;
    if (h->count[0] == n) return 0;
    int left = 1;
    for (int len = 1; len <= 15; len++) { left <<= 1; left -= h->count[len]; if (left < 0) return left; }
    offs[1] = 0;
    for (int len = 1; len < 15; len++) offs[len + 1] = (short)(offs[len] + h->count[len]);
    for (int sym = 0; sym < n; sym++) if (length[sym] != 0) h->symbol[offs[length[sym]]++] = (short)sym;
    return left;
}
static int icodes(Infl *s, const Huff *lencode, const Huff *distcode) {
    static const short lens[29] = { 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 17, 19, 23, 27, 31, 35, 43, 51, 59, 67, 83, 99, 115, 131, 163, 195, 227, 258 };
    static const short lext[29] = { 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 0 };
    static const short dists[30] = { 1, 2, 3, 4, 5, 7, 9, 13, 17, 25, 33, 49, 65, 97, 129, 193, 257, 385, 513, 769, 1025, 1537, 2049, 3073, 4097, 6145, 8193, 12289, 16385, 24577 };
    static const short dext[30] = { 0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13 };
    for (;;) {
        int sym = idecode(s, lencode);
        if (sym < 0 || s->err) return 1;
        if (sym < 256) { if (iout(s, 1)) return 1; s->out[s->outcnt++] = (unsigned char)sym; }
        else if (sym == 256) return 0;
        else {
            sym -= 257;
            if (sym >= 29) return 1;
            const int len = lens[sym] + ibits(s, lext[sym]);
            sym = idecode(s, distcode);
            if (sym < 0 || sym >= 30) return 1;
            const size_t dist = (size_t)dists[sym] + (size_t)ibits(s, dext[sym]);
            if (s->err || dist > s->outcnt) return 1;
            if (iout(s, (size_t)len)) return 1;
            for (int i = 0; i < len; i++) { s->out[s->outcnt] = s->out[s->outcnt - dist]; s->outcnt++; }
        }
    }
}
static int sd_inflate(const unsigned char *src, size_t srclen, unsigned char **dst, size_t *dstlen) {
    Infl S; memset(&S, 0, sizeof(S));
    S.in = src; S.inlen = srclen;
    int last;
    do {
        last = ibits(&S, 1);
        const int type = ibits(&S, 2);
        if (S.err) break;
        if (type == 0) {
            S.bitbuf = 0; S.bitcnt = 0;
            if (S.incnt + 4 > S.inlen) { S.err = 1; break; }
            const unsigned len = S.in[S.incnt] | ((unsigned)S.in[S.incnt + 1] << 8);
            S.incnt += 4;
            if (S.incnt + len > S.inlen || iout(&S, len)) { S.err = 1; break; }
            memcpy(S.out + S.outcnt, S.in + S.incnt, len); S.outcnt += len; S.incnt += len;
        } else if (type == 1) {
            Huff lc, dc; short lengths[288];
            int sym = 0;
            for (; sym < 144; sym++) lengths[sym] = 8;
            for (; sym < 256; sym++) lengths[sym] = 9;
            for (; sym < 280; sym++) lengths[sym] = 7;
            for (; sym < 288; sym++) lengths[sym] = 8;
            iconstruct(&lc, lengths, 288);
            for (sym = 0; sym < 30; sym++) lengths[sym] = 5;
            iconstruct(&dc, lengths, 30);
            if (icodes(&S, &lc, &dc)) { S.err = 1; break; }
        } else if (type == 2) {
            static const short order[19] = { 16, 17, 18, 0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15 };
            short lengths[320]; Huff lc, dc;
            const int nlen = ibits(&S, 5) + 257, ndist = ibits(&S, 5) + 1, ncode = ibits(&S, 4) + 4;
            if (S.err || nlen > 286 || ndist > 30) { S.err = 1; break; }
            int index = 0;
            for (; index < ncode; index++) lengths[order[index]] = (short)ibits(&S, 3);
            for (; index < 19; index++) lengths[order[index]] = 0;
            if (iconstruct(&lc, lengths, 19) != 0) { S.err = 1; break; }
            index = 0;
            while (index < nlen + ndist) {
                int sym = idecode(&S, &lc);
                if (sym < 0) { S.err = 1; break; }
                if (sym < 16) lengths[index++] = (short)sym;
                else {
                    int len = 0, rep;
                    if (sym == 16) { if (index == 0) { S.err = 1; break; } len = lengths[index - 1]; rep = 3 + ibits(&S, 2); }
                    else if (sym == 17) rep = 3 + ibits(&S, 3);
                    else rep = 11 + ibits(&S, 7);
                    if (index + rep > nlen + ndist) { S.err = 1; break; }
                    while (rep--) lengths[index++] = (short)len;
                }
            }
            if (S.err || lengths[256] == 0) { S.err = 1; break; }
            int e1 = iconstruct(&lc, lengths, nlen);
            if (e1 < 0 || (e1 > 0 && nlen - lc.count[0] != 1)) { S.err = 1; break; }
            e1 = iconstruct(&dc, lengths + nlen, ndist);
            if (e1 < 0 || (e1 > 0 && ndist - dc.count[0] != 1)) { S.err = 1; break; }
            if (icodes(&S, &lc, &dc)) { S.err = 1; break; }
        } else { S.err = 1; break; }
    } while (!last);
    if (S.err) { free(S.out); return 1; }
    *dst = S.out; *dstlen = S.outcnt;
    return 0;
}

/* ------------------------------------------------------------------ MAT-file (level 5) */
typedef struct {
    char name[64];
    int cls, nd, dims[4];
    int sparse; int *ir, *jc; long nzmax;      /* sparse: row indices, column pointers */
    double *pr; long npr;                      /* values (dense: column-major) */
    int nfields; char (*fname)[64]; const unsigned char **fdata; size_t *flen;   /* struct: the fields as raw matrix elements */
} MatVar;
static void matvar_free(MatVar *v) { free(v->ir); free(v->jc); free(v->pr); free(v->fname); free(v->fdata); free(v->flen); memset(v, 0, sizeof(*v)); }
static int g_mat_swap = 0;       /* the file is big-endian */
static unsigned rd32(const unsigned char *p) {
    return g_mat_swap ? p[3] | ((unsigned)p[2] << 8) | ((unsigned)p[1] << 16) | ((unsigned)p[0] << 24)
                      : p[0] | ((unsigned)p[1] << 8) | ((unsigned)p[2] << 16) | ((unsigned)p[3] << 24);
}
static void mat_cp(void *dst, const unsigned char *src, int n) {
    unsigned char *o = (unsigned char *)dst;
    if (g_mat_swap) for (int i = 0; i < n; i++) o[i] = src[n - 1 - i]; else memcpy(o, src, (size_t)n);
}
/* a data element at p: type, length, pointer to the data; returns the bytes consumed (0: error) */
static size_t mat_tag(const unsigned char *p, size_t avail, unsigned *type, size_t *len, const unsigned char **data) {
    if (avail < 8) return 0;
    const unsigned w = rd32(p);
    if (w >> 16) { *type = w & 0xffff; *len = w >> 16; *data = p + 4; if (*len > 4) return 0; return 8; }
    *type = w; *len = rd32(p + 4); *data = p + 8;
    if (*len > avail - 8) return 0;
    return 8 + ((*len + 7) & ~(size_t)7);
}
static long mat_count(unsigned type, size_t len) {
    const int sz = type == 1 || type == 2 ? 1 : type == 3 || type == 4 ? 2 : type == 5 || type == 6 || type == 7 ? 4 : type == 9 || type == 12 || type == 13 ? 8 : 0;
    return sz ? (long)(len / (size_t)sz) : -1;
}
static double mat_get(unsigned type, const unsigned char *d, long i) {
    switch (type) {
        case 1: return (double)((const signed char *)d)[i];
        case 2: return (double)d[i];
        case 3: { short v; mat_cp(&v, d + 2 * i, 2); return v; }
        case 4: { unsigned short v; mat_cp(&v, d + 2 * i, 2); return v; }
        case 5: { int v; mat_cp(&v, d + 4 * i, 4); return v; }
        case 6: { unsigned v; mat_cp(&v, d + 4 * i, 4); return v; }
        case 7: { float v; mat_cp(&v, d + 4 * i, 4); return v; }
        case 9: { double v; mat_cp(&v, d + 8 * i, 8); return v; }
        case 12: { long long v; mat_cp(&v, d + 8 * i, 8); return (double)v; }
        case 13: { unsigned long long v; mat_cp(&v, d + 8 * i, 8); return (double)v; }
    }
    return 0.0;
}
/* parses the body of a miMATRIX element */
static int mat_matrix(const unsigned char *p, size_t len, MatVar *v) {
    memset(v, 0, sizeof(*v));
    unsigned type; size_t l, used; const unsigned char *d;
    if (len == 0) return 0;                                                   /* an empty matrix */
    if (!(used = mat_tag(p, len, &type, &l, &d)) || l < 8) return 1;          /* array flags */
    const unsigned flags = rd32(d);
    v->cls = (int)(flags & 0xff); v->nzmax = (long)rd32(d + 4);
    if (flags & 0x0800) return 2;                                             /* complex */
    p += used; len -= used;
    if (!(used = mat_tag(p, len, &type, &l, &d))) return 1;                   /* dimensions */
    v->nd = (int)(l / 4); if (v->nd > 4) return 3;
    for (int i = 0; i < v->nd; i++) v->dims[i] = (int)rd32(d + 4 * i);
    p += used; len -= used;
    if (!(used = mat_tag(p, len, &type, &l, &d))) return 1;                   /* name */
    { const size_t c = l < 63 ? l : 63; memcpy(v->name, d, c); v->name[c] = 0; }
    p += used; len -= used;
    if (v->cls == 5) {                                                        /* sparse */
        v->sparse = 1;
        if (!(used = mat_tag(p, len, &type, &l, &d))) return 1;
        const long nir = (long)(l / 4);
        v->ir = (int *)malloc(sizeof(int) * (size_t)(nir + 1));
        for (long i = 0; i < nir; i++) v->ir[i] = (int)rd32(d + 4 * i);
        p += used; len -= used;
        if (!(used = mat_tag(p, len, &type, &l, &d))) return 1;
        const long njc = (long)(l / 4);
        if (njc < (long)v->dims[1] + 1) return 1;
        v->jc = (int *)malloc(sizeof(int) * (size_t)(njc + 1));
        for (long i = 0; i < njc; i++) v->jc[i] = (int)rd32(d + 4 * i);
        p += used; len -= used;
        if (v->jc[v->dims[1]] > nir) return 1;
        const long nnz = v->jc[v->dims[1]];
        v->pr = (double *)calloc((size_t)(nnz + 1), sizeof(double)); v->npr = nnz;
        if (len >= 8) {
            if (!(used = mat_tag(p, len, &type, &l, &d))) return 1;
            const long cnt = mat_count(type, l);
            if (cnt < nnz) return 1;
            for (long i = 0; i < nnz; i++) v->pr[i] = mat_get(type, d, i);
        } else if (nnz > 0) return 1;
        for (long i = 0; i < nnz; i++) if (v->ir[i] < 0 || v->ir[i] >= v->dims[0]) return 1;
    } else if (v->cls >= 6 && v->cls <= 15) {                                 /* numeric */
        if (!(used = mat_tag(p, len, &type, &l, &d))) return 1;
        const long cnt = mat_count(type, l);
        long want = 1; for (int i = 0; i < v->nd; i++) want *= v->dims[i];
        if (cnt < want) return 1;
        v->pr = (double *)malloc(sizeof(double) * (size_t)(want + 1)); v->npr = want;
        for (long i = 0; i < want; i++) v->pr[i] = mat_get(type, d, i);
    } else if (v->cls == 2) {                                                 /* struct (1 x 1) */
        if (!(used = mat_tag(p, len, &type, &l, &d))) return 1;
        const int fl = (int)rd32(d);
        p += used; len -= used;
        if (!(used = mat_tag(p, len, &type, &l, &d)) || fl < 1) return 1;
        v->nfields = (int)(l / (size_t)fl);
        v->fname = calloc((size_t)v->nfields + 1, 64); v->fdata = calloc((size_t)v->nfields + 1, sizeof(*v->fdata)); v->flen = calloc((size_t)v->nfields + 1, sizeof(size_t));
        for (int f = 0; f < v->nfields; f++) { const size_t c = (size_t)(fl < 63 ? fl : 63); memcpy(v->fname[f], d + (size_t)f * fl, c); v->fname[f][c] = 0; }
        p += used; len -= used;
        for (int f = 0; f < v->nfields; f++) {
            if (!(used = mat_tag(p, len, &type, &l, &d)) || type != 14) return 1;
            v->fdata[f] = d; v->flen[f] = l;
            p += used; len -= used;
        }
    } else return 4;
    return 0;
}

void sedumi_free(SedumiProb *P) {
    free(P->Ap); free(P->Ai); free(P->Ax); free(P->b); free(P->c); free(P->q); free(P->r); free(P->s);
    memset(P, 0, sizeof(*P));
}
/* the dense column vector of a variable (sparse or full, any shape), length *n */
static double *mat_vector(const MatVar *v, long *n) {
    const long tot = (long)v->dims[0] * (v->nd > 1 ? v->dims[1] : 1);
    double *x = (double *)calloc((size_t)(tot + 1), sizeof(double));
    if (v->sparse) { for (int j = 0; j < v->dims[1]; j++) for (int p = v->jc[j]; p < v->jc[j + 1]; p++) x[(long)j * v->dims[0] + v->ir[p]] = v->pr[p]; }
    else for (long i = 0; i < tot && i < v->npr; i++) x[i] = v->pr[i];
    *n = tot;
    return x;
}
static int mat_isfile(const unsigned char *h, size_t n) { return n >= 128 && !memcmp(h, "MATLAB", 6); }
int sedumi_is_mat(const char *fname) {
    FILE *f = fopen(fname, "rb");
    if (!f) return 0;
    unsigned char h[128];
    const size_t n = fread(h, 1, 128, f);
    fclose(f);
    return mat_isfile(h, n);
}
/* reads A (or At), b, c, K from a MAT-file. 0: ok; otherwise an error message in msg. */
int sedumi_read_mat(const char *fname, SedumiProb *P, char *msg, size_t nmsg) {
    memset(P, 0, sizeof(*P));
    FILE *f = fopen(fname, "rb");
    if (!f) { snprintf(msg, nmsg, "cannot open %s", fname); return 1; }
    fseek(f, 0, SEEK_END); const long sz = ftell(f); fseek(f, 0, SEEK_SET);
    unsigned char *buf = (unsigned char *)malloc((size_t)sz + 16);
    if (!buf || fread(buf, 1, (size_t)sz, f) != (size_t)sz) { fclose(f); free(buf); snprintf(msg, nmsg, "cannot read %s", fname); return 1; }
    fclose(f);
    if (!mat_isfile(buf, (size_t)sz)) { free(buf); snprintf(msg, nmsg, "%s is not a MAT-file", fname); return 1; }
    if (!memcmp(buf, "MATLAB 7.3", 10)) { free(buf); snprintf(msg, nmsg, "%s is a version 7.3 (HDF5) MAT-file: save it with  save -v7  (or -v6)", fname); return 1; }
    g_mat_swap = buf[126] == 'M' && buf[127] == 'I';                 /* written on a big-endian machine */
    if (!g_mat_swap && !(buf[126] == 'I' && buf[127] == 'M')) { free(buf); snprintf(msg, nmsg, "%s: not a level 5 MAT-file", fname); return 1; }
    MatVar vA, vb, vc, vK; int hA = 0, hb = 0, hc = 0, hK = 0, At = 0;
    memset(&vA, 0, sizeof vA); memset(&vb, 0, sizeof vb); memset(&vc, 0, sizeof vc); memset(&vK, 0, sizeof vK);
    unsigned char *keep[64]; int nkeep = 0;       /* inflated elements that the struct fields point into */
    size_t pos = 128; int bad = 0;
    while (pos + 8 <= (size_t)sz && !bad) {
        unsigned type; size_t l; const unsigned char *d;
        size_t used = mat_tag(buf + pos, (size_t)sz - pos, &type, &l, &d);
        if (!used) { bad = 1; break; }
        const unsigned char *body = d; size_t blen = l; unsigned char *infl = NULL;
        if (type == 15) {
            used = 8 + l;                                  /* compressed elements are not padded */
            size_t ol = 0;
            if (l < 6 || sd_inflate(d + 2, l - 2, &infl, &ol)) { bad = 1; break; }
            unsigned t2; size_t l2; const unsigned char *d2;
            if (!mat_tag(infl, ol, &t2, &l2, &d2) || t2 != 14) { free(infl); bad = 1; break; }
            body = d2; blen = l2; type = 14;
        }
        if (type == 14) {
            MatVar v;
            const int e = mat_matrix(body, blen, &v);
            int taken = 0;
            if (e == 0) {
                if ((!strcmp(v.name, "A") || !strcmp(v.name, "At")) && !hA) { vA = v; hA = 1; At = !strcmp(v.name, "At"); taken = 1; }
                else if (!strcmp(v.name, "b") && !hb) { vb = v; hb = 1; taken = 1; }
                else if ((!strcmp(v.name, "c") || !strcmp(v.name, "C")) && !hc) { vc = v; hc = 1; taken = 1; }
                else if (!strcmp(v.name, "K") && !hK && v.cls == 2) { vK = v; hK = 1; taken = 1; if (infl && nkeep < 64) { keep[nkeep++] = infl; infl = NULL; } }
            }
            if (!taken) matvar_free(&v);
        }
        free(infl);
        pos += used;
    }
    int rc = 1;
    if (bad) snprintf(msg, nmsg, "%s: the MAT-file could not be parsed", fname);
    else if (!hA || !hb || !hc || !hK) snprintf(msg, nmsg, "%s: the variables A (or At), b, c and K are needed (missing:%s%s%s%s)", fname, hA ? "" : " A", hb ? "" : " b", hc ? "" : " c", hK ? "" : " K");
    else {
        long nbv, ncv;
        P->b = mat_vector(&vb, &nbv); P->c = mat_vector(&vc, &ncv);
        P->m = (int)nbv; P->n = (int)ncv;
        /* K */
        for (int fld = 0; fld < vK.nfields; fld++) {
            MatVar v;
            if (vK.flen[fld] == 0 || mat_matrix(vK.fdata[fld], vK.flen[fld], &v)) continue;
            const char *nm = vK.fname[fld];
            if (v.pr && v.npr > 0 && !v.sparse) {
                if (!strcmp(nm, "f")) P->nf = (int)v.pr[0];
                else if (!strcmp(nm, "l")) P->nl = (int)v.pr[0];
                else if (!strcmp(nm, "q") || !strcmp(nm, "r") || !strcmp(nm, "s")) {
                    int *a = (int *)malloc(sizeof(int) * (size_t)(v.npr + 1)); int cnt = 0;
                    for (long i = 0; i < v.npr; i++) if (v.pr[i] > 0) a[cnt++] = (int)v.pr[i];
                    if (nm[0] == 'q') { P->q = a; P->nq = cnt; } else if (nm[0] == 'r') { P->r = a; P->nr = cnt; } else { P->s = a; P->ns = cnt; }
                }
            }
            matvar_free(&v);
        }
        /* A: by columns, m x n; At (or an n x m matrix) is transposed */
        const int ar = vA.dims[0], ac = vA.dims[1];
        int tr = At;
        if (!At && ar != P->m && ac == P->m && ar == P->n) tr = 1;
        if ((tr ? ac : ar) != P->m || (tr ? ar : ac) != P->n) snprintf(msg, nmsg, "%s: A is %d x %d, b has %d entries, c %d", fname, ar, ac, P->m, P->n);
        else {
            long nnz = 0;
            if (vA.sparse) nnz = vA.jc[ac]; else for (long i = 0; i < vA.npr; i++) nnz += vA.pr[i] != 0.0;
            P->Ap = (int *)calloc((size_t)P->n + 2, sizeof(int)); P->Ai = (int *)malloc(sizeof(int) * (size_t)(nnz + 1)); P->Ax = (double *)malloc(sizeof(double) * (size_t)(nnz + 1));
            /* triplets (row, col, v) of the m x n matrix, counted by column */
#define SD_EACH(body) do { if (vA.sparse) { for (int j_ = 0; j_ < ac; j_++) for (int p_ = vA.jc[j_]; p_ < vA.jc[j_ + 1]; p_++) { const int r_ = tr ? j_ : vA.ir[p_], c_ = tr ? vA.ir[p_] : j_; const double v_ = vA.pr[p_]; body } } \
                           else { for (int j_ = 0; j_ < ac; j_++) for (int i_ = 0; i_ < ar; i_++) { const double v_ = vA.pr[(long)j_ * ar + i_]; if (v_ == 0.0) continue; const int r_ = tr ? j_ : i_, c_ = tr ? i_ : j_; body } } } while (0)
            SD_EACH( (void)r_; (void)v_; P->Ap[c_ + 2]++; );
            for (int j = 0; j < P->n; j++) P->Ap[j + 2] += P->Ap[j + 1];
            SD_EACH( const int w_ = P->Ap[c_ + 1]++; P->Ai[w_] = r_; P->Ax[w_] = v_; );
#undef SD_EACH
            /* rows sorted within each column (a transposed sparse matrix comes out sorted; a plain one is sorted in the file) */
            rc = 0;
            long tot = (long)P->nf + P->nl;
            for (int k = 0; k < P->nq; k++) tot += P->q[k];
            for (int k = 0; k < P->nr; k++) tot += P->r[k];
            for (int k = 0; k < P->ns; k++) tot += (long)P->s[k] * P->s[k];
            if (tot != P->n) { snprintf(msg, nmsg, "%s: K describes %ld variables, c has %d", fname, tot, P->n); rc = 1; }
        }
    }
    matvar_free(&vA); matvar_free(&vb); matvar_free(&vc); matvar_free(&vK);
    for (int i = 0; i < nkeep; i++) free(keep[i]);
    free(buf);
    if (rc) sedumi_free(P);
    return rc;
}

/* ------------------------------------------------------------------ the solve */
typedef struct { int mat, blk, i, j; double v; } STrip;
static int cmp_strip(const void *a, const void *b) {
    const STrip *x = (const STrip *)a, *y = (const STrip *)b;
    if (x->mat != y->mat) return x->mat < y->mat ? -1 : 1;
    if (x->blk != y->blk) return x->blk < y->blk ? -1 : 1;
    if (x->i != y->i) return x->i < y->i ? -1 : 1;
    if (x->j != y->j) return x->j < y->j ? -1 : 1;
    return 0;
}
static const int g_exit_of_status[8] = { 0, 11, 12, 13, 14, 10, 15, 2 };

/* options of the cone solver from the command line's words; returns 0, 1 with a message, or 2:
 * the options ask for the semidefinite solver (-conesolver 0, high precision, bounds, first-order methods) */
static int socp_options(int argc, char **argv, SocpOpts *o, char *msg, size_t nmsg) {
    socp_default_opts(o);
    for (int i = 0; i < argc; i++) {
        const char *a = argv[i];
        if (!a) continue;
        if (!strcmp(a, "-q")) o->verbose = 0;
        else if (!strcmp(a, "-v")) o->verbose = 2;
        else if (!strcmp(a, "-tol") && i + 1 < argc) o->tol = atof(argv[++i]);
        else if (!strcmp(a, "-acc") && i + 1 < argc) { const char *l = argv[++i]; o->tol = !strcmp(l, "low") ? 1e-6 : !strcmp(l, "high") ? 1e-10 : 1e-8; }
        else if (!strcmp(a, "-maxit") && i + 1 < argc) o->maxit = atoi(argv[++i]);
        else if (!strcmp(a, "-timelimit") && i + 1 < argc) o->timelimit = atof(argv[++i]);
        else if (!strcmp(a, "-threads") && i + 1 < argc) {
#ifdef _OPENMP
            const int k = atoi(argv[i + 1]); if (k > 0) omp_set_num_threads(k);
#endif
            i++;
        }
        else if (!strcmp(a, "-x") || !strcmp(a, "-y") || !strcmp(a, "-z")) i++;
        else if (!strcmp(a, "-conesolver") && i + 1 < argc) { if (atoi(argv[++i]) == 0) return 2; }
        else if (!strcmp(a, "-prec") || !strcmp(a, "-bound") || !strcmp(a, "-certify") || !strcmp(a, "-fom") || !strcmp(a, "-mfipm") || !strcmp(a, "-lralm")) return 2;   /* features of the semidefinite solver */
        else if (a[0] == '-' && i + 1 < argc && argv[i + 1] && argv[i + 1][0] != '-') i++;      /* an option of the semidefinite solver with a value: no effect here */
    }
    if (!(o->tol > 0)) { snprintf(msg, nmsg, "invalid tolerance"); return 1; }
    return 0;
}

/* Solves P. argv: the option words (argc of them). The result's vectors are allocated here
 * (sedumi_result_free). run_sdp solves an SDPA problem given in memory with the same options. */
int sedumi_solve(const SedumiProb *P, int argc, char **argv, SedumiRes *R,
                 int (*run_sdp)(const BriskData *d, int argc, char **argv, BriskResult *res)) {
    memset(R, 0, sizeof(*R));
    R->status = -1;
    const int m = P->m, n = P->n;
    R->m = m; R->n = n;
    SocpOpts o; char msg[200];
    const int orc = P->ns == 0 ? socp_options(argc, argv, &o, msg, sizeof msg) : 2;
    if (orc == 1) { fprintf(stderr, "brisk: %s\n", msg); R->exit_code = 1; return 1; }
    if (orc == 0) {
        SocpProb Q = { m, n, P->Ap, P->Ai, P->Ax, P->b, P->c, P->nf, P->nl, P->nq, P->q, P->nr, P->r };
        SocpRes S;
#ifdef BRISK_LIBRARY
        socp_printf = brisk_printf;
#endif
        socp_stop = (volatile int *)&brisk_stop_flag;
        const int st = socp_solve(&Q, &o, &S);
        if (st == SOCP_INPUT) { fprintf(stderr, "brisk: the cone dimensions (K) do not match the number of columns of A\n"); R->exit_code = 2; return 2; }
        R->status = st; snprintf(R->status_str, sizeof R->status_str, "%s", socp_status_str(st));
        R->x = S.x; R->y = S.y; R->z = S.z; R->pobj = S.pobj; R->dobj = S.dobj; R->iters = S.iters; R->time = S.time;
        R->err[0] = S.err[0]; R->err[2] = S.err[1]; R->err[4] = S.err[2]; R->err[5] = S.err[3];
        R->exit_code = g_exit_of_status[st];
        return R->exit_code;
    }
    /* ---- semidefinite blocks: the SDPA form */
    int n1 = 0; for (int k = 0; k < P->ns; k++) n1 += P->s[k] == 1;
    const int nlp = 2 * P->nf + P->nl + n1, ncone = P->nq + P->nr;
    int nblk = (nlp > 0) + ncone + (P->ns - n1);
    int *bs = (int *)malloc(sizeof(int) * (size_t)(nblk + 1));
    int kb = 0;
    if (nlp > 0) bs[kb++] = -nlp;
    const int blk_cone0 = kb + 1;
    for (int k = 0; k < ncone; k++) bs[kb++] = k < P->nq ? P->q[k] : P->r[k - P->nq];
    const int blk_sdp0 = kb + 1;
    for (int k = 0; k < P->ns; k++) if (P->s[k] > 1) bs[kb++] = P->s[k];
    /* per column: block, kind, base; kind 0 free, 1 lp slot, 2 cone (t = col - start), 3 sdp */
    size_t cap = 0;
    for (int j = 0; j < n; j++) cap += (size_t)(P->Ap[j + 1] - P->Ap[j]) + (P->c[j] != 0.0);
    {   /* a cone's first column fans out to d diagonal entries; free columns to two; rotated first two columns to both */
        int col = P->nf + P->nl;
        size_t extra = 0;
        for (int j = 0; j < P->nf; j++) extra += (size_t)(P->Ap[j + 1] - P->Ap[j]) + 1;
        for (int k = 0; k < ncone; k++) {
            const int d = bs[blk_cone0 - 1 + k];
            const int rot = k >= P->nq;
            for (int t = 0; t < (rot ? 2 : 1); t++) extra += ((size_t)(P->Ap[col + t + 1] - P->Ap[col + t]) + 1) * (size_t)(d + 1);
            col += d;
        }
        cap += extra;
    }
    STrip *T = (STrip *)malloc(sizeof(STrip) * (cap + 1));
    size_t nt = 0;
    const double rs = sqrt(0.5);
#define SD_ADD(mat_, blk_, i_, j_, v_) do { T[nt].mat = (mat_); T[nt].blk = (blk_); T[nt].i = (i_); T[nt].j = (j_); T[nt].v = (v_); nt++; } while (0)
    {
        int col = 0;
#define SD_COL(j, EMIT) do { for (int p = P->Ap[j]; p < P->Ap[(j) + 1]; p++) { const int mat = P->Ai[p] + 1; const double v = P->Ax[p]; EMIT } \
                             if (P->c[j] != 0.0) { const int mat = 0; const double v = -P->c[j]; EMIT } } while (0)
        for (int j = 0; j < P->nf; j++, col++) SD_COL(col, SD_ADD(mat, 1, j + 1, j + 1, v); SD_ADD(mat, 1, P->nf + j + 1, P->nf + j + 1, -v););
        for (int j = 0; j < P->nl; j++, col++) SD_COL(col, SD_ADD(mat, 1, 2 * P->nf + j + 1, 2 * P->nf + j + 1, v););
        for (int k = 0; k < ncone; k++) {
            const int d = bs[blk_cone0 - 1 + k], blk = blk_cone0 + k, rot = k >= P->nq;
            for (int t = 0; t < d; t++, col++) {
                if (rot && t < 2) {
                    const double s2 = t == 0 ? rs : -rs;
                    SD_COL(col, for (int i = 1; i <= d; i++) SD_ADD(mat, blk, i, i, rs * v); SD_ADD(mat, blk, 1, 2, s2 * v););
                } else if (t == 0) SD_COL(col, for (int i = 1; i <= d; i++) SD_ADD(mat, blk, i, i, v););
                else SD_COL(col, SD_ADD(mat, blk, 1, t + 1, v););
            }
        }
        int slot = 2 * P->nf + P->nl, kb2 = blk_sdp0;
        for (int k = 0; k < P->ns; k++) {
            const int d = P->s[k];
            if (d == 1) { slot++; SD_COL(col, SD_ADD(mat, 1, slot, slot, v);); col++; continue; }
            for (int c2 = 0; c2 < d; c2++) for (int r2 = 0; r2 < d; r2++, col++) {
                const int lo = r2 < c2 ? r2 : c2, hi = r2 < c2 ? c2 : r2; const double w = r2 == c2 ? 1.0 : 0.5;
                SD_COL(col, SD_ADD(mat, kb2, lo + 1, hi + 1, w * v););
            }
            kb2++;
        }
    }
#undef SD_COL
#undef SD_ADD
    qsort(T, nt, sizeof(STrip), cmp_strip);
    size_t w = 0;
    for (size_t e = 0; e < nt; ) {
        size_t f = e; double v = 0;
        while (f < nt && cmp_strip(&T[e], &T[f]) == 0) v += T[f++].v;
        if (v != 0.0) { T[w] = T[e]; T[w].v = v; w++; }
        e = f;
    }
    nt = w;
    int *tm = (int *)malloc(sizeof(int) * (nt + 1)), *tb = (int *)malloc(sizeof(int) * (nt + 1)), *ti = (int *)malloc(sizeof(int) * (nt + 1)), *tj = (int *)malloc(sizeof(int) * (nt + 1));
    double *tv = (double *)malloc(sizeof(double) * (nt + 1));
    for (size_t e = 0; e < nt; e++) { tm[e] = T[e].mat; tb[e] = T[e].blk; ti[e] = T[e].i; tj[e] = T[e].j; tv[e] = T[e].v; }
    free(T);
    BriskData D = { m, nblk, bs, P->b, nt, tm, tb, ti, tj, tv };
    BriskResult B; memset(&B, 0, sizeof B);
    char **av = (char **)malloc(sizeof(char *) * ((size_t)argc + 1)); int ac = 0;      /* the options without -conesolver */
    for (int i = 0; i < argc; i++) { if (argv[i] && !strcmp(argv[i], "-conesolver") && i + 1 < argc) { i++; continue; } av[ac++] = argv[i]; }
    const int rc = run_sdp(&D, ac, av, &B);
    free(av);
    R->exit_code = rc;
    if (B.status >= 0 && B.y) {
        R->status = B.status; snprintf(R->status_str, sizeof R->status_str, "%s", B.status_str);
        R->iters = B.iters; R->time = B.time; R->have_x = B.have_x;
        for (int i = 0; i < 6; i++) R->err[i] = B.err[i + 1];
        R->x = (double *)calloc((size_t)n + 1, sizeof(double)); R->y = (double *)calloc((size_t)m + 1, sizeof(double)); R->z = (double *)calloc((size_t)n + 1, sizeof(double));
        memcpy(R->y, B.y, sizeof(double) * (size_t)m);
        if (B.have_x && B.X) {
            int col = 0;
            const double *xl = nlp > 0 ? B.X[0] : NULL;
            for (int j = 0; j < P->nf; j++, col++) R->x[col] = xl[j] - xl[P->nf + j];
            for (int j = 0; j < P->nl; j++, col++) R->x[col] = xl[2 * P->nf + j];
            for (int k = 0; k < ncone; k++) {
                const int d = bs[blk_cone0 - 1 + k]; const double *X = B.X[blk_cone0 - 1 + k];
                double tr = 0; for (int i = 0; i < d; i++) tr += X[(size_t)i * d + i];
                R->x[col] = tr;
                for (int t = 1; t < d; t++) R->x[col + t] = 2.0 * X[(size_t)t * d];
                if (k >= P->nq) { const double a = R->x[col], b2 = R->x[col + 1]; R->x[col] = rs * (a + b2); R->x[col + 1] = rs * (a - b2); }
                col += d;
            }
            int slot = 2 * P->nf + P->nl, kb2 = blk_sdp0 - 1;
            for (int k = 0; k < P->ns; k++) {
                const int d = P->s[k];
                if (d == 1) { R->x[col++] = xl[slot++]; continue; }
                memcpy(R->x + col, B.X[kb2], sizeof(double) * (size_t)d * d);
                col += d * d; kb2++;
            }
        }
        /* z = c - A'y */
        for (int j = 0; j < n; j++) { double s = 0; for (int p = P->Ap[j]; p < P->Ap[j + 1]; p++) s += P->Ax[p] * R->y[P->Ai[p]]; R->z[j] = P->c[j] - s; }
        R->pobj = 0; for (int j = 0; j < n; j++) R->pobj += P->c[j] * R->x[j];
        R->dobj = 0; for (int i = 0; i < m; i++) R->dobj += P->b[i] * R->y[i];
        if (!B.have_x) R->pobj = B.pobj;
    }
    brisk_result_free(&B);
    free(bs); free(tm); free(tb); free(ti); free(tj); free(tv);
    return rc;
}
void sedumi_result_free(SedumiRes *R) { free(R->x); free(R->y); free(R->z); memset(R, 0, sizeof(*R)); R->status = -1; }
