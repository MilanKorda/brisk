/* lpio.c - linear programs in general form: the MPS reader, a presolve, and the standard form
 *
 *     min / max  c'x + c0,   rl <= A x <= ru,   l <= x <= u
 *
 * for the SOCP solver (socp.c through sedumi.c), with the map back to the variables of the file.
 *
 * MPS: fixed and free format (fields split at white space: names without blanks), sections
 * NAME, OBJSENSE (MAX / MAXIMIZE), OBJSENSE inline, ROWS, COLUMNS (MARKER lines skipped: integer
 * variables are read as continuous), RHS (on the objective row: minus the constant), RANGES, BOUNDS
 * (UP LO FX FR MI PL BV LI UI), ENDATA. The first N row is the objective, the others are dropped.
 *
 * Presolve (lp_to_sedumi, level >= 1), repeated until nothing changes:
 *   - empty rows (checked) and rows of one entry (a bound on its variable);
 *   - fixed variables (l = u) and empty columns (at the bound their cost prefers);
 *   - rows that the bounds of their variables make redundant on a side or on both, forcing rows
 *     (the variables are fixed at the bounds that attain the side);
 *   - dual fixing: a column that can only help feasibility and the objective in one direction is
 *     fixed at that bound;
 *   - a free column with one entry in an equality row: the row defines it (row and column go, the
 *     costs of the row's other variables are updated); in an inequality row with zero cost the row
 *     is redundant;
 *   - dominated columns (level 2, when nothing else changes): bounds on the row duals from the columns'
 *     dual constraints, and a column whose reduced cost has one sign over those bounds fixed at its bound.
 * Rows are removed, never changed, so the map back needs the fixed values and, for the free
 * singletons, the row they were defined by (in reverse order).
 *
 * Standard form: x = l + x' (x' >= 0), x = u - x' when only u is finite, free variables as they
 * are, a bound row x' + s = u - l for a variable with two finite bounds; rows: an equality as it
 * is, one slack for an inequality, a slack and a bound row for a range.                       */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <ctype.h>
#include "lpio.h"
#include "fastnum.h"
#ifndef _WIN32
#include <spawn.h>
#include <sys/wait.h>
#include <unistd.h>
extern char **environ;
#endif

static void *xm(size_t n) { void *p = malloc(n ? n : 1); if (!p) { fprintf(stderr, "brisk: out of memory (LP reader)\n"); exit(1); } return p; }
static void *xz(size_t n) { void *p = calloc(1, n ? n : 1); if (!p) { fprintf(stderr, "brisk: out of memory (LP reader)\n"); exit(1); } return p; }

/* ---- names ---- */
/* (25 October 2026) each slot keeps its key's hash beside the key and the value: a probe reads one slot and compares the
 * hashes before the strings (neos-3402454-bohle, LP: 14.7 million lookups among 2.9 million row names took 2.5 s of the
 * reading with keys, values and hashes in three arrays, one cache miss each) */
/* (25 October 2026, later) a key of at most 8 bytes is kept in the slot itself, its bytes packed into w (zero padded),
 * so that a probe reads no string; a longer key keeps its pointer in w and the top bit of h marks it. Slots of 16 bytes
 * (were 24, with every comparison through the key's pointer: a second cache miss per lookup). MIPLIB's names are mostly
 * of at most 8 characters (neos-3402294-bobin, LP: 591,076 rows of 5 to 7) */
typedef struct { unsigned long long w; unsigned h; int val; } HSlot;
typedef struct { HSlot *t; int cap, n; } Hash;
#define H_LONG 0x80000000u
static unsigned long hstr(const char *s) { unsigned long h = 1469598103934665603UL; while (*s) { h ^= (unsigned char)*s++; h *= 1099511628211UL; } return h; }
/* the slot's form of key k: 1 and the packed bytes when it has at most 8, else 0 */
static int h_pack(const char *k, unsigned long long *w) {
    unsigned long long v = 0; int i = 0;
    for (; i < 8 && k[i]; i++) v |= (unsigned long long)(unsigned char)k[i] << (8 * i);
    if (k[i]) return 0;
    *w = v; return 1;
}
static void h_init(Hash *H, int cap) { H->cap = 16; while (H->cap < 2 * cap) H->cap *= 2; H->t = xz(sizeof(HSlot) * H->cap); H->n = 0; }
static void h_grow(Hash *H);
static void h_put_slot(Hash *H, unsigned long long w, unsigned h, int v, unsigned long hi) {
    if (2 * (H->n + 1) > H->cap) h_grow(H);
    unsigned long i = hi & (H->cap - 1);
    while (H->t[i].w) i = (i + 1) & (H->cap - 1);
    H->t[i].w = w; H->t[i].h = h; H->t[i].val = v; H->n++;
}
/* the slot's hash: the key's hash's low 31 bits (its index in the table is taken from the full hash, kept by the
 * caller of h_put_slot; on growth the slot's 31 bits index the new table, enough for 2^31 slots) */
static void h_put(Hash *H, char *k, int v) {
    const unsigned long hv = hstr(k); unsigned long long w;
    const unsigned h = (unsigned)(hv & 0x7fffffffu);
    if (h_pack(k, &w)) h_put_slot(H, w, h, v, h);
    else h_put_slot(H, (unsigned long long)(size_t)k, h | H_LONG, v, h);
}
static void h_grow(Hash *H) {
    Hash G; G.cap = 2 * H->cap; G.t = xz(sizeof(HSlot) * G.cap); G.n = 0;
    for (int i = 0; i < H->cap; i++) if (H->t[i].w) h_put_slot(&G, H->t[i].w, H->t[i].h, H->t[i].val, H->t[i].h & ~H_LONG);
    free(H->t); *H = G;
}
static int h_get(const Hash *H, const char *k) {
    const unsigned h = (unsigned)(hstr(k) & 0x7fffffffu); unsigned long long w;
    unsigned long i = h & (H->cap - 1);
    if (h_pack(k, &w)) {
        while (H->t[i].w) { if (H->t[i].h == h && H->t[i].w == w) return H->t[i].val; i = (i + 1) & (H->cap - 1); }
    } else {
        while (H->t[i].w) { if (H->t[i].h == (h | H_LONG) && !strcmp((const char *)(size_t)H->t[i].w, k)) return H->t[i].val; i = (i + 1) & (H->cap - 1); }
    }
    return -1;
}
static void h_free(Hash *H) { free(H->t); }

/* (25 October 2026) a number of the MPS file: decimal digits, a point and an exponent read directly when the digits fit in
 * 53 bits and the power of ten is at most 22 (Clinger's fast path: the product or quotient of two exact doubles, rounded
 * once, is the correctly rounded value strtod returns); anything else by atof, as before. The same bits as atof on every
 * file of the LP sets (BRISK_READSUM) */
static double mps_num(const char *s) {
    static const double p10[23] = { 1e0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9, 1e10, 1e11, 1e12, 1e13, 1e14, 1e15, 1e16, 1e17, 1e18, 1e19, 1e20, 1e21, 1e22 };
    const char *p = s; int neg = 0;
    if (*p == '-') { neg = 1; p++; } else if (*p == '+') p++;
    unsigned long long mant = 0; int nd = 0, e10 = 0, any = 0;
    while (*p >= '0' && *p <= '9') { any = 1; if (mant || *p != '0') { if (nd >= 19) return atof(s); mant = mant * 10 + (unsigned)(*p - '0'); nd++; } p++; }
    if (*p == '.') {
        p++;
        while (*p >= '0' && *p <= '9') { any = 1; if (mant || *p != '0') { if (nd >= 19) return atof(s); mant = mant * 10 + (unsigned)(*p - '0'); nd++; } e10--; p++; }
    }
    if (!any) return atof(s);
    if (*p == 'e' || *p == 'E') {
        p++; int eneg = 0, ex = 0, ed = 0;
        if (*p == '-') { eneg = 1; p++; } else if (*p == '+') p++;
        while (*p >= '0' && *p <= '9') { if (ex < 10000) ex = ex * 10 + (*p - '0'); ed++; p++; }
        if (!ed) return atof(s);
        e10 += eneg ? -ex : ex;
    }
    if (*p) return atof(s);
    if (mant > (1ULL << 53) || e10 < -22 || e10 > 22) return atof(s);
    double v = (double)mant;
    v = e10 >= 0 ? v * p10[e10] : v / p10[-e10];
    return neg ? -v : v;
}

/* (28 October 2026, the LP developer) a file opened for reading, through gzip -dc when its name ends in .gz: gzip is
 * started without a shell (the name an argument of its own, not part of a command line, which a name with a quote broke
 * and could turn into a command); *pid its process (0 for a plain file). lp_close_in: nonzero when gzip failed (a damaged
 * or truncated archive) */
static FILE *lp_open_in(const char *fname, long *pid) {
    *pid = 0;
    const size_t fl = strlen(fname);
    if (!(fl > 3 && !strcmp(fname + fl - 3, ".gz"))) return fopen(fname, "r");
#ifdef _WIN32
    return NULL;
#else
    { FILE *t = fopen(fname, "r"); if (!t) return NULL; fclose(t); }
    int fd[2]; if (pipe(fd)) return NULL;
    posix_spawn_file_actions_t fa; posix_spawn_file_actions_init(&fa);
    posix_spawn_file_actions_adddup2(&fa, fd[1], 1); posix_spawn_file_actions_addclose(&fa, fd[0]); posix_spawn_file_actions_addclose(&fa, fd[1]);
    char *argv[] = { (char *)"gzip", (char *)"-dc", (char *)"--", (char *)fname, NULL };
    pid_t p;
    const int e = posix_spawnp(&p, "gzip", &fa, NULL, argv, environ);
    posix_spawn_file_actions_destroy(&fa);
    close(fd[1]);
    if (e) { close(fd[0]); return NULL; }
    FILE *f = fdopen(fd[0], "r");
    if (!f) { close(fd[0]); waitpid(p, NULL, 0); return NULL; }
    *pid = (long)p;
    return f;
#endif
}
static int lp_close_in(FILE *f, long pid) {
    fclose(f);
    if (!pid) return 0;
#ifdef _WIN32
    return 1;
#else
    int st = 0;
    if (waitpid((pid_t)pid, &st, 0) < 0) return 1;
    return !(WIFEXITED(st) && WEXITSTATUS(st) == 0);
#endif
}
/* (28 October 2026) also a name ending in .mps.gz or .MPS.gz (read through gzip: MIPLIB's files) */
int lp_is_mps(const char *fname) {
    const size_t l = strlen(fname);
    if (l > 4 && (!strcmp(fname + l - 4, ".mps") || !strcmp(fname + l - 4, ".MPS") || !strcmp(fname + l - 4, ".sif") || !strcmp(fname + l - 4, ".SIF"))) return 1;
    if (l > 7 && (!strcmp(fname + l - 7, ".mps.gz") || !strcmp(fname + l - 7, ".MPS.gz"))) return 1;
    return 0;
}

void lp_free(LpProb *L) {
    free(L->Ap); free(L->Ai); free(L->Ax); free(L->c); free(L->rl); free(L->ru); free(L->l); free(L->u);
    if (L->rname) { for (int i = 0; i < L->m; i++) free(L->rname[i]); free(L->rname); }
    if (L->cname) { for (int j = 0; j < L->n; j++) free(L->cname[j]); free(L->cname); }
    memset(L, 0, sizeof(*L));
}

/* (28 October 2026, the night) a column of the problem as read whose cost improves without limit in a direction that no row
 * and no bound stops: x_j can grow (or fall) alone, every row's activity moving away from its finite sides, so the
 * direction e_j is a ray and the problem is infeasible or unbounded (dual infeasible), exactly: the test reads only signs
 * and infinite bounds, so the rounding of the file's digits cannot make it wrong. A cost within 1e-9 (1 + the largest |cost|)
 * of zero counts as zero (within the dual tolerance it makes no ray). Returns the column, or -1. The presolves found such
 * columns after their reductions and the problem as read was then solved again to confirm the verdict: 10 to 64 s on four
 * held-out LPs (neos-3283608-agout, neos-3230376-yser, neos-3699044-maipo, neos-3709489-menik), Gurobi 0.01 to 0.26 s. */
/* (28 October 2026, the night) the caller's stop flag for the standard presolve's long loops, per thread (the default LP
 * route's interior-point side, when the dual simplex method beside it has its result): the eliminator stops between two
 * substitutions, and the problem goes on with those made (neos-4984115-canche, LP: its eliminator took 45 s after the
 * dual simplex method had ended at 14 s) */
static _Thread_local const volatile int *g_lp_stopf = NULL;
void lp_set_stop(const volatile int *f) { g_lp_stopf = f; }
const volatile int *lp_get_stop(void) { return g_lp_stopf; }
int lp_ray_column(const LpProb *L) {
    const double inf = 1e29;
    double cmax = 0; for (int j = 0; j < L->n; j++) cmax = fmax(cmax, fabs(L->c[j]));
    const double tol = 1e-9 * (1 + cmax);
    for (int j = 0; j < L->n; j++) {
        const double c = L->maxim ? -L->c[j] : L->c[j];
        if (fabs(c) <= tol) continue;
        const int up = c < 0;           /* the improving direction */
        if (up ? L->u[j] < inf : L->l[j] > -inf) continue;
        int ok = 1;
        for (int p = L->Ap[j]; p < L->Ap[j + 1] && ok; p++) {
            const double a = L->Ax[p]; const int i = L->Ai[p];
            if (a == 0) continue;
            if ((a > 0) == up) { if (L->ru[i] < inf) ok = 0; }     /* the row's activity grows */
            else if (L->rl[i] > -inf) ok = 0;                     /* it falls */
        }
        if (ok) return j;
    }
    return -1;
}

static int split(char *line, char **tok, int maxt) {
    int n = 0; char *p = line;
    while (*p && n < maxt) {
        while (*p == ' ' || *p == '\t' || *p == '\r' || *p == '\n') p++;
        if (!*p) break;
        tok[n++] = p;
        while (*p && *p != ' ' && *p != '\t' && *p != '\r' && *p != '\n') p++;
        if (*p) *p++ = 0;
    }
    return n;
}

/* fixed format (names with blanks): the fields by their columns 2-3, 5-12, 15-22, 25-36, 40-47, 50-61,
 * returned as the tokens the free-format logic expects for the section (sec: 1 rows, 2 columns,
 * 3 rhs / ranges, 4 bounds); a blank set name is replaced by a placeholder */
static char *field(char *line, size_t len, int a, int b, char *save) {
    (void)save;
    if ((size_t)a > len) return NULL;
    size_t e = (size_t)b < len ? (size_t)b : len;
    char *p = line + a - 1, *q = line + e;          /* [a-1, e) */
    while (q > p && (q[-1] == ' ' || q[-1] == '\t' || q[-1] == '\r' || q[-1] == '\n')) q--;
    while (p < q && (*p == ' ' || *p == '\t')) p++;
    if (p == q) return NULL;
    return p;
}
static int split_fixed(char *line, char **tok, int sec) {
    static _Thread_local char buf[6][64]; static _Thread_local char dflt[] = "_SET_";   /* (per thread: files read on several threads at once) */
    const size_t len = strlen(line);
    static const int fa[6] = { 2, 5, 15, 25, 40, 50 }, fb[6] = { 3, 12, 22, 36, 47, 61 };
    char *f[6];
    for (int k = 0; k < 6; k++) {
        char *p = field(line, len, fa[k], fb[k], NULL);
        if (!p) { f[k] = NULL; continue; }
        size_t e = (size_t)fb[k] < len ? (size_t)fb[k] : len, l = (size_t)(line + e - p);
        while (l > 0 && (p[l - 1] == ' ' || p[l - 1] == '\t' || p[l - 1] == '\r' || p[l - 1] == '\n')) l--;
        if (l > 63) l = 63;
        memcpy(buf[k], p, l); buf[k][l] = 0; f[k] = buf[k];
    }
    int n = 0;
    if (sec == 1) { if (f[0]) tok[n++] = f[0]; if (f[1]) tok[n++] = f[1]; return n; }
    if (sec == 2) { for (int k = 1; k < 6; k++) if (f[k]) tok[n++] = f[k]; return n; }
    if (sec == 3) { tok[n++] = f[1] ? f[1] : dflt; for (int k = 2; k < 6; k++) if (f[k]) tok[n++] = f[k]; return n; }
    if (f[0]) tok[n++] = f[0];
    tok[n++] = f[1] ? f[1] : dflt;
    for (int k = 2; k < 4; k++) if (f[k]) tok[n++] = f[k];
    return n;
}
static int lp_read_mps_(const char *fname, LpProb *L, char *msg, size_t nmsg, int fixed, signed char **isint);
/* (25 October 2026) BRISK_READSUM=1: a checksum of the problem as read (every number's bits and every name), printed
 * to compare two readers on the same files */
static void lp_readsum(const LpProb *L) {
    unsigned long long h = 1469598103934665603ULL;
#define RS_MIX(p, nb) do { const unsigned char *q_ = (const unsigned char *)(p); for (size_t k_ = 0; k_ < (size_t)(nb); k_++) { h ^= q_[k_]; h *= 1099511628211ULL; } } while (0)
    RS_MIX(&L->m, sizeof L->m); RS_MIX(&L->n, sizeof L->n); RS_MIX(&L->maxim, sizeof L->maxim); RS_MIX(&L->c0, sizeof L->c0);
    RS_MIX(L->Ap, sizeof(int) * (size_t)(L->n + 1)); RS_MIX(L->Ai, sizeof(int) * (size_t)L->Ap[L->n]); RS_MIX(L->Ax, sizeof(double) * (size_t)L->Ap[L->n]);
    RS_MIX(L->c, sizeof(double) * (size_t)L->n); RS_MIX(L->l, sizeof(double) * (size_t)L->n); RS_MIX(L->u, sizeof(double) * (size_t)L->n);
    RS_MIX(L->rl, sizeof(double) * (size_t)L->m); RS_MIX(L->ru, sizeof(double) * (size_t)L->m);
    for (int i = 0; i < L->m; i++) RS_MIX(L->rname[i], strlen(L->rname[i]) + 1);
    for (int j = 0; j < L->n; j++) RS_MIX(L->cname[j], strlen(L->cname[j]) + 1);
#undef RS_MIX
    printf("readsum %016llx m %d n %d nz %d\n", h, L->m, L->n, L->Ap[L->n]);
}
int lp_read_mps(const char *fname, LpProb *L, char *msg, size_t nmsg) {
    int rc = lp_read_mps_(fname, L, msg, nmsg, 0, NULL);
    if (rc == 3) rc = lp_read_mps_(fname, L, msg, nmsg, 1, NULL);      /* names with blanks: the fixed format */
    if (rc == 0 && getenv("BRISK_READSUM")) lp_readsum(L);
    return rc;
}
/* (22 October 2026, the MIP developer) the mixed-integer program of an MPS file (lpio.h): the same reading, and the
 * integer columns in the same pass (the markers and the bounds BV, LI, UI); an integer column of a marker section with
 * no bound in the BOUNDS section is binary, [0, 1], as MIPLIB 2017's solution checker and Gurobi read it */
int lp_read_mps_mip(const char *fname, LpProb *L, signed char **isint, char *msg, size_t nmsg) {
    *isint = NULL;
    const int rc = lp_read_mps_(fname, L, msg, nmsg, 0, isint);
    if (rc == 3) return lp_read_mps_(fname, L, msg, nmsg, 1, isint);
    return rc;
}
static int lp_read_mps_(const char *fname, LpProb *L, char *msg, size_t nmsg, int fixed, signed char **isint) {
    memset(L, 0, sizeof(*L));
    long gzpid; FILE *f = lp_open_in(fname, &gzpid);
    if (!f) { snprintf(msg, nmsg, "cannot open %s", fname); return 1; }
    setvbuf(f, NULL, _IOFBF, 1 << 20);
    enum { S_NONE, S_ROWS, S_COLS, S_RHS, S_RANGES, S_BOUNDS, S_OBJSENSE, S_SKIP } sec = S_NONE;
    int mcap = 1024, m = 0, objrow = -1;          /* rows in file order, including N rows (type 'N') */
    char *rtype = xm(mcap); char **rn = xm(sizeof(char *) * mcap);
    Hash HR, HC; h_init(&HR, 1024); h_init(&HC, 1024);
    int ncap = 1024, n = 0; char **cn = xm(sizeof(char *) * ncap); long *cp = xm(sizeof(long) * (ncap + 1));
    signed char *ci = isint ? xz((size_t)ncap) : NULL; int inint = 0;     /* the integer columns (isint != NULL) */
    long zcap = 1 << 16, nz = 0; int *zi = xm(sizeof(int) * zcap); double *zv = xm(sizeof(double) * zcap);
    double *rhs = NULL, *rng = NULL, *lo = NULL, *up = NULL; char *hasrng = NULL, *bset = NULL;
    char cur[256] = "";
    char *line = NULL; size_t lcap = 0; long lineno = 0;
    char *tok[8];
    int rc = 0, ended = 0;
    while (getline(&line, &lcap, f) >= 0) {
        lineno++;
        if (line[0] == '*' || line[0] == '\n' || line[0] == '\r') continue;
        if (line[0] != ' ' && line[0] != '\t') {
            /* a section header */
            const int nt = split(line, tok, 4);
            if (nt == 0) continue;
            if (sec == S_COLS && strcmp(tok[0], "COLUMNS")) {
                /* the columns are complete: allocate the per-row and per-column data */
                rhs = xz(sizeof(double) * (m + 1)); rng = xz(sizeof(double) * (m + 1)); hasrng = xz(m + 1);
                lo = xz(sizeof(double) * (n + 1)); up = xm(sizeof(double) * (n + 1)); bset = xz(n + 1);
                for (int j = 0; j < n; j++) up[j] = LP_INF;
            }
            if (!strcmp(tok[0], "NAME")) { if (nt > 1) snprintf(L->name, sizeof L->name, "%s", tok[1]); sec = S_NONE; }
            else if (!strcmp(tok[0], "ROWS")) sec = S_ROWS;
            else if (!strcmp(tok[0], "COLUMNS")) sec = S_COLS;
            else if (!strcmp(tok[0], "RHS")) sec = S_RHS;
            else if (!strcmp(tok[0], "RANGES")) sec = S_RANGES;
            else if (!strcmp(tok[0], "BOUNDS")) sec = S_BOUNDS;
            else if (!strcmp(tok[0], "OBJSENSE")) { sec = S_OBJSENSE; if (nt > 1 && (tok[1][0] == 'M' || tok[1][0] == 'm') && (tok[1][1] == 'A' || tok[1][1] == 'a')) L->maxim = 1; }
            else if (!strcmp(tok[0], "ENDATA")) { ended = 1; break; }
            else sec = S_SKIP;            /* OBJSENSE variants, QSECTION etc.: skipped */
            if (!strcmp(tok[0], "QSECTION") || !strcmp(tok[0], "QUADOBJ") || !strcmp(tok[0], "QMATRIX") || !strcmp(tok[0], "QCMATRIX")) {
                snprintf(msg, nmsg, "%s: a quadratic section (%s): not a linear program", fname, tok[0]); rc = 1; break;
            }
            /* (25 October 2026, the MIP developer) constraints that are not rows: skipping them would solve another problem
             * (MIPLIB 2017's problems with indicator constraints were read as if every indicator row always held, and
             * reported infeasible) */
            if (!strcmp(tok[0], "INDICATORS") || !strcmp(tok[0], "SOS") || !strcmp(tok[0], "GENCONS") || !strcmp(tok[0], "PWLOBJ")) {
                snprintf(msg, nmsg, "%s: a section of constraints that are not linear rows (%s: %s), which BRISK does not support",
                         fname, tok[0], !strcmp(tok[0], "INDICATORS") ? "indicator constraints" : !strcmp(tok[0], "SOS") ? "special ordered sets" :
                         !strcmp(tok[0], "GENCONS") ? "general constraints" : "a piecewise-linear objective");
                rc = 1; break;
            }
            continue;
        }
        int nt = (fixed && (sec == S_ROWS || sec == S_COLS || sec == S_RHS || sec == S_RANGES || sec == S_BOUNDS))
                 ? split_fixed(line, tok, sec == S_ROWS ? 1 : sec == S_COLS ? 2 : sec == S_BOUNDS ? 4 : 3) : split(line, tok, 7);
        /* (28 October 2026) a name beginning with '$' after the line's first field begins a comment, the rest of the line
         * (the MPS convention CPLEX's and Gurobi's readers follow): neos-5044663-wairoa (LP, MIPLIB 2017 collection) has
         * "C0063540  LB_full  0  $ empty column", and the whole file was refused */
        if (sec == S_COLS || sec == S_RHS || sec == S_RANGES || sec == S_BOUNDS)
            for (int t = 1; t < nt; t++) if (tok[t][0] == '$') { nt = t; break; }
        if (nt == 0) continue;
        if (!fixed && sec == S_ROWS && nt > 2) { rc = 3; break; }
        if (sec == S_OBJSENSE) { if ((tok[0][0] == 'M' || tok[0][0] == 'm') && (tok[0][1] == 'A' || tok[0][1] == 'a')) L->maxim = 1; continue; }
        if (sec == S_ROWS) {
            if (nt < 2) { snprintf(msg, nmsg, "%s, line %ld: a row needs a type and a name", fname, lineno); rc = 1; break; }
            if (m == mcap) { mcap *= 2; rtype = realloc(rtype, mcap); rn = realloc(rn, sizeof(char *) * mcap); }
            rtype[m] = (char)toupper((unsigned char)tok[0][0]);
            rn[m] = strdup(tok[1]);
            if (h_get(&HR, rn[m]) >= 0) { snprintf(msg, nmsg, "%s, line %ld: row %s twice", fname, lineno, tok[1]); rc = 1; m++; break; }
            h_put(&HR, rn[m], m);
            if (rtype[m] == 'N' && objrow < 0) objrow = m;
            m++;
        } else if (sec == S_COLS) {
            if (nt >= 3 && !strcmp(tok[1], "'MARKER'")) {
                if (strstr(tok[2], "INTORG")) inint = 1; else if (strstr(tok[2], "INTEND")) inint = 0;
                continue;
            }
            if (nt < 3 || (nt % 2) == 0) { snprintf(msg, nmsg, "%s, line %ld: a column line is: column row value [row value]", fname, lineno); rc = 1; break; }
            if (strcmp(cur, tok[0])) {
                if (h_get(&HC, tok[0]) >= 0) { snprintf(msg, nmsg, "%s, line %ld: the entries of column %s are not contiguous", fname, lineno, tok[0]); rc = 1; break; }
                if (n == ncap) {
                    ncap *= 2; cn = realloc(cn, sizeof(char *) * ncap); cp = realloc(cp, sizeof(long) * (ncap + 1));
                    if (ci) { ci = realloc(ci, (size_t)ncap); memset(ci + n, 0, (size_t)(ncap - n)); }
                }
                if (ci) ci[n] = (signed char)inint;
                cn[n] = strdup(tok[0]); h_put(&HC, cn[n], n); cp[n] = nz; n++;
                snprintf(cur, sizeof cur, "%s", tok[0]);
            }
            for (int t = 1; t + 1 < nt; t += 2) {
                const int r = h_get(&HR, tok[t]);
                if (r < 0) { snprintf(msg, nmsg, "%s, line %ld: unknown row %s", fname, lineno, tok[t]); rc = 1; break; }
                const double v = mps_num(tok[t + 1]);
                if (v == 0.0) continue;
                if (nz == zcap) { zcap *= 2; zi = realloc(zi, sizeof(int) * zcap); zv = realloc(zv, sizeof(double) * zcap); }
                zi[nz] = r; zv[nz] = v; nz++;
            }
            if (rc) break;
        } else if (sec == S_RHS || sec == S_RANGES) {
            if (!rhs) { snprintf(msg, nmsg, "%s, line %ld: RHS before COLUMNS", fname, lineno); rc = 1; break; }
            for (int t = (nt % 2); t + 1 < nt; t += 2) {
                const int r = h_get(&HR, tok[t]);
                if (r < 0) { snprintf(msg, nmsg, "%s, line %ld: unknown row %s", fname, lineno, tok[t]); rc = 1; break; }
                if (sec == S_RHS) rhs[r] = mps_num(tok[t + 1]); else { rng[r] = mps_num(tok[t + 1]); hasrng[r] = 1; }
            }
            if (rc) break;
        } else if (sec == S_BOUNDS) {
            if (!lo) { snprintf(msg, nmsg, "%s, line %ld: BOUNDS before COLUMNS", fname, lineno); rc = 1; break; }
            char ty[3] = { (char)toupper((unsigned char)tok[0][0]), (char)(tok[0][0] ? toupper((unsigned char)tok[0][1]) : 0), 0 };
            const int noval = !strcmp(ty, "FR") || !strcmp(ty, "MI") || !strcmp(ty, "PL") || !strcmp(ty, "BV");
            int ic;                                   /* the token of the column: after an optional set name */
            if (noval) ic = nt >= 3 ? 2 : 1; else ic = nt >= 4 ? 2 : 1;
            if (ic >= nt || (!noval && ic + 1 >= nt)) { snprintf(msg, nmsg, "%s, line %ld: incomplete bound", fname, lineno); rc = 1; break; }
            const int j = h_get(&HC, tok[ic]);
            if (j < 0) { snprintf(msg, nmsg, "%s, line %ld: unknown column %s", fname, lineno, tok[ic]); rc = 1; break; }
            const double v = noval ? 0.0 : mps_num(tok[ic + 1]);
            if (ci && (!strcmp(ty, "BV") || !strcmp(ty, "LI") || !strcmp(ty, "UI"))) ci[j] = 1;
            if (!strcmp(ty, "UP") || !strcmp(ty, "UI")) { up[j] = v; if (v < 0 && !(bset[j] & 1) && lo[j] == 0.0) lo[j] = -LP_INF; bset[j] |= 2; }
            else if (!strcmp(ty, "LO") || !strcmp(ty, "LI")) { lo[j] = v; bset[j] |= 1; }
            else if (!strcmp(ty, "FX")) { lo[j] = up[j] = v; bset[j] |= 3; }
            else if (!strcmp(ty, "FR")) { lo[j] = -LP_INF; up[j] = LP_INF; bset[j] |= 3; }
            else if (!strcmp(ty, "MI")) { lo[j] = -LP_INF; bset[j] |= 1; }
            else if (!strcmp(ty, "PL")) { up[j] = LP_INF; bset[j] |= 2; }
            else if (!strcmp(ty, "BV")) { lo[j] = 0; up[j] = 1; bset[j] |= 3; }
            else { snprintf(msg, nmsg, "%s, line %ld: unknown bound type %s", fname, lineno, tok[0]); rc = 1; break; }
        }
    }
    free(line);
    if (lp_close_in(f, gzpid) && !rc) { snprintf(msg, nmsg, "%s: the decompression failed (gzip -dc: the file is damaged or cut short)", fname); rc = 1; }
    /* (24 October 2026) a file that ends before ENDATA is an error: cut at a line boundary it would be read as a smaller
     * problem (the QPS and CBF readers likewise) */
    if (!rc && !ended) { snprintf(msg, nmsg, "%s: the file ends before ENDATA (cut short?)", fname); rc = 1; }
    if (!rc && (m == 0 || n == 0)) { snprintf(msg, nmsg, "%s: no rows or no columns (not an MPS file?)", fname); rc = 1; }
    if (!rc && !rhs) {       /* no section after COLUMNS */
        rhs = xz(sizeof(double) * (m + 1)); rng = xz(sizeof(double) * (m + 1)); hasrng = xz(m + 1);
        lo = xz(sizeof(double) * (n + 1)); up = xm(sizeof(double) * (n + 1)); bset = xz(n + 1);
        for (int j = 0; j < n; j++) up[j] = LP_INF;
    }
    if (!rc) {
        cp[n] = nz;
        /* constraint rows: the N rows are dropped (the first is the objective) */
        int *rmap = xm(sizeof(int) * (m + 1)), mc = 0;
        for (int i = 0; i < m; i++) rmap[i] = rtype[i] == 'N' ? -1 : mc++;
        L->m = mc; L->n = n;
        if (ci) for (int j = 0; j < n; j++) if (ci[j] && !bset[j]) up[j] = 1;     /* an integer column without bounds: binary */
        L->c = xz(sizeof(double) * (n + 1)); L->l = lo; L->u = up; lo = up = NULL;
        L->rl = xm(sizeof(double) * (mc + 1)); L->ru = xm(sizeof(double) * (mc + 1));
        L->rname = xm(sizeof(char *) * (mc + 1));
        for (int i = 0; i < m; i++) {
            const int r = rmap[i];
            if (r < 0) { if (i == objrow) L->c0 = -rhs[i]; free(rn[i]); continue; }
            L->rname[r] = rn[i];
            double a = -LP_INF, b = LP_INF;
            if (rtype[i] == 'E') { a = b = rhs[i]; if (hasrng[i]) { if (rng[i] >= 0) b = a + rng[i]; else a = b + rng[i]; } }
            else if (rtype[i] == 'L') { b = rhs[i]; if (hasrng[i]) a = b - fabs(rng[i]); }
            else if (rtype[i] == 'G') { a = rhs[i]; if (hasrng[i]) b = a + fabs(rng[i]); }
            else { snprintf(msg, nmsg, "%s: unknown row type %c", fname, rtype[i]); rc = 1; }
            L->rl[r] = a; L->ru[r] = b;
        }
        L->cname = cn; cn = NULL;
        L->Ap = xm(sizeof(int) * (n + 1)); L->Ai = xm(sizeof(int) * (nz + 1)); L->Ax = xm(sizeof(double) * (nz + 1));
        long w = 0;
        for (int j = 0; j < n; j++) {
            L->Ap[j] = (int)w;
            for (long p = cp[j]; p < cp[j + 1]; p++) {
                if (zi[p] == objrow) { L->c[j] += zv[p]; continue; }
                const int r = rmap[zi[p]];
                if (r < 0) continue;
                L->Ai[w] = r; L->Ax[w] = zv[p]; w++;
            }
        }
        L->Ap[n] = (int)w;
        free(rmap);
    } else {
        for (int i = 0; i < m; i++) free(rn[i]);
        if (cn) for (int j = 0; j < n; j++) free(cn[j]);
    }
    free(rtype); free(rn); free(cn); free(cp); free(zi); free(zv); free(rhs); free(rng); free(hasrng); free(lo); free(up); free(bset);
    h_free(&HR); h_free(&HC);
    if (rc) { lp_free(L); free(ci); ci = NULL; }
    if (isint) *isint = ci;
    return rc;
}

/* ---- presolve and standard form ---- */
typedef struct { int kind, j, i; double v; } PsOp;       /* 0: x_j = v; 1: x_j from row i (free column singleton); 2..5: x_j free was the pair of opposite columns j and i (v: the bound of x_j; the directions in the kind) */
struct LpPs { PsOp *op; int nop, cap; int *rows, nrows, rcap, vcap; double *vals; };     /* rows: the lists of the operations of kinds 6, 7 (rows) and 8 (columns, with vals) */

void lp_map_free(LpMap *M) { free(M->ckind); free(M->cpos); free(M->cval); free(M->rpos); free(M->rsgn); memset(M, 0, sizeof(*M)); }

static _Thread_local struct LpPs g_ps;   /* the operations of the last presolve (per thread: LPs may be solved on several threads at once, ISSUES 526) */
static void ps_push(int kind, int j, int i, double v) {
    if (g_ps.nop == g_ps.cap) { g_ps.cap = 2 * g_ps.cap + 256; g_ps.op = realloc(g_ps.op, sizeof(PsOp) * g_ps.cap); }
    g_ps.op[g_ps.nop].kind = kind; g_ps.op[g_ps.nop].j = j; g_ps.op[g_ps.nop].i = i; g_ps.op[g_ps.nop].v = v; g_ps.nop++;
}

int lp_to_sedumi(LpProb *L, int presolve, int verbose, SedumiProb *S, LpMap *M) { return lp_to_std(L, presolve, verbose, S, M, NULL); }
/* ub != NULL: the bounded form - upper bounds stay bounds (*ub, one per column of the standard
 * form, LP_INF: none) and a range row gets one slack with the bound ru - rl: no bound rows */
/* (28 October 2026, the night; ISSUES 626) linearly dependent equality rows (Andersen and Andersen 1995, section 4.7; Gurobi's
 * and HiGHS's presolves remove them), for both presolves (lpio.c's and simplex_pre.c's): the caller's equality rows, by rows
 * (ep, ej, ev: the active entries; eb: the sides; columns in [0, n)). Each row, the shortest first, is reduced by Gaussian
 * elimination against the pivot rows found before it; a row that reduces to zero is a combination of rows that stay, and
 * when its side reduces to zero too, dep[t] = 1 (the caller drops it: its dual 0, its logical basic, which keeps a basis
 * nonsingular). A side that does not reduce to zero is a contradiction the solve then finds; the row stays. The pivot: an
 * entry within a tenth of the row's largest, in the column of the fewest equations. Stops at a work cap (the pivot rows'
 * fill; what was found stands: each row found is a combination of pivot rows, which stay) and at the stop flag. Returns the
 * number found. The robustness copies of an independent tester's LPs with a tenth more rows each the sum of two equations
 * took 5 to 12 times the original's time (the quantile regressions: the dual form's 899 free variables; the coupled
 * networks: the dependent rows joined the network's parts in the factorization, the solves 10 times dearer), where
 * Gurobi's time hardly changed */
static _Thread_local int g_lp_nodep = 0;
void lp_set_nodeprows(int v) { g_lp_nodep = v; }
int lp_dependent_rows(int me, int n, const long *ep, const int *ej, const double *ev, const double *eb, char *dep, const volatile int *stop) {
    memset(dep, 0, (size_t)me);
    if (me < 2) return 0;
    const long nze = ep[me];
    int *rows = xm(sizeof(int) * (size_t)me), *cnt = xz(sizeof(int) * ((size_t)n + 1));
    {   int mx = 0; for (int t = 0; t < me; t++) if (ep[t + 1] - ep[t] > mx) mx = (int)(ep[t + 1] - ep[t]);
        int *b = xz(sizeof(int) * ((size_t)mx + 2));
        for (int t = 0; t < me; t++) b[ep[t + 1] - ep[t] + 1]++;
        for (int k = 0; k <= mx; k++) b[k + 1] += b[k];
        for (int t = 0; t < me; t++) rows[b[ep[t + 1] - ep[t]]++] = t;
        free(b); }
    for (long q = 0; q < nze; q++) if (ev[q] != 0) cnt[ej[q]]++;
    int *pk = xm(sizeof(int) * ((size_t)n + 1)); for (int j = 0; j < n; j++) pk[j] = -1;   /* the pivot of column j */
    int *pc = xm(sizeof(int) * (size_t)me); long *ps = xm(sizeof(long) * ((size_t)me + 1));
    double *pval = xm(sizeof(double) * (size_t)me), *pb = xm(sizeof(double) * (size_t)me), *pmx = xm(sizeof(double) * (size_t)me);
    long cap = 2 * nze + 1024, used = 0; int *pi = xm(sizeof(int) * (size_t)cap); double *pv = xm(sizeof(double) * (size_t)cap);
    double *acc = xz(sizeof(double) * ((size_t)n + 1)); int *lst = xm(sizeof(int) * ((size_t)n + 1)); char *inl = xz((size_t)n + 1);
    int *heap = xm(sizeof(int) * ((size_t)me + 1)); int *hst = xm(sizeof(int) * ((size_t)me + 1)); for (int k = 0; k < me; k++) hst[k] = -1;
    /* (the work: entries of pivot rows applied and stored; capped at 4 nnz + 50,000: the copies' dependent rows were found within
     * 1.5 nnz, while a transportation problem's rows taken by length fill until the cap, 40 nnz at first: n_transport_07, LP,
     * 0.016 -> 0.041 s for none found) */
    const long wcap = 4 * nze + 50000;
    long work = 0; int npiv = 0, ndep = 0, aborted = 0;
    for (int tt = 0; tt < me && !aborted; tt++) {
        if ((tt & 255) == 0 && stop && *stop) { aborted = 1; break; }
        const int t = rows[tt];
        int nl = 0, nh = 0; double smax = 0, rhs = eb[t], bmag = fabs(rhs);
        for (long q = ep[t]; q < ep[t + 1]; q++) {
            const int j = ej[q]; const double a = ev[q];
            if (a == 0) continue;
            if (!inl[j]) { inl[j] = 1; lst[nl++] = j; }
            acc[j] += a; if (fabs(a) > smax) smax = fabs(a);
            const int k = pk[j];
            if (k >= 0 && hst[k] != tt) { hst[k] = tt; int r = nh++; while (r > 0 && heap[(r - 1) / 2] > k) { heap[r] = heap[(r - 1) / 2]; r = (r - 1) / 2; } heap[r] = k; }
        }
        double mag = smax;
        while (nh > 0) {
            const int k = heap[0];       /* (the earliest pivot first: its row holds no column of an earlier pivot) */
            { const int last = heap[--nh]; int r = 0; for (;;) { int c = 2 * r + 1; if (c >= nh) break; if (c + 1 < nh && heap[c + 1] < heap[c]) c++; if (heap[c] >= last) break; heap[r] = heap[c]; r = c; } if (nh > 0) heap[r] = last; }
            const int jp = pc[k];
            const double f = acc[jp] / pval[k];
            if (f == 0) continue;
            rhs -= f * pb[k]; if (fabs(f * pb[k]) > bmag) bmag = fabs(f * pb[k]);
            if (fabs(f) * pmx[k] > mag) mag = fabs(f) * pmx[k];
            for (long q = ps[k]; q < ps[k + 1]; q++) {
                const int j = pi[q];
                if (!inl[j]) { inl[j] = 1; lst[nl++] = j; }
                acc[j] -= f * pv[q];
                const int k2 = pk[j];
                if (k2 > k && hst[k2] != tt) { hst[k2] = tt; int r = nh++; while (r > 0 && heap[(r - 1) / 2] > k2) { heap[r] = heap[(r - 1) / 2]; r = (r - 1) / 2; } heap[r] = k2; }
            }
            acc[jp] = 0;
            work += ps[k + 1] - ps[k];
        }
        /* what is left on the columns not yet pivots (a pivot column's entry is eliminated; its residue is rounding) */
        double rmax = 0;
        for (int q = 0; q < nl; q++) { const int j = lst[q]; if (pk[j] >= 0) acc[j] = 0; else if (fabs(acc[j]) > rmax) rmax = fabs(acc[j]); }
        if (rmax <= 1e-11 * mag) {
            if (fabs(rhs) <= 1e-9 * (1.0 + bmag)) { dep[t] = 1; ndep++; }
        } else {
            int jb = -1;
            for (int q = 0; q < nl; q++) { const int j = lst[q]; if (pk[j] < 0 && fabs(acc[j]) >= 0.1 * rmax && (jb < 0 || cnt[j] < cnt[jb])) jb = j; }
            long need = 0; for (int q = 0; q < nl; q++) { const int j = lst[q]; if (pk[j] < 0 && j != jb && fabs(acc[j]) > 1e-14 * rmax) need++; }
            if (used + need + 1 > cap) { cap = 2 * (used + need + 1); int *pi2 = realloc(pi, sizeof(int) * (size_t)cap); double *pv2 = realloc(pv, sizeof(double) * (size_t)cap);
                if (!pi2 || !pv2) { fprintf(stderr, "brisk: out of memory (LP presolve)\n"); exit(1); } pi = pi2; pv = pv2; }
            const int k = npiv++;
            pc[k] = jb; pval[k] = acc[jb]; pb[k] = rhs; pmx[k] = rmax; ps[k] = used;
            for (int q = 0; q < nl; q++) { const int j = lst[q]; if (pk[j] < 0 && j != jb && fabs(acc[j]) > 1e-14 * rmax) { pi[used] = j; pv[used++] = acc[j]; } }
            ps[k + 1] = used; pk[jb] = k;
            work += need;
        }
        for (int q = 0; q < nl; q++) { acc[lst[q]] = 0; inl[lst[q]] = 0; }
        if (work > wcap) aborted = 1;
    }
    if (getenv("BRISK_LPDEPDBG")) fprintf(stderr, "  [dependent rows: %d equations, %d pivots, %d dependent%s; work %ld of %ld]\n", me, npiv, ndep, aborted ? ", stopped at the work cap or the stop flag" : "", work, wcap);
    free(rows); free(cnt); free(pk); free(pc); free(ps); free(pval); free(pb); free(pmx); free(pi); free(pv); free(acc); free(lst); free(inl); free(heap); free(hst);
    return ndep;
}
int lp_to_std(LpProb *L, int presolve, int verbose, SedumiProb *S, LpMap *M, double **ub) {
    const int bform = ub != NULL;
    const int m = L->m, n = L->n;
    memset(S, 0, sizeof(*S)); memset(M, 0, sizeof(*M));
    g_ps.nop = 0; g_ps.nrows = 0;
    M->n0 = n; M->m0 = m; M->maxim = L->maxim;
    /* minimization */
    double *c = xm(sizeof(double) * (n + 1)); double c0 = L->maxim ? -L->c0 : L->c0;
    for (int j = 0; j < n; j++) c[j] = L->maxim ? -L->c[j] : L->c[j];
    double *l = xm(sizeof(double) * (n + 1)), *u = xm(sizeof(double) * (n + 1)), *rl = xm(sizeof(double) * (m + 1)), *ru = xm(sizeof(double) * (m + 1));
    memcpy(l, L->l, sizeof(double) * n); memcpy(u, L->u, sizeof(double) * n); memcpy(rl, L->rl, sizeof(double) * m); memcpy(ru, L->ru, sizeof(double) * m);
    for (int j = 0; j < n; j++) { if (l[j] <= -LP_INF) l[j] = -LP_INF; if (u[j] >= LP_INF) u[j] = LP_INF; }
    for (int i = 0; i < m; i++) { if (rl[i] <= -LP_INF) rl[i] = -LP_INF; if (ru[i] >= LP_INF) ru[i] = LP_INF; }
    /* rows of A */
    int *Rp = xz(sizeof(int) * (m + 2)), *Rj = xm(sizeof(int) * (L->Ap[n] + 1)); double *Rx = xm(sizeof(double) * (L->Ap[n] + 1));
    for (int p = 0; p < L->Ap[n]; p++) Rp[L->Ai[p] + 2]++;
    for (int i = 0; i < m; i++) Rp[i + 2] += Rp[i + 1];
    for (int j = 0; j < n; j++) for (int p = L->Ap[j]; p < L->Ap[j + 1]; p++) { const int q = Rp[L->Ai[p] + 1]++; Rj[q] = j; Rx[q] = L->Ax[p]; }
    char *ra = xm(m + 1), *ca = xm(n + 1);            /* alive */
    memset(ra, 1, m); memset(ca, 1, n);
    int *rcnt = xz(sizeof(int) * (m + 1)), *ccnt = xz(sizeof(int) * (n + 1));     /* alive entries */
    for (int j = 0; j < n; j++) for (int p = L->Ap[j]; p < L->Ap[j + 1]; p++) { rcnt[L->Ai[p]]++; ccnt[j]++; }
    double *xfix = xz(sizeof(double) * (n + 1));
    int status = 0, nfixed = 0, nrowrm = 0;
    const double ftol = 1e-9;
#define FIXCOL(j_, v_) do { const int jj = (j_); const double vv = (v_); ca[jj] = 0; xfix[jj] = vv; ps_push(0, jj, -1, vv); nfixed++; c0 += c[jj] * vv; \
        for (int p_ = L->Ap[jj]; p_ < L->Ap[jj + 1]; p_++) { const int ii = L->Ai[p_]; if (!ra[ii]) continue; rcnt[ii]--; \
            if (vv != 0.0) { if (rl[ii] > -LP_INF) rl[ii] -= L->Ax[p_] * vv; if (ru[ii] < LP_INF) ru[ii] -= L->Ax[p_] * vv; } } } while (0)
#define DROPROW(i_) do { const int ii = (i_); ra[ii] = 0; nrowrm++; for (int q_ = Rp[ii]; q_ < Rp[ii + 1]; q_++) if (ca[Rj[q_]]) ccnt[Rj[q_]]--; } while (0)
    for (int pass = 0; presolve > 0 && pass < 50 && status == 0; pass++) {
        int changed = 0;
        /* rows */
        for (int i = 0; i < m && status == 0; i++) {
            if (!ra[i]) continue;
            if (rl[i] > ru[i] + ftol * (1.0 + fabs(rl[i]))) { status = 1; break; }
            if (rcnt[i] == 0) {
                if (rl[i] > ftol * (1.0 + fabs(rl[i])) || ru[i] < -ftol * (1.0 + fabs(ru[i]))) { status = 1; break; }
                DROPROW(i); changed = 1; continue;
            }
            if (rcnt[i] == 1) {
                int j = -1; double a = 0;
                for (int q = Rp[i]; q < Rp[i + 1]; q++) if (ca[Rj[q]]) { j = Rj[q]; a = Rx[q]; }
                double lo = -LP_INF, hi = LP_INF;
                if (a > 0) { if (rl[i] > -LP_INF) lo = rl[i] / a; if (ru[i] < LP_INF) hi = ru[i] / a; }
                else { if (ru[i] < LP_INF) lo = ru[i] / a; if (rl[i] > -LP_INF) hi = rl[i] / a; }
                if (lo > l[j]) l[j] = lo;
                if (hi < u[j]) u[j] = hi;
                if (l[j] > u[j] + ftol * (1.0 + fabs(l[j]))) { status = 1; break; }
                if (l[j] > u[j]) l[j] = u[j] = 0.5 * (l[j] + u[j]);
                DROPROW(i); changed = 1; continue;
            }
            /* activity bounds */
            double amin = 0, amax = 0; int imin = 0, imax = 0;
            for (int q = Rp[i]; q < Rp[i + 1]; q++) {
                const int j = Rj[q]; if (!ca[j]) continue;
                const double a = Rx[q];
                if (a > 0) { if (l[j] > -LP_INF) amin += a * l[j]; else imin++; if (u[j] < LP_INF) amax += a * u[j]; else imax++; }
                else { if (u[j] < LP_INF) amin += a * u[j]; else imin++; if (l[j] > -LP_INF) amax += a * l[j]; else imax++; }
            }
            const double sc = 1.0 + fmax(fabs(amin), fabs(amax));
            if (!imax && rl[i] > -LP_INF && amax < rl[i] - 1e-7 * (sc + fabs(rl[i]))) { status = 1; break; }
            if (!imin && ru[i] < LP_INF && amin > ru[i] + 1e-7 * (sc + fabs(ru[i]))) { status = 1; break; }
            if (!imin && rl[i] > -LP_INF && amin >= rl[i] - ftol * (sc + fabs(rl[i])) && rl[i] != ru[i]) { rl[i] = -LP_INF; changed = 1; }
            if (!imax && ru[i] < LP_INF && amax <= ru[i] + ftol * (sc + fabs(ru[i])) && rl[i] != ru[i]) { ru[i] = LP_INF; changed = 1; }
            if (rl[i] <= -LP_INF && ru[i] >= LP_INF) { DROPROW(i); changed = 1; continue; }
            /* forcing: the row can only be met with every variable at the bound of that side */
            int force = 0;
            if (!imax && rl[i] > -LP_INF && fabs(amax - rl[i]) <= ftol * (sc + fabs(rl[i]))) force = 1;       /* at amax */
            else if (!imin && ru[i] < LP_INF && fabs(amin - ru[i]) <= ftol * (sc + fabs(ru[i]))) force = -1;   /* at amin */
            if (force && getenv("BRISK_LPNOFORCING")) force = 0;     /* (experiments) */
            /* (27 October 2026, the night) only when the points the fixing removes lie within each column's tolerance of
             * its bound: they have activities between the side and the extreme one (gap apart), column j at most
             * gap / |a_ij| from its bound. The gap's test against the row's tolerance alone depends on the row's scale (a
             * row of coefficients 1e-10 met at 0 by x = 0 forced its columns to the far bounds: dell, LP, in the simplex
             * method's presolve, which has the same test) */
            const double fgap = force > 0 ? amax - rl[i] : force < 0 ? ru[i] - amin : 0;
            if (force && fgap > 0)
                for (int q = Rp[i]; q < Rp[i + 1]; q++) {
                    const int j = Rj[q]; if (!ca[j]) continue;
                    const double v = ((Rx[q] > 0) == (force > 0)) ? u[j] : l[j];
                    if (fgap > fabs(Rx[q]) * ftol * (1.0 + fabs(v))) { force = 0; break; }
                }
            if (force) {
                for (int q = Rp[i]; q < Rp[i + 1]; q++) {
                    const int j = Rj[q]; if (!ca[j]) continue;
                    const double v = ((Rx[q] > 0) == (force > 0)) ? u[j] : l[j];
                    FIXCOL(j, v);
                }
                DROPROW(i); changed = 1; continue;
            }
        }
        if (status) break;
        /* columns */
        for (int j = 0; j < n && status == 0; j++) {
            if (!ca[j]) continue;
            if (l[j] > u[j] + ftol * (1.0 + fabs(l[j]))) { status = 1; break; }
            if (l[j] > -LP_INF && u[j] < LP_INF && u[j] - l[j] <= 1e-13 * (1.0 + fabs(l[j]))) { FIXCOL(j, l[j]); changed = 1; continue; }
            if (ccnt[j] == 0) {
                double v;
                if (c[j] > 0) { if (l[j] <= -LP_INF) { status = 2; break; } v = l[j]; }
                else if (c[j] < 0) { if (u[j] >= LP_INF) { status = 2; break; } v = u[j]; }
                else v = l[j] > -LP_INF ? (u[j] < LP_INF ? (l[j] <= 0 && u[j] >= 0 ? 0.0 : l[j]) : fmax(l[j], 0.0)) : (u[j] < LP_INF ? fmin(u[j], 0.0) : 0.0);
                FIXCOL(j, v); changed = 1; continue;
            }
            /* dual fixing: the column can move down (up) without hurting any row */
            int down_ok = 1, up_ok = 1;
            for (int p = L->Ap[j]; p < L->Ap[j + 1] && (down_ok || up_ok); p++) {
                const int i = L->Ai[p]; if (!ra[i]) continue;
                const double a = L->Ax[p];
                /* decreasing x: a > 0 lowers the row (bad if rl finite), a < 0 raises it (bad if ru finite) */
                if (a > 0) { if (rl[i] > -LP_INF) down_ok = 0; if (ru[i] < LP_INF) up_ok = 0; }
                else { if (ru[i] < LP_INF) down_ok = 0; if (rl[i] > -LP_INF) up_ok = 0; }
            }
            if (getenv("BRISK_LPNODUALFIX")) down_ok = up_ok = 0;      /* (experiments) */
            if (down_ok && c[j] > 0) { if (l[j] <= -LP_INF) continue; FIXCOL(j, l[j]); changed = 1; continue; }
            if (up_ok && c[j] < 0) { if (u[j] >= LP_INF) continue; FIXCOL(j, u[j]); changed = 1; continue; }
            if (down_ok && c[j] == 0 && l[j] > -LP_INF) { FIXCOL(j, l[j]); changed = 1; continue; }
            if (up_ok && c[j] == 0 && u[j] < LP_INF) { FIXCOL(j, u[j]); changed = 1; continue; }
            /* no cost and no bound in a harmless direction: far enough along it every row of the
             * column holds - the rows and the column go, x_j is found from the rows afterwards */
            if (presolve > 1 && c[j] == 0 && ((up_ok && u[j] >= LP_INF) || (down_ok && l[j] <= -LP_INF))) {
                if (g_ps.nrows + ccnt[j] + 1 > g_ps.rcap) { g_ps.rcap = 2 * g_ps.rcap + ccnt[j] + 256; g_ps.rows = realloc(g_ps.rows, sizeof(int) * g_ps.rcap); }
                const int r0 = g_ps.nrows;
                for (int p = L->Ap[j]; p < L->Ap[j + 1]; p++) if (ra[L->Ai[p]]) g_ps.rows[g_ps.nrows++] = L->Ai[p];
                g_ps.rows[g_ps.nrows++] = -1;
                ps_push(6, j, r0, (up_ok && u[j] >= LP_INF) ? (l[j] > -LP_INF ? l[j] : 0.0) : (u[j] < LP_INF ? u[j] : 0.0));
                g_ps.op[g_ps.nop - 1].kind = (up_ok && u[j] >= LP_INF) ? 6 : 7;        /* 6: x_j upwards from v, 7: downwards */
                ca[j] = 0; nfixed++;
                for (int p = L->Ap[j]; p < L->Ap[j + 1]; p++) { const int i = L->Ai[p]; if (ra[i]) { rcnt[i]--; DROPROW(i); } }
                changed = 1; continue;
            }
            /* a free column with one entry */
            if (presolve > 1 && ccnt[j] == 1 && l[j] <= -LP_INF && u[j] >= LP_INF) {
                int i = -1; double a = 0;
                for (int p = L->Ap[j]; p < L->Ap[j + 1]; p++) if (ra[L->Ai[p]]) { i = L->Ai[p]; a = L->Ax[p]; }
                if (i < 0) continue;
                if (rl[i] == ru[i]) {
                    /* x_j = (b - sum a_ik x_k) / a: costs c_k -= c_j a_ik / a, constant += c_j b / a */
                    double amaxr = 0; for (int q = Rp[i]; q < Rp[i + 1]; q++) if (ca[Rj[q]]) amaxr = fmax(amaxr, fabs(Rx[q]));
                    if (fabs(a) < 1e-3 * amaxr) continue;
                    const double f = c[j] / a;
                    if (f != 0.0) { for (int q = Rp[i]; q < Rp[i + 1]; q++) { const int k = Rj[q]; if (ca[k] && k != j) c[k] -= f * Rx[q]; } c0 += f * rl[i]; }
                    ca[j] = 0; ps_push(1, j, i, a); nfixed++;
                    DROPROW(i); changed = 1; continue;
                } else if (c[j] == 0.0) {
                    /* the row can always be met: x_j takes what is needed */
                    ca[j] = 0; ps_push(1, j, i, a); nfixed++;
                    DROPROW(i); changed = 1; continue;
                }
            }
        }
        /* ---- dominated columns by the bounds of the row duals (21 October 2026, LP/SOCP; Andersen and Andersen 1995).
         * At every dual feasible y the reduced cost is d_j = c_j - a_j'y, with y_i >= 0 on a row with only a lower side,
         * y_i <= 0 with only an upper side. A column with no upper bound needs d_j >= 0 (no lower bound: d_j <= 0), which
         * bounds each y_i of its rows by the others' extreme values. A column whose d_j is positive for every y in those
         * bounds is at its lower bound in every optimum (negative: at its upper bound) and is fixed there. An upper bound
         * counts as absent when c_j > 0 and at x_j = u_j each row of the column holds whatever its other variables do
         * (all the other entries of the row of one sign, from finite lower bounds): no optimum has x_j > u_j then, so
         * dropping the bound leaves the optimal set as it is. Set covering, x in [0, 1] and A x >= 1: y_i <= the least
         * cost of a column covering row i, and a column costlier than the sum of those over its rows goes (lp_setcover_5k:
         * 33,000 of 40,001 columns). BRISK_LPNODOMCOL=1: off. */
        if (!changed && presolve > 1 && status == 0 && !getenv("BRISK_LPNODOMCOL")) {
            double *ylo = xm(sizeof(double) * (m + 1)), *yhi = xm(sizeof(double) * (m + 1)), *slp = xz(sizeof(double) * (m + 1)), *sln = xz(sizeof(double) * (m + 1));
            int *npos = xz(sizeof(int) * (m + 1)), *nneg = xz(sizeof(int) * (m + 1)), *nplinf = xz(sizeof(int) * (m + 1)), *nnlinf = xz(sizeof(int) * (m + 1));
            for (int i = 0; i < m; i++) {
                ylo[i] = rl[i] > -LP_INF && ru[i] >= LP_INF ? 0.0 : -LP_INF; yhi[i] = ru[i] < LP_INF && rl[i] <= -LP_INF ? 0.0 : LP_INF;
                if (!ra[i]) continue;
                for (int q = Rp[i]; q < Rp[i + 1]; q++) { const int k = Rj[q]; if (!ca[k]) continue; const double a = Rx[q];
                    if (a > 0) { npos[i]++; if (l[k] > -LP_INF) slp[i] += a * l[k]; else nplinf[i]++; }
                    else if (a < 0) { nneg[i]++; if (l[k] > -LP_INF) sln[i] += a * l[k]; else nnlinf[i]++; } }
            }
            /* the effective bounds: an upper bound that no optimum can exceed counts as none */
            char *unone = xz(n + 1);
            for (int j = 0; j < n; j++) {
                if (!ca[j]) continue;
                if (u[j] >= LP_INF) { unone[j] = 1; continue; }
                if (!(c[j] > 0) || l[j] <= -LP_INF) continue;
                int ok = 1;
                for (int p = L->Ap[j]; p < L->Ap[j + 1] && ok; p++) {
                    const int i = L->Ai[p]; if (!ra[i]) continue; const double a = L->Ax[p];
                    if (a > 0 && rl[i] > -LP_INF) {           /* lowering x_j to u_j must keep the row above rl */
                        if (nneg[i] || nplinf[i]) { ok = 0; break; }
                        const double rest = slp[i] - a * l[j];
                        if (!(a * u[j] + rest >= rl[i] - 1e-12 * (1.0 + fabs(rl[i])))) ok = 0;
                    } else if (a < 0 && ru[i] < LP_INF) {     /* and below ru */
                        if (npos[i] || nnlinf[i]) { ok = 0; break; }
                        const double rest = sln[i] - a * l[j];
                        if (!(a * u[j] + rest <= ru[i] + 1e-12 * (1.0 + fabs(ru[i])))) ok = 0;
                    }
                }
                unone[j] = ok;
            }
            /* the bounds of y from the columns: a_j'y <= c_j (no upper bound), a_j'y >= c_j (no lower bound) */
            for (int j = 0; j < n; j++) {
                if (!ca[j]) continue;
                const int nolo = l[j] <= -LP_INF;
                for (int side = 0; side < 2; side++) {          /* side 0: a_j'y <= c_j; side 1: -a_j'y <= -c_j */
                    if (side == 0 ? !unone[j] : !nolo) continue;
                    const double sg = side == 0 ? 1.0 : -1.0, cc = sg * c[j];
                    double S_ = 0; int nI = 0;
                    for (int p = L->Ap[j]; p < L->Ap[j + 1]; p++) { const int i = L->Ai[p]; if (!ra[i]) continue; const double a = sg * L->Ax[p];
                        const double mn = a > 0 ? (ylo[i] > -LP_INF ? a * ylo[i] : -LP_INF) : (yhi[i] < LP_INF ? a * yhi[i] : -LP_INF);
                        if (mn <= -LP_INF) nI++; else S_ += mn; }
                    if (nI > 1) continue;
                    for (int p = L->Ap[j]; p < L->Ap[j + 1]; p++) { const int i = L->Ai[p]; if (!ra[i]) continue; const double a = sg * L->Ax[p]; if (a == 0.0) continue;
                        const double mn = a > 0 ? (ylo[i] > -LP_INF ? a * ylo[i] : -LP_INF) : (yhi[i] < LP_INF ? a * yhi[i] : -LP_INF);
                        double rest;
                        if (nI == 0) rest = S_ - mn; else if (mn <= -LP_INF) rest = S_; else continue;
                        const double bnd = (cc - rest) / a;      /* a y_i <= cc - rest */
                        if (a > 0) { if (bnd < yhi[i]) yhi[i] = bnd; } else { if (bnd > ylo[i]) ylo[i] = bnd; }
                    }
                }
            }
            /* the columns whose reduced cost has one sign over the box of y */
            int ndom = 0;
            for (int j = 0; j < n; j++) {
                if (!ca[j]) continue;
                double mx = 0, mn = 0; int imx = 0, imn = 0;
                for (int p = L->Ap[j]; p < L->Ap[j + 1]; p++) { const int i = L->Ai[p]; if (!ra[i]) continue; const double a = L->Ax[p];
                    if (a > 0) { if (yhi[i] < LP_INF) mx += a * yhi[i]; else imx++; if (ylo[i] > -LP_INF) mn += a * ylo[i]; else imn++; }
                    else if (a < 0) { if (ylo[i] > -LP_INF) mx += a * ylo[i]; else imx++; if (yhi[i] < LP_INF) mn += a * yhi[i]; else imn++; } }
                if (l[j] > -LP_INF && !imx && c[j] - mx > 1e-9 * (1.0 + fabs(c[j]) + fabs(mx))) { FIXCOL(j, l[j]); ndom++; changed = 1; continue; }
                if (u[j] < LP_INF && !imn && c[j] - mn < -1e-9 * (1.0 + fabs(c[j]) + fabs(mn))) { FIXCOL(j, u[j]); ndom++; changed = 1; continue; }
            }
            if (ndom && verbose > 1) printf("presolve: %d dominated columns fixed at a bound (the bounds of the row duals)\n", ndom);
            free(ylo); free(yhi); free(slp); free(sln); free(npos); free(nneg); free(nplinf); free(nnlinf); free(unone);
        }
        if (!changed) break;
    }
    /* the bounded form: a pair of columns x_j, x_k >= 0 with a_k = -a_j and c_k = -c_j is one free
     * variable x_j - x_k (its split has no interior on the dual side: z_j + z_k = 0) */
    int npair = 0;
    if (bform && presolve > 1 && status == 0) {
        /* (a column with one finite bound moves away from it in the direction dir: +1 from a lower, -1 from an upper bound) */
#define ONESIDED(j_) ((l[j_] > -LP_INF) != (u[j_] < LP_INF))
#define DIROF(j_) (l[j_] > -LP_INF ? 1.0 : -1.0)
        int nc = 0, *cand = xm(sizeof(int) * (n + 1)); double *key = xm(sizeof(double) * (n + 1)), *wk = xz(sizeof(double) * (m + 1));
        for (int j = 0; j < n; j++) if (ca[j] && ONESIDED(j) && ccnt[j] > 0) {
            double h = fabs(c[j]) * 0.6180339887;
            for (int p = L->Ap[j]; p < L->Ap[j + 1]; p++) if (ra[L->Ai[p]]) h += fabs(L->Ax[p]) * (1.0 + 0.7548776662 * (double)((L->Ai[p] * 2654435761u) % 1000003u));
            key[j] = h; cand[nc++] = j;
        }
        /* sort the candidates by key (shell sort; the keys of a pair are equal up to the order of summation) */
        for (int gap = nc / 2; gap > 0; gap /= 2) for (int a = gap; a < nc; a++) { const int t = cand[a]; int b = a; while (b >= gap && key[cand[b - gap]] > key[t]) { cand[b] = cand[b - gap]; b -= gap; } cand[b] = t; }
        for (int a = 0; a + 1 < nc; a++) {
            const int j = cand[a]; if (!ca[j] || !ONESIDED(j)) continue;
            const double dj = DIROF(j);
            for (int b = a + 1; b < nc && key[cand[b]] - key[j] <= 1e-9 * (1.0 + fabs(key[j])); b++) {
                const int k = cand[b]; if (!ca[k] || !ONESIDED(k) || ccnt[k] != ccnt[j]) continue;
                const double dk = DIROF(k);
                if (dk * c[k] != -dj * c[j]) continue;
                for (int p = L->Ap[j]; p < L->Ap[j + 1]; p++) if (ra[L->Ai[p]]) wk[L->Ai[p]] = dj * L->Ax[p];
                int same = 1;
                for (int p = L->Ap[k]; p < L->Ap[k + 1]; p++) if (ra[L->Ai[p]] && (wk[L->Ai[p]] == 0.0 || wk[L->Ai[p]] != -dk * L->Ax[p])) { same = 0; break; }
                for (int p = L->Ap[j]; p < L->Ap[j + 1]; p++) wk[L->Ai[p]] = 0.0;
                if (!same) continue;
                /* x_j = b_j + dj p, x_k = b_k + dk q, p, q >= 0: x_j free and x_k = b_k; p and q come back in lp_recover */
                const double bj = dj > 0 ? l[j] : u[j], bk = dk > 0 ? l[k] : u[k];
                FIXCOL(k, bk);
                ps_push(2 + (dj < 0) + 2 * (dk < 0), j, k, bj);
                l[j] = -LP_INF; u[j] = LP_INF; npair++;
                break;
            }
        }
        free(cand); free(key); free(wk);
        if (npair && verbose > 0) printf("presolve: %d pairs of opposite columns are free variables\n", npair);
    }
    /* ---- upper bounds that a row implies (x_j <= u_j follows from the row and the bounds of its
     * other columns as they are now) are dropped: the column can then be a pivot of the eliminator,
     * and the interior-point method has one bound less. Row by row, each test with the current
     * bounds, so that no bound is dropped on the strength of one dropped before. */
    int nredb = 0; double *udrop = NULL;        /* (the bounds dropped: a substitution of a row of two entries may need them back) */
    if (bform && presolve > 1 && status == 0 && !getenv("BRISK_LPNOREDB")) {
        for (int i = 0; i < m; i++) {
            if (!ra[i] || rcnt[i] < 2) continue;
            double smin = 0, smax = 0; int imin = 0, imax = 0;
            for (int q = Rp[i]; q < Rp[i + 1]; q++) {
                const int k = Rj[q]; if (!ca[k]) continue;
                const double a = Rx[q];
                if (a > 0) { if (l[k] > -LP_INF) smin += a * l[k]; else imin++; if (u[k] < LP_INF) smax += a * u[k]; else imax++; }
                else { if (u[k] < LP_INF) smin += a * u[k]; else imin++; if (l[k] > -LP_INF) smax += a * l[k]; else imax++; }
            }
            for (int q = Rp[i]; q < Rp[i + 1]; q++) {
                const int j = Rj[q]; if (!ca[j] || !(u[j] < LP_INF) || !(l[j] > -LP_INF)) continue;
                const double a = Rx[q];
                double ub;
                if (a > 0) {                       /* a x_j <= ru - (the least of the rest) */
                    if (!(ru[i] < LP_INF) || imin > 0) continue;            /* (x_j itself has a lower bound: not among the infinite ones) */
                    ub = (ru[i] - (smin - a * l[j])) / a;
                } else {                           /* a x_j >= rl - (the most of the rest) */
                    if (!(rl[i] > -LP_INF) || imax > 0) continue;
                    ub = (rl[i] - (smax - a * l[j])) / a;
                }
                if (ub <= u[j] + 1e-9 * (1.0 + fabs(u[j]))) {
                    /* the totals of this row without the bound */
                    if (a > 0) { smax -= a * u[j]; imax++; } else { smin -= a * u[j]; imin++; }
                    if (!udrop) { udrop = xm(sizeof(double) * (n + 1)); for (int t = 0; t < n; t++) udrop[t] = LP_INF; }
                    udrop[j] = u[j]; u[j] = LP_INF; nredb++;
                }
            }
        }
        if (nredb && verbose > 0) printf("presolve: %d upper bounds are implied by rows and dropped\n", nredb);
    }
    /* ---- the eliminator: an equality row i and a column j that the row keeps within its bounds
     * (an implied free column: x_j = (b_i - sum a_ik x_k) / a_ij holds its bounds for all x_k in
     * theirs) - x_j is substituted in its other rows and in the cost, the row and the column go.
     * Taken when the substitution cannot add nonzeros: (len(row) - 1)(len(col) - 1) <= len(row) + len(col) - 1 */
    const int *A2p = L->Ap, *A2i = L->Ai; const double *A2x = L->Ax;
    int *E_p = NULL, *E_i = NULL; double *E_x = NULL; int nelim = 0;
    if (bform && presolve > 1 && status == 0 && !getenv("BRISK_LPNOELIM")) {
        int **rc = xz(sizeof(int *) * (m + 1)), *rn = xz(sizeof(int) * (m + 1)), *rcap = xz(sizeof(int) * (m + 1)); double **rv = xz(sizeof(double *) * (m + 1));
        int **cr = xz(sizeof(int *) * (n + 1)), *cn = xz(sizeof(int) * (n + 1)), *ccap = xz(sizeof(int) * (n + 1)), *cl = xz(sizeof(int) * (n + 1));
        for (int i = 0; i < m; i++) if (ra[i]) { rcap[i] = rcnt[i] + 4; rc[i] = xm(sizeof(int) * rcap[i]); rv[i] = xm(sizeof(double) * rcap[i]); }
        for (int j = 0; j < n; j++) if (ca[j]) {
            ccap[j] = ccnt[j] + 4; cr[j] = xm(sizeof(int) * ccap[j]);
            for (int p = L->Ap[j]; p < L->Ap[j + 1]; p++) { const int i = L->Ai[p]; if (!ra[i] || L->Ax[p] == 0.0) continue; rc[i][rn[i]] = j; rv[i][rn[i]++] = L->Ax[p]; cr[j][cn[j]++] = i; cl[j]++; }
        }
        int *pos = xz(sizeof(int) * (n + 1)); long dbg[6] = {0, 0, 0, 0, 0, 0}, d2[4] = {0, 0, 0, 0};
        const int maxlc = getenv("BRISK_LPELIMLC") ? atoi(getenv("BRISK_LPELIMLC")) : 12;
        const int xfill = getenv("BRISK_LPELIMFILL") ? atoi(getenv("BRISK_LPELIMFILL")) : 40;
        const int dbl2 = getenv("BRISK_LPELIM2") ? atoi(getenv("BRISK_LPELIM2")) : 1;
        for (int round = 0; round < 6; round++) {
            int did = 0;
            for (int i = 0; i < m; i++) {
                if (g_lp_stopf && *g_lp_stopf) break;
                if (!ra[i] || rl[i] != ru[i] || rn[i] < 2) continue;
                const int lr = rn[i];
                /* the activity of the row without one column: the sums over the finite bounds and the counts of infinite ones */
                double smin = 0, smax = 0, amax = 0; int imin = 0, imax = 0;
                for (int q = 0; q < lr; q++) {
                    const int k = rc[i][q]; const double a = rv[i][q];
                    if (fabs(a) > amax) amax = fabs(a);
                    if (a > 0) { if (l[k] > -LP_INF) smin += a * l[k]; else imin++; if (u[k] < LP_INF) smax += a * u[k]; else imax++; }
                    else { if (u[k] < LP_INF) smin += a * u[k]; else imin++; if (l[k] > -LP_INF) smax += a * l[k]; else imax++; }
                }
                int bj = -1, bq = -1, blc = 1 << 30;
                for (int q = 0; q < lr; q++) {
                    const int j = rc[i][q]; const double a = rv[i][q];
                    const int lc = cl[j];
                    if (lc > maxlc && (lr != 2 || dbl2 == 0)) { dbg[0]++; if (lr == 2) d2[0]++; continue; }
                    if (lc >= blc) continue;
                    if ((long)(lr - 1) * (lc - 1) > (long)lr + lc - 1 + xfill) { dbg[1]++; continue; }
                    if (fabs(a) < 1e-2 * amax) { if (lr == 2) d2[1]++; continue; }
                    /* the rest of the row: [rmin, rmax] */
                    double rmin, rmax; int fmin_ = 1, fmax_ = 1;       /* finite */
                    { double cmin, cmax; int nmin = imin, nmax = imax;
                      if (a > 0) { cmin = l[j] > -LP_INF ? a * l[j] : 0.0; if (l[j] <= -LP_INF) nmin--; cmax = u[j] < LP_INF ? a * u[j] : 0.0; if (u[j] >= LP_INF) nmax--; }
                      else { cmin = u[j] < LP_INF ? a * u[j] : 0.0; if (u[j] >= LP_INF) nmin--; cmax = l[j] > -LP_INF ? a * l[j] : 0.0; if (l[j] <= -LP_INF) nmax--; }
                      rmin = smin - cmin; rmax = smax - cmax; fmin_ = nmin == 0; fmax_ = nmax == 0; }
                    /* x_j = (b - rest) / a in [lo, hi] */
                    const double b = rl[i], tl = 1e-9 * (1.0 + fabs(b) + fabs(rmin) + fabs(rmax));
                    int lo_ok, hi_ok;
                    if (a > 0) { lo_ok = l[j] <= -LP_INF || (fmax_ && (b - rmax) >= a * l[j] - tl); hi_ok = u[j] >= LP_INF || (fmin_ && (b - rmin) <= a * u[j] + tl); }
                    else { lo_ok = l[j] <= -LP_INF || (fmin_ && (b - rmin) <= a * l[j] + tl); hi_ok = u[j] >= LP_INF || (fmax_ && (b - rmax) >= a * u[j] - tl); }
                    /* a row of two columns: the bounds of x_j become bounds of the other column (taken below) */
                    if (!lo_ok || !hi_ok) {
                        /* a row of two columns: the bounds of x_j become bounds of the other column (below) -
                         * only where that column has a bound already (a new bound far from the solution
                         * would make it a free variable in disguise) */
                        if (lr != 2) { dbg[2 + (lo_ok ? 1 : 0)]++; continue; }
                        const int k2 = rc[i][1 - q]; const double ak = rv[i][1 - q];
                        const int needs_lo = a > 0 ? u[j] < LP_INF : l[j] > -LP_INF, needs_hi = a > 0 ? l[j] > -LP_INF : u[j] < LP_INF;   /* sides of ak x_k */
                        const int new_l = ak > 0 ? needs_lo : needs_hi, new_u = ak > 0 ? needs_hi : needs_lo;
                        if ((new_l && l[k2] <= -LP_INF) || (new_u && u[k2] >= LP_INF && !(dbl2 && udrop && udrop[k2] < LP_INF))) { dbg[2 + (lo_ok ? 1 : 0)]++; d2[2]++; continue; }
                    }
                    /* a pivot that is not small in its column either. (28 October 2026, the night) The column's largest
                     * entry only where the test applies: a row of two entries takes candidate columns of any length,
                     * and the scan of each of their rows for the column was made for every such row and never used;
                     * neos-4984115-canche (LP, MIPLIB 2017 collection; 119,078 rows of two entries taken) spent 45 s
                     * of its 46 s presolve here. The same substitutions */
                    if (lr != 2 || dbl2 == 0) {
                        double cmaxa = 0;
                        for (int t = 0; t < cn[j]; t++) { const int r = cr[j][t]; if (!ra[r]) continue; for (int e = 0; e < rn[r]; e++) if (rc[r][e] == j) { cmaxa = fmax(cmaxa, fabs(rv[r][e])); break; } }
                        if (fabs(a) < 1e-2 * cmaxa) { if (lr == 2) d2[3]++; continue; }
                    }
                    bj = j; bq = q; blc = lc;
                }
                if (bj < 0) { if (lr == 2) dbg[4]++; continue; }
                if (lr == 2) dbg[5]++;
                const int j = bj; const double a = rv[i][bq], b = rl[i];
                if (lr == 2) {
                    /* x_j = (b - ak x_k) / a in [l_j, u_j]: the interval of x_k that keeps it there */
                    const int k = rc[i][1 - bq]; const double ak = rv[i][1 - bq];
                    double lo = -LP_INF, hi = LP_INF;                 /* of ak x_k = b - a x_j */
                    if (a > 0) { if (u[j] < LP_INF) lo = b - a * u[j]; if (l[j] > -LP_INF) hi = b - a * l[j]; }
                    else { if (l[j] > -LP_INF) lo = b - a * l[j]; if (u[j] < LP_INF) hi = b - a * u[j]; }
                    double lk = -LP_INF, uk = LP_INF;
                    if (ak > 0) { if (lo > -LP_INF) lk = lo / ak; if (hi < LP_INF) uk = hi / ak; }
                    else { if (hi < LP_INF) lk = hi / ak; if (lo > -LP_INF) uk = lo / ak; }
                    if (lk > l[k]) l[k] = lk;
                    if (uk < u[k]) u[k] = uk;
                    if (uk < LP_INF && udrop && udrop[k] < u[k]) u[k] = udrop[k];       /* (a bound dropped as implied comes back, the tighter) */
                    if (l[k] > u[k] + 1e-9 * (1.0 + fabs(l[k]))) { status = 1; break; }
                    if (l[k] > u[k]) l[k] = u[k] = 0.5 * (l[k] + u[k]);
                }
                /* the operation: the row (without j) and b, for lp_recover */
                if (g_ps.nrows + lr + 1 > g_ps.rcap) { g_ps.rcap = 2 * g_ps.rcap + lr + 256; g_ps.rows = realloc(g_ps.rows, sizeof(int) * g_ps.rcap); }
                if (g_ps.vcap < g_ps.rcap) { g_ps.vcap = g_ps.rcap; g_ps.vals = realloc(g_ps.vals, sizeof(double) * g_ps.vcap); }
                const int r0 = g_ps.nrows;
                for (int q = 0; q < lr; q++) if (q != bq) { g_ps.rows[g_ps.nrows] = rc[i][q]; g_ps.vals[g_ps.nrows++] = rv[i][q]; }
                g_ps.rows[g_ps.nrows] = -1; g_ps.vals[g_ps.nrows++] = b;
                ps_push(8, j, r0, a);
                /* the cost */
                if (c[j] != 0.0) { const double f = c[j] / a; for (int q = 0; q < lr; q++) if (q != bq) c[rc[i][q]] -= f * rv[i][q]; c0 += f * b; }
                /* the other rows of the column: r <- r - (a_rj / a) row i */
                for (int q = 0; q < lr; q++) pos[rc[i][q]] = 0;
                for (int t = 0; t < cn[j]; t++) {
                    const int r = cr[j][t]; if (!ra[r] || r == i) continue;
                    int ej = -1;
                    for (int e = 0; e < rn[r]; e++) { pos[rc[r][e]] = e + 1; if (rc[r][e] == j) ej = e; }
                    if (ej >= 0) {
                        const double f = rv[r][ej] / a;
                        for (int q = 0; q < lr; q++) {
                            if (q == bq) continue;
                            const int k = rc[i][q]; const double dv = -f * rv[i][q];
                            if (pos[k]) rv[r][pos[k] - 1] += dv;
                            else {
                                if (rn[r] == rcap[r]) { rcap[r] = 2 * rcap[r] + 4; rc[r] = realloc(rc[r], sizeof(int) * rcap[r]); rv[r] = realloc(rv[r], sizeof(double) * rcap[r]); }
                                rc[r][rn[r]] = k; rv[r][rn[r]++] = dv; pos[k] = rn[r];
                                if (cn[k] == ccap[k]) { ccap[k] = 2 * ccap[k] + 4; cr[k] = realloc(cr[k], sizeof(int) * ccap[k]); }
                                cr[k][cn[k]++] = r; cl[k]++;
                            }
                        }
                        if (rl[r] > -LP_INF) rl[r] -= f * b;
                        if (ru[r] < LP_INF) ru[r] -= f * b;
                    }
                    for (int e = 0; e < rn[r]; e++) pos[rc[r][e]] = 0;
                    /* x_j and the entries that cancelled leave the row */
                    if (ej >= 0) { int w2 = 0; for (int e = 0; e < rn[r]; e++) { const int k = rc[r][e]; if (k == j) continue; if (fabs(rv[r][e]) < 1e-13) { cl[k]--; continue; } rc[r][w2] = k; rv[r][w2++] = rv[r][e]; } rn[r] = w2; }
                }
                for (int q = 0; q < lr; q++) if (q != bq) cl[rc[i][q]]--;
                ra[i] = 0; nrowrm++; ca[j] = 0; nfixed++; nelim++; did = 1;
            }
            if (!did || status || (g_lp_stopf && *g_lp_stopf)) break;
        }
        free(pos);
        if (getenv("BRISK_LPDBG")) printf("   [eliminator: %d taken; candidates refused: column too long %ld, fill %ld, lower bound not implied %ld, upper bound not implied %ld; rows of two entries: %ld refused, %ld taken (candidates refused: long %ld, small in the row %ld, a new bound on a free column %ld, small in the column %ld)]\n", nelim, dbg[0], dbg[1], dbg[2], dbg[3], dbg[4], dbg[5], d2[0], d2[1], d2[2], d2[3]);
        if (nelim) {
            /* the matrix after the substitutions, by columns */
            long nz2 = 0; for (int i = 0; i < m; i++) if (ra[i]) nz2 += rn[i];
            E_p = xz(sizeof(int) * (n + 2)); E_i = xm(sizeof(int) * (nz2 + 1)); E_x = xm(sizeof(double) * (nz2 + 1));
            for (int i = 0; i < m; i++) if (ra[i]) for (int e = 0; e < rn[i]; e++) E_p[rc[i][e] + 2]++;
            for (int j = 0; j < n; j++) E_p[j + 2] += E_p[j + 1];
            for (int i = 0; i < m; i++) if (ra[i]) for (int e = 0; e < rn[i]; e++) { const int q = E_p[rc[i][e] + 1]++; E_i[q] = i; E_x[q] = rv[i][e]; }
            A2p = E_p; A2i = E_i; A2x = E_x;
            if (verbose > 0) printf("presolve: %d columns substituted by their equality rows (the eliminator)\n", nelim);
        }
        for (int i = 0; i < m; i++) { free(rc[i]); free(rv[i]); }
        for (int j = 0; j < n; j++) free(cr[j]);
        free(rc); free(rv); free(rn); free(rcap); free(cr); free(cn); free(ccap); free(cl);
    }
    /* columns that the bound transfers fixed: their value into the rows (the matrix after the substitutions) */
    if (nelim && status == 0)
        for (int j = 0; j < n; j++) if (ca[j] && l[j] > -LP_INF && u[j] < LP_INF && u[j] - l[j] <= 1e-13 * (1.0 + fabs(l[j]))) {
            const double v = l[j];
            for (int p = A2p[j]; p < A2p[j + 1]; p++) { const int i = A2i[p]; if (!ra[i] || v == 0.0) continue; if (rl[i] > -LP_INF) rl[i] -= A2x[p] * v; if (ru[i] < LP_INF) ru[i] -= A2x[p] * v; }
            c0 += c[j] * v; ca[j] = 0; xfix[j] = v; nfixed++;
        }
    /* (27 October 2026, the night) the upper bounds dropped as implied come back on the columns the eliminator did not
     * substitute: they were dropped so that it could take those columns, and the interior-point method starts infeasible,
     * so a bound that a row implies does not hold for its iterates until that row does. neos-914441 (LP, MIPLIB 2017
     * collection): 14,884 bounds dropped, none of their columns substituted; the method's steps fell below 0.03 from the
     * fourth iteration and it stopped at the iteration limit, its objective 2.9e6 for 5.66e6; with the bounds it ends
     * optimal. A bound implied by the rows leaves the feasible set as it is. BRISK_LPREDBALL=1: all stay dropped, as before */
    int nredb_back = 0;
    if (udrop && status == 0 && !getenv("BRISK_LPREDBALL"))
        for (int j = 0; j < n; j++) if (ca[j] && udrop[j] < u[j]) { u[j] = udrop[j]; nredb_back++; }
    if (nredb_back && verbose > 0) printf("presolve: %d of the %d upper bounds dropped as implied kept (their columns not substituted)\n", nredb_back, nredb);
    /* ---- bounds far larger than the bounds the rows imply tightened to those (the interior-point method's form, at
     * every level of the presolve). The standard form shifts a column to its bound; a bound far from every feasible point (triptim1's
     * objective sum, x >= -9,999,999, while the rows keep it in [0, 204]) then gives values and an objective constant of
     * 1e7 against an optimum of 22.87, and the method stalled at a relative gap of 0.15 (22 October 2026; 26 iterations
     * with the implied bound). Each implied bound comes from one row and the bounds of its other columns as they were
     * before this step, so each follows from the rows and all hold together (unlike dropping implied bounds, where
     * x + y = 1, 0 <= x, y <= 1 lets either go but not both); no bound is added where there is none. */
    int ntight = 0;
    if (bform && status == 0 && !getenv("BRISK_LPNOTIGHT")) {
        double *amn = xz(sizeof(double) * (m + 1)), *amx = xz(sizeof(double) * (m + 1)), *asc = xz(sizeof(double) * (m + 1));
        int *imn = xz(sizeof(int) * (m + 1)), *imx = xz(sizeof(int) * (m + 1));
        for (int j = 0; j < n; j++) {
            if (!ca[j]) continue;
            for (int p = A2p[j]; p < A2p[j + 1]; p++) {
                const int i = A2i[p]; if (!ra[i]) continue;
                const double a = A2x[p], bl = a > 0 ? l[j] : u[j], bu = a > 0 ? u[j] : l[j];
                if (fabs(bl) >= LP_INF) imn[i]++; else { amn[i] += a * bl; asc[i] = fmax(asc[i], fabs(a * bl)); }
                if (fabs(bu) >= LP_INF) imx[i]++; else { amx[i] += a * bu; asc[i] = fmax(asc[i], fabs(a * bu)); }
            }
        }
        double *l2 = xm(sizeof(double) * (n + 1)), *u2 = xm(sizeof(double) * (n + 1));
        memcpy(l2, l, sizeof(double) * n); memcpy(u2, u, sizeof(double) * n);
        for (int j = 0; j < n; j++) {
            if (!ca[j] || (l[j] <= -LP_INF && u[j] >= LP_INF)) continue;
            double lo = -LP_INF, hi = LP_INF;
            for (int p = A2p[j]; p < A2p[j + 1]; p++) {
                const int i = A2i[p]; if (!ra[i]) continue;
                const double a = A2x[p], bl = a > 0 ? l[j] : u[j], bu = a > 0 ? u[j] : l[j];
                const int il = fabs(bl) >= LP_INF, iu = fabs(bu) >= LP_INF;
                /* a x_j <= ru - (the least of the others), a x_j >= rl - (the most of the others); a margin for the
                 * rounding of the sums */
                const double tol = 1e-9 * (1.0 + asc[i] + fmax(rl[i] > -LP_INF ? fabs(rl[i]) : 0, ru[i] < LP_INF ? fabs(ru[i]) : 0)) / fabs(a);
                if (ru[i] < LP_INF && imn[i] - il == 0) {
                    const double v = (ru[i] - (amn[i] - (il ? 0 : a * bl))) / a;
                    if (a > 0) hi = fmin(hi, v + tol); else lo = fmax(lo, v - tol);
                }
                if (rl[i] > -LP_INF && imx[i] - iu == 0) {
                    const double v = (rl[i] - (amx[i] - (iu ? 0 : a * bu))) / a;
                    if (a > 0) lo = fmax(lo, v - tol); else hi = fmin(hi, v + tol);
                }
            }
            /* only a bound whose size exceeds that of the implied one by three orders of magnitude: what harms is the
             * size the shift brings into the right-hand side (the method's relative accuracy then cannot resolve the
             * problem's own numbers); a tightening of a bound of the problem's own scale cuts the box instead, and on
             * greenbea (LP; lower bounds 0 -> 29 to 258, likely active at the optimum) it cost the accuracy of the
             * result on the problem as read (a row violated by 1.2e-6 of its activity, above the limit of 1e-6) */
            if (l[j] > -LP_INF && lo > l[j] && fabs(l[j]) > 1e3 * (1.0 + fabs(lo)) && lo < u[j] && lo < hi) { l2[j] = lo; ntight++;
                if (getenv("BRISK_LPDBG")) printf("   [tightened: column %d lower %.6g -> %.6g (upper %.6g)]\n", j, l[j], lo, u[j]); }
            if (u[j] < LP_INF && hi < u[j] && fabs(u[j]) > 1e3 * (1.0 + fabs(hi)) && hi > l[j] && hi > lo) { u2[j] = hi; ntight++;
                if (getenv("BRISK_LPDBG")) printf("   [tightened: column %d upper %.6g -> %.6g (lower %.6g)]\n", j, u[j], hi, l[j]); }
            if (l2[j] >= u2[j]) { l2[j] = l[j]; u2[j] = u[j]; }
        }
        memcpy(l, l2, sizeof(double) * n); memcpy(u, u2, sizeof(double) * n);
        free(l2); free(u2); free(amn); free(amx); free(asc); free(imn); free(imx);
        if (ntight && verbose > 0) printf("presolve: %d bounds far beyond the bounds the rows imply tightened to those\n", ntight);
    }
    M->status = status;
    M->ckind = xm(sizeof(int) * (n + 1)); M->cpos = xm(sizeof(int) * (n + 1)); M->cval = xz(sizeof(double) * (n + 1));
    M->rpos = xm(sizeof(int) * (m + 1)); M->rsgn = xm(sizeof(int) * (m + 1));
    for (int j = 0; j < n; j++) { M->ckind[j] = 3; M->cpos[j] = -1; M->cval[j] = xfix[j]; }
    for (int i = 0; i < m; i++) { M->rpos[i] = -1; M->rsgn[i] = 1; }
    /* (28 October 2026, the night; ISSUES 626) the linearly dependent equality rows, once after the passes (lp_dependent_rows
     * above): dropped as the redundant rows are, their duals 0. BRISK_LPDEPROWS=0: kept */
    if (status == 0 && presolve > 0 && !g_lp_nodep && !(g_lp_stopf && *g_lp_stopf) && !(getenv("BRISK_LPDEPROWS") && atoi(getenv("BRISK_LPDEPROWS")) == 0)) {
        int me = 0; for (int i = 0; i < m; i++) if (ra[i] && rcnt[i] > 0 && rl[i] == ru[i] && rl[i] > -LP_INF) me++;
        if (me >= 2) {
            int *erow = xm(sizeof(int) * (size_t)me), *ei = xm(sizeof(int) * ((size_t)m + 1)); long *ep = xz(sizeof(long) * ((size_t)me + 2));
            for (int i = 0, t = 0; i < m; i++) { ei[i] = -1; if (ra[i] && rcnt[i] > 0 && rl[i] == ru[i] && rl[i] > -LP_INF) { ei[i] = t; erow[t++] = i; } }
            for (int j = 0; j < n; j++) if (ca[j]) for (int p = A2p[j]; p < A2p[j + 1]; p++) if (ei[A2i[p]] >= 0 && A2x[p] != 0) ep[ei[A2i[p]] + 2]++;
            for (int t = 0; t < me; t++) ep[t + 2] += ep[t + 1];
            int *ej = xm(sizeof(int) * ((size_t)ep[me + 1] + 1)); double *ev = xm(sizeof(double) * ((size_t)ep[me + 1] + 1)), *eb = xm(sizeof(double) * (size_t)me);
            for (int j = 0; j < n; j++) if (ca[j]) for (int p = A2p[j]; p < A2p[j + 1]; p++) if (ei[A2i[p]] >= 0 && A2x[p] != 0) { const long q = ep[ei[A2i[p]] + 1]++; ej[q] = j; ev[q] = A2x[p]; }
            for (int t = 0; t < me; t++) eb[t] = rl[erow[t]];
            char *dep = xm((size_t)me);
            const int nd = lp_dependent_rows(me, n, ep, ej, ev, eb, dep, g_lp_stopf);
            if (nd > 0) { for (int t = 0; t < me; t++) if (dep[t]) { ra[erow[t]] = 0; nrowrm++; }
                          if (verbose > 0) printf("presolve: %d linearly dependent equality rows removed\n", nd); }
            free(erow); free(ei); free(ep); free(ej); free(ev); free(eb); free(dep);
        }
    }
    M->nrow_removed = nrowrm; M->ncol_removed = nfixed;
    if (status) {
        if (verbose > 0) printf("presolve: the problem is %s\n", status == 1 ? "primal infeasible" : "unbounded or infeasible (a column with a cost and no bound in its direction)");
        free(c); free(l); free(u); free(rl); free(ru); free(Rp); free(Rj); free(Rx); free(ra); free(ca); free(rcnt); free(ccnt); free(xfix); free(udrop);
        free(E_p); free(E_i); free(E_x);
        M->c0 = c0;
        return 0;
    }
    /* ---- the standard form: free columns first, then the nonnegative ones, slacks, bound slacks */
    int nfree = 0, nnn = 0, nbnd = 0, nslack = 0, nrng = 0, mrow = 0; long nza = 0;
    for (int j = 0; j < n; j++) if (ca[j]) {
        if (l[j] <= -LP_INF && u[j] >= LP_INF) { M->ckind[j] = 2; nfree++; }
        else if (l[j] > -LP_INF) { M->ckind[j] = 0; M->cval[j] = l[j]; nnn++; if (u[j] < LP_INF) nbnd++; }
        else { M->ckind[j] = 1; M->cval[j] = u[j]; nnn++; }
        for (int p = A2p[j]; p < A2p[j + 1]; p++) if (ra[A2i[p]]) nza++;
    }
    for (int i = 0; i < m; i++) if (ra[i]) { M->rpos[i] = mrow++; if (rl[i] != ru[i]) { nslack++; if (rl[i] > -LP_INF && ru[i] < LP_INF) nrng++; } }
    const int nbnd0 = nbnd, nrng0 = nrng;
    if (bform) { nbnd = 0; nrng = 0; }
    const int ms = mrow + nbnd + nrng, ns = nfree + nnn + nslack + nbnd + nrng;
    double *U = NULL;
    if (bform) { U = xm(sizeof(double) * (ns + 1)); for (int j = 0; j < ns; j++) U[j] = LP_INF; *ub = U; }
    S->m = ms; S->n = ns; S->nf = nfree; S->nl = ns - nfree;
    S->Ap = xm(sizeof(int) * (ns + 1)); S->Ai = xm(sizeof(int) * (nza + nslack + 2 * nbnd + 2 * nrng + 1)); S->Ax = xm(sizeof(double) * (nza + nslack + 2 * nbnd + 2 * nrng + 1));
    S->b = xz(sizeof(double) * (ms + 1)); S->c = xz(sizeof(double) * (ns + 1));
    /* right-hand sides: an equality b = rl; a >= row: a'x - s = rl; a <= row: a'x + s = ru; a range: a'x - s = rl, s + t = ru - rl */
    for (int i = 0; i < m; i++) if (ra[i]) S->b[M->rpos[i]] = rl[i] > -LP_INF ? rl[i] : ru[i];
    int col = 0, brow = mrow; long w = 0;
    int *bndrow = xm(sizeof(int) * (n + 1));
    for (int pass = 0; pass < 2; pass++)                     /* free columns, then the others */
        for (int j = 0; j < n; j++) {
            if (!ca[j] || (pass == 0) != (M->ckind[j] == 2)) continue;
            M->cpos[j] = col; S->Ap[col] = (int)w;
            const double sg = M->ckind[j] == 1 ? -1.0 : 1.0, sh = M->ckind[j] == 2 ? 0.0 : M->cval[j];
            S->c[col] = sg * c[j]; c0 += c[j] * sh;
            for (int p = A2p[j]; p < A2p[j + 1]; p++) {
                const int i = A2i[p]; if (!ra[i]) continue;
                S->Ai[w] = M->rpos[i]; S->Ax[w] = sg * A2x[p]; w++;
                S->b[M->rpos[i]] -= A2x[p] * sh;
            }
            bndrow[j] = -1;
            if (bform) { if (M->ckind[j] == 0 && u[j] < LP_INF) U[col] = u[j] - l[j]; }
            else if (M->ckind[j] == 0 && u[j] < LP_INF) { bndrow[j] = brow; S->Ai[w] = brow; S->Ax[w] = 1.0; w++; S->b[brow] = u[j] - l[j]; brow++; }
            col++;
        }
    /* row slacks */
    for (int i = 0; i < m; i++) if (ra[i] && rl[i] != ru[i]) {
        S->Ap[col] = (int)w; S->Ai[w] = M->rpos[i]; S->Ax[w] = rl[i] > -LP_INF ? -1.0 : 1.0; w++;
        if (bform) { if (rl[i] > -LP_INF && ru[i] < LP_INF) U[col] = ru[i] - rl[i]; }
        else if (rl[i] > -LP_INF && ru[i] < LP_INF) { S->Ai[w] = brow; S->Ax[w] = 1.0; w++; S->b[brow] = ru[i] - rl[i]; brow++; }
        col++;
    }
    /* the slacks of the bound rows: columns in the order of their rows */
    for (int r = mrow; r < ms; r++) { S->Ap[col] = (int)w; S->Ai[w] = r; S->Ax[w] = 1.0; w++; col++; }
    S->Ap[col] = (int)w;
    M->c0 = c0; M->nnz_removed = (long)L->Ap[n] - nza;
    if (verbose > 0 && presolve > 0)
        printf("presolve: %d of %d rows and %d of %d columns removed (%ld of %d nonzeros); standard form: %d rows, %d columns (%d free), %ld nonzeros; %d %s\n",
               nrowrm, m, nfixed, n, M->nnz_removed, L->Ap[n], ms, ns, nfree, w, nbnd0 + nrng0, bform ? "upper bounds" : "bound rows");
    free(bndrow); free(E_p); free(E_i); free(E_x);
    free(c); free(l); free(u); free(rl); free(ru); free(Rp); free(Rj); free(Rx); free(ra); free(ca); free(rcnt); free(ccnt); free(xfix); free(udrop);
    return 0;
}

double lp_recover(const LpProb *L, const LpMap *M, const double *xstd, double *x) {
    const int n = L->n;
    for (int j = 0; j < n; j++) {
        if (M->ckind[j] == 3) x[j] = M->cval[j];
        else if (M->ckind[j] == 2) x[j] = xstd[M->cpos[j]];
        else if (M->ckind[j] == 0) x[j] = M->cval[j] + xstd[M->cpos[j]];
        else x[j] = M->cval[j] - xstd[M->cpos[j]];
    }
    /* the operations of the presolve that left a value open, in reverse */
    int any = 0;
    for (int e = 0; e < g_ps.nop; e++) if (g_ps.op[e].kind >= 1) any = 1;
    if (any) {
        const int m = L->m;
        int *Rp = xz(sizeof(int) * (m + 2)), *Rj = xm(sizeof(int) * (L->Ap[n] + 1)); double *Rx = xm(sizeof(double) * (L->Ap[n] + 1));
        for (int p = 0; p < L->Ap[n]; p++) Rp[L->Ai[p] + 2]++;
        for (int i = 0; i < m; i++) Rp[i + 2] += Rp[i + 1];
        for (int j = 0; j < n; j++) for (int p = L->Ap[j]; p < L->Ap[j + 1]; p++) { const int q = Rp[L->Ai[p] + 1]++; Rj[q] = j; Rx[q] = L->Ax[p]; }
        for (int e = g_ps.nop - 1; e >= 0; e--) {
            const int kind = g_ps.op[e].kind, j = g_ps.op[e].j;
            if (kind == 1) {
                /* a free column singleton: x_j = (b_i - sum_{k != j} a_ik x_k) / a_ij from the row as read */
                const int i = g_ps.op[e].i;
                double s = 0;
                for (int q = Rp[i]; q < Rp[i + 1]; q++) if (Rj[q] != j) s += Rx[q] * x[Rj[q]];
                /* the row's target: an equality, or the nearest point of its range to the rest (zero cost) */
                double t = L->rl[i] == L->ru[i] ? L->rl[i] : (L->rl[i] > -LP_INF ? L->rl[i] : L->ru[i]);
                if (L->rl[i] != L->ru[i]) { const double lo = L->rl[i] > -LP_INF ? L->rl[i] : -1e300, hi = L->ru[i] < LP_INF ? L->ru[i] : 1e300; t = s < lo ? lo : s > hi ? hi : s; }
                x[j] = (t - s) / g_ps.op[e].v;
            } else if (kind >= 2 && kind <= 5) {
                /* a free variable that was a pair of opposite columns: its two parts */
                const int k = g_ps.op[e].i, kd = kind - 2;
                const double dj = (kd & 1) ? -1.0 : 1.0, dk = (kd & 2) ? -1.0 : 1.0, bj = g_ps.op[e].v;
                const double t = dj * (x[j] - bj);                 /* p - q; x_k is at its bound b_k */
                if (t < 0) { x[j] = bj; x[k] += dk * (-t); }
            } else if (kind == 8) {
                /* a column substituted by its equality row: x_j = (b - sum a_k x_k) / a_j */
                int r = g_ps.op[e].i; double sacc = 0;
                for (; g_ps.rows[r] >= 0; r++) sacc += g_ps.vals[r] * x[g_ps.rows[r]];
                x[j] = (g_ps.vals[r] - sacc) / g_ps.op[e].v;
            } else if (kind == 6 || kind == 7) {
                /* a column without cost that only helps its rows in one direction: as far as they need */
                double v = g_ps.op[e].v;
                for (int r = g_ps.op[e].i; g_ps.rows[r] >= 0; r++) {
                    const int i = g_ps.rows[r];
                    double s = 0, a = 0;
                    for (int q = Rp[i]; q < Rp[i + 1]; q++) { if (Rj[q] != j) s += Rx[q] * x[Rj[q]]; else a = Rx[q]; }
                    if (a == 0.0) continue;
                    /* the side of the row that x_j repairs: rl when a x_j grows in the direction of the move */
                    const int grows = (a > 0) == (kind == 6);
                    if (grows && L->rl[i] > -LP_INF) { const double need = (L->rl[i] - s) / a; if (kind == 6 ? need > v : need < v) v = need; }
                    if (!grows && L->ru[i] < LP_INF) { const double need = (L->ru[i] - s) / a; if (kind == 6 ? need > v : need < v) v = need; }
                }
                x[j] = v;
            }
        }
        free(Rp); free(Rj); free(Rx);
    }
    double obj = L->c0;
    for (int j = 0; j < n; j++) obj += L->c[j] * x[j];
    return obj;
}

void lp_errors(const LpProb *L, const double *x, double *ebound, double *erow) {
    const int m = L->m, n = L->n;
    double eb = 0, er = 0, xn = 0;
    double *ax = xz(sizeof(double) * (m + 1)), *an = xz(sizeof(double) * (m + 1));
    for (int j = 0; j < n; j++) {
        xn = fmax(xn, fabs(x[j]));
        if (L->l[j] > -LP_INF) eb = fmax(eb, (L->l[j] - x[j]) / (1.0 + fabs(L->l[j])));
        if (L->u[j] < LP_INF) eb = fmax(eb, (x[j] - L->u[j]) / (1.0 + fabs(L->u[j])));
        for (int p = L->Ap[j]; p < L->Ap[j + 1]; p++) { ax[L->Ai[p]] += L->Ax[p] * x[j]; an[L->Ai[p]] += fabs(L->Ax[p] * x[j]); }
    }
    int iw = -1;
    for (int i = 0; i < m; i++) {
        const double e0 = er;
        if (L->rl[i] > -LP_INF) er = fmax(er, (L->rl[i] - ax[i]) / (1.0 + fabs(L->rl[i])));
        if (L->ru[i] < LP_INF) er = fmax(er, (ax[i] - L->ru[i]) / (1.0 + fabs(L->ru[i])));
        if (er > e0) iw = i;
    }
    if (iw >= 0 && er > 1e-6 && getenv("BRISK_LPDBG")) {     /* (debugging: the worst row and its entries) */
        printf("   [worst row %d: [%.6g, %.6g], activity %.6g; entries:", iw, L->rl[iw], L->ru[iw], ax[iw]);
        int c = 0;
        for (int j = 0; j < n && c < 12; j++) for (int p = L->Ap[j]; p < L->Ap[j + 1]; p++) if (L->Ai[p] == iw) { printf(" %d:%.3g*%.6g[%.3g,%.3g]", j, L->Ax[p], x[j], L->l[j], L->u[j]); c++; }
        printf("]\n");
    }
    free(ax); free(an);
    *ebound = eb > 0 ? eb : 0; *erow = er > 0 ? er : 0;
}
/* (21 October 2026, the night) the largest row violation against the row's activity, (violation) / (1 + |side| +
 * sum_j |a_ij x_j|): unlike lp_errors' measure it does not grow with the terms of a row (agg, pilot.we: 1.6e-10 and
 * 3.2e-10 at points lp_errors puts at 3.0e-5 and 1.4e-6), so a point off by more than the solve's tolerance shows */
double lp_rowviol_act(const LpProb *L, const double *x) {
    const int m = L->m, n = L->n;
    double *ax = xz(sizeof(double) * (m + 1)), *an = xz(sizeof(double) * (m + 1)), ea = 0;
    for (int j = 0; j < n; j++) for (int p = L->Ap[j]; p < L->Ap[j + 1]; p++) { ax[L->Ai[p]] += L->Ax[p] * x[j]; an[L->Ai[p]] += fabs(L->Ax[p] * x[j]); }
    for (int i = 0; i < m; i++) {
        if (L->rl[i] > -LP_INF) ea = fmax(ea, (L->rl[i] - ax[i]) / (1.0 + fabs(L->rl[i]) + an[i]));
        if (L->ru[i] < LP_INF) ea = fmax(ea, (ax[i] - L->ru[i]) / (1.0 + fabs(L->ru[i]) + an[i]));
    }
    free(ax); free(an);
    return ea;
}

/* ---- CBF (conic benchmark format, CBLIB): continuous problems with linear and second-order cones ----
 *     min / max  c'x + c0,   x in K_var,   A x + b in K_con
 * with cones F (free), L+, L-, L= (zero), Q (x0 >= |x(1:)|), QR (2 x0 x1 >= |x(2:)|^2). Brought to
 * the SeDuMi form: L- variables are negated, L= variables dropped, a constraint in L= is an
 * equality row A x = -b, one in L+ / L- / Q / QR gets slack variables s = A x + b in its cone, one
 * in F is dropped. Integer variables (INT) are relaxed, with a message; PSD, exponential and power
 * cones are refused. A file name ending in .gz is read through gzip. */
int cbf_is_file(const char *fname) {
    const size_t l = strlen(fname);
    return (l > 4 && !strcmp(fname + l - 4, ".cbf")) || (l > 7 && !strcmp(fname + l - 7, ".cbf.gz"));
}
/* (23 October 2026, the SOCP developer) numbers of the CBF's coordinate lists without sscanf or strtod where it is exact:
 * an integer by its digits; a decimal number with at most 19 significant digits, a mantissa M <= 2^53 and a decimal
 * exponent e with |e| <= 22 is M * 10^e or M / 10^-e, one correctly rounded operation on two exact operands (Clinger's
 * fast path), so the same double as strtod's; any other form goes to strtod. 2013_firL1 (SOCP): 40 million lines. */
static long fast_long(const char *p, char **end) {
    while (*p == ' ' || *p == '\t') p++;
    const char *st = p;
    int neg = 0; if (*p == '-' || *p == '+') { neg = *p == '-'; p++; }
    if (*p < '0' || *p > '9') return strtol(st, end, 10);
    long v = 0; int nd = 0;
    while (*p >= '0' && *p <= '9') { v = 10 * v + (*p - '0'); p++; if (++nd > 18) return strtol(st, end, 10); }
    *end = (char *)p; return neg ? -v : v;
}
static int cone_code(const char *s) {
    if (!strcmp(s, "F")) return 0;
    if (!strcmp(s, "L+")) return 1;
    if (!strcmp(s, "L-")) return 2;
    if (!strcmp(s, "L=")) return 3;
    if (!strcmp(s, "Q")) return 4;
    if (!strcmp(s, "QR")) return 5;
    return -1;
}
/* (22 October 2026, the SOCP developer) cbf_read_ex: also the SeDuMi columns of the integer variables (intcol, -1 for a
 * variable without a column) and their signs (intsgn: -1 for an L- variable, negated in the SeDuMi form), for a
 * branch and bound over the relaxation; both NULL: as cbf_read */
int cbf_read_ex(const char *fname, SedumiProb *S, double *c0_out, int *maxim_out, int *nint_out, int **intcol, signed char **intsgn, char *msg, size_t nmsg);
int cbf_read(const char *fname, SedumiProb *S, double *c0_out, int *maxim_out, int *nint_out, char *msg, size_t nmsg) {
    return cbf_read_ex(fname, S, c0_out, maxim_out, nint_out, NULL, NULL, msg, nmsg);
}
static int cbf_read_core(const char *fname, SedumiProb *S, double *c0_out, int *maxim_out, int *nint_out, int **intcol, signed char **intsgn, int **vmap, signed char **vmsgn, int *nvar_out, char *msg, size_t nmsg);
int cbf_read_ex(const char *fname, SedumiProb *S, double *c0_out, int *maxim_out, int *nint_out, int **intcol, signed char **intsgn, char *msg, size_t nmsg) {
    return cbf_read_core(fname, S, c0_out, maxim_out, nint_out, intcol, intsgn, NULL, NULL, NULL, msg, nmsg);
}
/* (27 October 2026, the SOCP developer) cbf_read_vmap: also the SeDuMi column of every CBF variable (vmap, -1 for an L=
 * variable, which is zero) and its sign (vmsgn: -1 for an L- variable), so that a solution can be written in the file's
 * own variables (the command line's -x) */
int cbf_read_vmap(const char *fname, SedumiProb *S, double *c0_out, int *maxim_out, int *nint_out, int **vmap, signed char **vmsgn, int *nvar_out, char *msg, size_t nmsg) {
    return cbf_read_core(fname, S, c0_out, maxim_out, nint_out, NULL, NULL, vmap, vmsgn, nvar_out, msg, nmsg);
}
static int cbf_read_core(const char *fname, SedumiProb *S, double *c0_out, int *maxim_out, int *nint_out, int **intcol, signed char **intsgn, int **vmap, signed char **vmsgn, int *nvar_out, char *msg, size_t nmsg) {
    memset(S, 0, sizeof(*S)); *c0_out = 0; *maxim_out = 0; *nint_out = 0;
    int *intv = NULL; int nintv = 0;
    if (intcol) *intcol = NULL;
    if (intsgn) *intsgn = NULL;
    if (vmap) *vmap = NULL;
    if (vmsgn) *vmsgn = NULL;
    if (nvar_out) *nvar_out = 0;
    long gzpid; FILE *f = lp_open_in(fname, &gzpid);
    if (!f) { snprintf(msg, nmsg, "cannot open %s", fname); return 1; }
    int nvar = 0, ncon = 0, nvg = 0, ncg = 0;
    int *vgk = NULL, *vgd = NULL, *cgk = NULL, *cgd = NULL;       /* cone groups: kind, dimension */
    long nza = 0; int *ai = NULL, *aj = NULL; double *av = NULL;
    double *cobj = NULL, *bcon = NULL, c0 = 0;
    char *line = NULL; size_t lcap = 0; int rc = 0;
    char key[64];
#define NEXTLINE() (getline(&line, &lcap, f) >= 0)
    /* a counted section or a keyword's value line that the file does not hold is an error (24 October 2026: a CBLIB file
     * joined from its parts while the disk was full was read cut short and solved as another problem, "dual infeasible") */
#define SKIPLINE() do { if (getline(&line, &lcap, f) < 0) { line[0] = 0; if (!rc) { snprintf(msg, nmsg, "%s: the file ends inside section %s (cut short?)", fname, key); rc = 1; } } } while (0)
    while (!rc && NEXTLINE()) {
        if (line[0] == '#' || line[0] == '\n' || line[0] == '\r') continue;
        if (sscanf(line, "%63s", key) != 1) continue;
        if (!strcmp(key, "VER")) { SKIPLINE(); }
        else if (!strcmp(key, "OBJSENSE")) { SKIPLINE(); if (sscanf(line, "%63s", key) == 1 && !strcmp(key, "MAX")) *maxim_out = 1; }
        else if (!strcmp(key, "VAR") || !strcmp(key, "CON")) {
            const int isv = key[0] == 'V';
            int n = 0, ng = 0;
            if (!NEXTLINE() || sscanf(line, "%d %d", &n, &ng) != 2) { snprintf(msg, nmsg, "%s: bad %s header", fname, key); rc = 1; break; }
            int *gk = xm(sizeof(int) * (ng + 1)), *gd = xm(sizeof(int) * (ng + 1)), tot = 0;
            for (int g = 0; g < ng && !rc; g++) {
                char cn[64]; int d = 0;
                if (!NEXTLINE() || sscanf(line, "%63s %d", cn, &d) != 2) { snprintf(msg, nmsg, "%s: bad cone line in %s", fname, key); rc = 1; break; }
                gk[g] = cone_code(cn); gd[g] = d; tot += d;
                if (gk[g] < 0) { snprintf(msg, nmsg, "%s: cone %s is not supported (linear and second-order cones only)", fname, cn); rc = 1; }
            }
            if (!rc && tot != n) { snprintf(msg, nmsg, "%s: the cones of %s do not add up", fname, key); rc = 1; }
            if (isv) { nvar = n; nvg = ng; vgk = gk; vgd = gd; cobj = xz(sizeof(double) * (n + 1)); }
            else { ncon = n; ncg = ng; cgk = gk; cgd = gd; bcon = xz(sizeof(double) * (n + 1)); }
        }
        else if (!strcmp(key, "INT")) {
            int k = 0; SKIPLINE(); sscanf(line, "%d", &k); *nint_out = k;
            if (intcol) { free(intv); intv = xm(sizeof(int) * (k + 1)); nintv = 0; }
            for (int t = 0; t < k && !rc; t++) { SKIPLINE(); int j; if (intv && sscanf(line, "%d", &j) == 1) intv[nintv++] = j; }
        }
        else if (!strcmp(key, "OBJACOORD")) {
            long k = 0; SKIPLINE(); sscanf(line, "%ld", &k);
            for (long t = 0; t < k && !rc; t++) { SKIPLINE(); char *e1 = line, *e2 = line; const long j = fast_long(line, &e1); const double v = fast_strtod(e1, &e2); if (e1 != line && e2 != e1 && cobj && j >= 0 && j < nvar) cobj[j] += v; }
        }
        else if (!strcmp(key, "OBJBCOORD")) { SKIPLINE(); sscanf(line, "%lf", &c0); }
        else if (!strcmp(key, "ACOORD")) {
            long k = 0; SKIPLINE(); sscanf(line, "%ld", &k);
            ai = xm(sizeof(int) * (k + 1)); aj = xm(sizeof(int) * (k + 1)); av = xm(sizeof(double) * (k + 1));
            for (long t = 0; t < k && !rc; t++) {
                SKIPLINE(); if (rc) break;
                char *e; const long i = fast_long(line, &e); const long j = fast_long(e, &e); const double v = fast_strtod(e, NULL);
                if (i < 0 || i >= ncon || j < 0 || j >= nvar) { snprintf(msg, nmsg, "%s: ACOORD index out of range", fname); rc = 1; break; }
                ai[nza] = (int)i; aj[nza] = (int)j; av[nza] = v; nza++;
            }
        }
        else if (!strcmp(key, "BCOORD")) {
            long k = 0; SKIPLINE(); sscanf(line, "%ld", &k);
            for (long t = 0; t < k && !rc; t++) { SKIPLINE(); char *e1 = line, *e2 = line; const long i = fast_long(line, &e1); const double v = fast_strtod(e1, &e2); if (e1 != line && e2 != e1 && bcon && i >= 0 && i < ncon) bcon[i] += v; }
        }
        else if (!strcmp(key, "PSDVAR") || !strcmp(key, "PSDCON") || !strcmp(key, "FCOORD") || !strcmp(key, "HCOORD") || !strcmp(key, "DCOORD") || !strcmp(key, "OBJFCOORD")
                 || !strcmp(key, "POWCONES") || !strcmp(key, "POW*CONES")) {
            snprintf(msg, nmsg, "%s: section %s: semidefinite and power cones are not read from CBF files", fname, key); rc = 1;
        }
        /* other keywords (CHANGE etc.): ignored */
    }
#undef NEXTLINE
#undef SKIPLINE
    free(line);
    /* gzip's exit status: a damaged or truncated archive is an error, not the part that could be read */
    if (lp_close_in(f, gzpid) && !rc) { snprintf(msg, nmsg, "%s: the decompression failed (gzip -dc: the file is damaged or cut short)", fname); rc = 1; }
    if (!rc && nvar == 0) { snprintf(msg, nmsg, "%s: no variables (not a CBF file?)", fname); rc = 1; }
    if (!rc) {
        /* columns of the SeDuMi form: [free | nonnegative | Q cones | QR cones]; var j -> column vcol[j] (or -1), sign vsg[j] */
        int *vk = xm(sizeof(int) * (nvar + 1)), *ck = xm(sizeof(int) * (ncon + 1));
        { int p = 0; for (int g = 0; g < nvg; g++) for (int t = 0; t < vgd[g]; t++) vk[p++] = vgk[g]; }
        { int p = 0; for (int g = 0; g < ncg; g++) for (int t = 0; t < cgd[g]; t++) ck[p++] = cgk[g]; }
        int nf = 0, nl = 0, nq = 0, nr = 0, qd = 0, rd = 0;
        for (int j = 0; j < nvar; j++) { if (vk[j] == 0) nf++; else if (vk[j] == 1 || vk[j] == 2) nl++; }
        for (int i = 0; i < ncon; i++) if (ck[i] == 1 || ck[i] == 2) nl++;
        for (int g = 0; g < nvg; g++) { if (vgk[g] == 4) { nq++; qd += vgd[g]; } else if (vgk[g] == 5) { nr++; rd += vgd[g]; } }
        for (int g = 0; g < ncg; g++) { if (cgk[g] == 4) { nq++; qd += cgd[g]; } else if (cgk[g] == 5) { nr++; rd += cgd[g]; } }
        const int n = nf + nl + qd + rd;
        int *vcol = xm(sizeof(int) * (nvar + 1)), *ccol = xm(sizeof(int) * (ncon + 1)), *crow = xm(sizeof(int) * (ncon + 1));
        int pf = 0, pl = nf, pq = nf + nl, pr = nf + nl + qd, m = 0;
        S->q = xm(sizeof(int) * (nq + 1)); S->r = xm(sizeof(int) * (nr + 1)); S->nq = 0; S->nr = 0;
        for (int j = 0; j < nvar; j++) vcol[j] = vk[j] == 0 ? pf++ : (vk[j] == 1 || vk[j] == 2) ? pl++ : -1;
        for (int i = 0; i < ncon; i++) { ccol[i] = (ck[i] == 1 || ck[i] == 2) ? pl++ : -1; crow[i] = ck[i] == 0 ? -1 : m++; }
        { int p = 0; for (int g = 0; g < nvg; g++) { if (vgk[g] == 4) { S->q[S->nq++] = vgd[g]; for (int t = 0; t < vgd[g]; t++) vcol[p + t] = pq++; } p += vgd[g]; } }
        { int p = 0; for (int g = 0; g < ncg; g++) { if (cgk[g] == 4) { S->q[S->nq++] = cgd[g]; for (int t = 0; t < cgd[g]; t++) ccol[p + t] = pq++; } p += cgd[g]; } }
        { int p = 0; for (int g = 0; g < nvg; g++) { if (vgk[g] == 5) { S->r[S->nr++] = vgd[g]; for (int t = 0; t < vgd[g]; t++) vcol[p + t] = pr++; } p += vgd[g]; } }
        { int p = 0; for (int g = 0; g < ncg; g++) { if (cgk[g] == 5) { S->r[S->nr++] = cgd[g]; for (int t = 0; t < cgd[g]; t++) ccol[p + t] = pr++; } p += cgd[g]; } }
        S->m = m; S->n = n; S->nf = nf; S->nl = nl; S->ns = 0;
        S->b = xz(sizeof(double) * (m + 1)); S->c = xz(sizeof(double) * (n + 1));
        for (int i = 0; i < ncon; i++) if (crow[i] >= 0) S->b[crow[i]] = -bcon[i];
        for (int j = 0; j < nvar; j++) if (vcol[j] >= 0) S->c[vcol[j]] = vk[j] == 2 ? -cobj[j] : cobj[j];
        /* A by columns: count, fill (the slack of constraint i: -1 for L+ and the cones (A x - s = -b), +1 for L-) */
        int *cnt = xz(sizeof(int) * (n + 2));
        for (long t = 0; t < nza; t++) if (crow[ai[t]] >= 0 && vcol[aj[t]] >= 0 && av[t] != 0.0) cnt[vcol[aj[t]] + 1]++;
        for (int i = 0; i < ncon; i++) if (ccol[i] >= 0) cnt[ccol[i] + 1]++;
        S->Ap = xm(sizeof(int) * (n + 1)); S->Ap[0] = 0;
        for (int j = 0; j < n; j++) S->Ap[j + 1] = S->Ap[j] + cnt[j + 1];
        S->Ai = xm(sizeof(int) * (S->Ap[n] + 1)); S->Ax = xm(sizeof(double) * (S->Ap[n] + 1));
        int *nx = xm(sizeof(int) * (n + 1)); memcpy(nx, S->Ap, sizeof(int) * n);
        for (long t = 0; t < nza; t++) if (crow[ai[t]] >= 0 && vcol[aj[t]] >= 0 && av[t] != 0.0) {
            const int col = vcol[aj[t]]; S->Ai[nx[col]] = crow[ai[t]]; S->Ax[nx[col]] = vk[aj[t]] == 2 ? -av[t] : av[t]; nx[col]++;
        }
        for (int i = 0; i < ncon; i++) if (ccol[i] >= 0) { const int col = ccol[i]; S->Ai[nx[col]] = crow[i]; S->Ax[nx[col]] = ck[i] == 2 ? 1.0 : -1.0; nx[col]++; }
        /* rows within a column sorted and duplicates summed (ACOORD may repeat an entry) */
        for (int j = 0; j < n; j++) {
            const int a = S->Ap[j], e = S->Ap[j + 1];
            for (int p = a + 1; p < e; p++) { const int r = S->Ai[p]; const double v = S->Ax[p]; int q = p - 1; while (q >= a && S->Ai[q] > r) { S->Ai[q + 1] = S->Ai[q]; S->Ax[q + 1] = S->Ax[q]; q--; } S->Ai[q + 1] = r; S->Ax[q + 1] = v; }
        }
        { int w = 0, a = 0;
          for (int j = 0; j < n; j++) {
              const int e = S->Ap[j + 1]; S->Ap[j] = w;
              for (int p = a; p < e; p++) { if (w > S->Ap[j] && S->Ai[w - 1] == S->Ai[p]) S->Ax[w - 1] += S->Ax[p]; else { S->Ai[w] = S->Ai[p]; S->Ax[w] = S->Ax[p]; w++; } }
              a = e;
          }
          S->Ap[n] = w; }
        if (intcol && intv) {
            *intcol = xm(sizeof(int) * (nintv + 1)); if (intsgn) *intsgn = xm(nintv + 1);
            for (int t = 0; t < nintv; t++) {
                const int j = intv[t];
                (*intcol)[t] = (j >= 0 && j < nvar) ? vcol[j] : -1;
                if (intsgn) (*intsgn)[t] = (j >= 0 && j < nvar && vk[j] == 2) ? -1 : 1;
            }
            *nint_out = nintv;
        }
        if (vmap) {
            *vmap = vcol; vcol = NULL; *nvar_out = nvar;
            if (vmsgn) { *vmsgn = xm(nvar + 1); for (int j = 0; j < nvar; j++) (*vmsgn)[j] = vk[j] == 2 ? -1 : 1; }
        }
        free(cnt); free(nx); free(vk); free(ck); free(vcol); free(ccol); free(crow);
        *c0_out = c0;
    }
    free(vgk); free(vgd); free(cgk); free(cgd); free(ai); free(aj); free(av); free(cobj); free(bcon); free(intv);
    return rc;
}
