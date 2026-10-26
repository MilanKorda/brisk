/* reader_test: the MPS, QPS and CBF readers refuse a file that was cut short (a file ending before ENDATA, a CBF section
 * with fewer entries than it announces, a gzip archive without its end) and read the complete files. (The SOCP developer,
 * 24 October 2026: a CBLIB file joined from its parts while the disk was full was read cut short and solved as another
 * problem.) */
#include "../lpio.h"
#include "../sedumi.h"
#include "../qp.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int fails = 0;
#define CHECK(c, ...) do { if (!(c)) { fails++; printf("  FAIL " __VA_ARGS__); printf("\n"); } else { printf("  ok   " __VA_ARGS__); printf("\n"); } } while (0)

static char dir[256];
static const char *path(const char *name) { static char p[512]; snprintf(p, sizeof p, "%s/%s", dir, name); return p; }
static void put(const char *name, const char *text) { FILE *f = fopen(path(name), "w"); if (f) { fputs(text, f); fclose(f); } }

static const char *MPS = "NAME T\nROWS\n N obj\n L c1\n L c2\nCOLUMNS\n x obj 1 c1 1\n x c2 1\n y obj 2 c1 1\nRHS\n rhs c1 4 c2 3\nBOUNDS\n UP bnd y 5\n";
/* min t over (t, u) in a second-order cone with u + v = 2, v = 0 (two rows): the optimum 2 */
static const char *CBF = "VER\n3\n\nOBJSENSE\nMIN\n\nVAR\n3 2\nQ 2\nF 1\n\nCON\n2 1\nL= 2\n\nOBJACOORD\n1\n0 1\n\nACOORD\n3\n0 1 1\n0 2 1\n1 2 1\n\nBCOORD\n1\n0 -2\n";

int main(void) {
    const char *t = getenv("TMPDIR");
    snprintf(dir, sizeof dir, "%s/brisk_reader_XXXXXX", t && *t ? t : "/tmp");
    if (!mkdtemp(dir)) { printf("reader_test: no temporary directory\n"); return 1; }
    char msg[512], buf[2048];
    printf("1. MPS (the LP reader) and QPS (the QP reader): ENDATA\n");
    snprintf(buf, sizeof buf, "%sENDATA\n", MPS); put("full.mps", buf); put("cut.mps", MPS);
    { LpProb L; int rc = lp_read_mps(path("full.mps"), &L, msg, sizeof msg); CHECK(rc == 0, "the complete MPS file is read"); if (rc == 0) lp_free(&L); }
    { LpProb L; int rc = lp_read_mps(path("cut.mps"), &L, msg, sizeof msg); CHECK(rc != 0 && strstr(msg, "cut short"), "the MPS file without ENDATA is refused (%s)", rc ? msg : "read"); if (rc == 0) lp_free(&L); }
    put("full.qps", buf); put("cut.qps", MPS);
    { QpProb P; int rc = qp_read_qps(path("full.qps"), &P, msg, sizeof msg); CHECK(rc == 0, "the complete QPS file is read"); if (rc == 0) qp_prob_free(&P); }
    { QpProb P; int rc = qp_read_qps(path("cut.qps"), &P, msg, sizeof msg); CHECK(rc != 0 && strstr(msg, "cut short"), "the QPS file without ENDATA is refused (%s)", rc ? msg : "read"); if (rc == 0) qp_prob_free(&P); }
    printf("2. CBF (the SOCP reader): the sections' counts and the gzip archive\n");
    put("full.cbf", CBF);
    { char cut[2048]; snprintf(cut, sizeof cut, "%s", CBF); *strstr(cut, "0 2 1\n") = 0; put("cut.cbf", cut); }
    SedumiProb S; double c0; int maxim, nint;
    { int rc = cbf_read(path("full.cbf"), &S, &c0, &maxim, &nint, msg, sizeof msg); CHECK(rc == 0, "the complete CBF file is read"); if (rc == 0) sedumi_free(&S); }
    { int rc = cbf_read(path("cut.cbf"), &S, &c0, &maxim, &nint, msg, sizeof msg); CHECK(rc != 0 && strstr(msg, "cut short"), "the CBF file cut inside ACOORD is refused (%s)", rc ? msg : "read"); if (rc == 0) sedumi_free(&S); }
    { char src[512]; snprintf(src, sizeof src, "%s", path("full.cbf")); snprintf(buf, sizeof buf, "gzip -c '%s' > '%s' 2>/dev/null", src, path("full.cbf.gz")); }
    if (system(buf) == 0) {
        { int rc = cbf_read(path("full.cbf.gz"), &S, &c0, &maxim, &nint, msg, sizeof msg); CHECK(rc == 0, "the complete gzip archive is read (%s)", rc ? msg : "read"); if (rc == 0) sedumi_free(&S); }
        FILE *f = fopen(path("full.cbf.gz"), "rb"); char gz[4096]; size_t ng = f ? fread(gz, 1, sizeof gz, f) : 0; if (f) fclose(f);
        f = fopen(path("cut.cbf.gz"), "wb"); if (f) { fwrite(gz, 1, ng > 4 ? ng - 4 : 0, f); fclose(f); }       /* its last four bytes (the length) cut */
        { int rc = cbf_read(path("cut.cbf.gz"), &S, &c0, &maxim, &nint, msg, sizeof msg); CHECK(rc != 0, "the archive without its end is refused (%s)", rc ? msg : "read"); if (rc == 0) sedumi_free(&S); }
    } else printf("  (no gzip on this host: the archive checks skipped)\n");
    const char *names[] = {"full.mps", "cut.mps", "full.qps", "cut.qps", "full.cbf", "cut.cbf", "full.cbf.gz", "cut.cbf.gz"};
    for (size_t k = 0; k < sizeof names / sizeof *names; k++) remove(path(names[k]));
    remove(dir);
    printf(fails ? "reader_test: %d failures\n" : "reader_test: all checks passed\n", fails);
    return fails != 0;
}
