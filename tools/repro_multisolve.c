/* repro_multisolve (tools/repro_suite.sh; a user's driver of 27 October 2026, ISSUES 708): several solves in one process through the library entry brisk_run, as the MEX file makes them.
 * Solves are separated by "--"; ENV=NAME=VALUE sets a variable before that solve (VALUE "-" unsets it). Prints
 * each solve's status, iterations and a hash of y. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include "brisk.h"
int brisk_run(int argc, char **argv, BriskResult *res);
int main(int argc, char **argv) {
    int i = 1, k = 0;
    while (i < argc) {
        char *av[64]; int ac = 0; av[ac++] = "brisk";
        for (; i < argc && strcmp(argv[i], "--"); i++) {
            if (!strncmp(argv[i], "ENV=", 4)) { char *e = strdup(argv[i] + 4), *eq = strchr(e, '='); *eq = 0; if (!strcmp(eq + 1, "-")) unsetenv(e); else setenv(e, eq + 1, 1); continue; }
            av[ac++] = argv[i];
        }
        i++; av[ac] = NULL;
        BriskResult r; brisk_run(ac, av, &r);
        uint64_t h = 1469598103934665603ULL; const unsigned char *c = (const void *)r.y;
        for (size_t q = 0; r.y && q < sizeof(double) * (size_t)r.m; q++) { h ^= c[q]; h *= 1099511628211ULL; }
        printf("%2d %s: %s, %d its, y hash %016llx\n", ++k, av[1], r.status_str, r.iters, (unsigned long long)h);
        fflush(stdout);
    }
    return 0;
}
