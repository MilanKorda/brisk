/* capi_bound_test.c: the C API with -bound pd three times on one handle, then a plain solve on it (the bounds
 * freed and cleared between solves); run under AddressSanitizer by tools/asan_test.sh (RELEASE_CHECKS.md R5). */
#include <stdio.h>
#include <math.h>
typedef struct BriskResult BriskResult;
BriskResult *brisk_result_new(void); void brisk_result_delete(BriskResult *);
int brisk_solve_file(const char *f, int n, const char *const *o, BriskResult *r);
int brisk_result_bound_of(const BriskResult *r, int side, double *out);
const char *brisk_result_bound_src(const BriskResult *r, int side);
const double *brisk_result_bound_X(const BriskResult *r, int k);
const double *brisk_result_bound_y(const BriskResult *r);
void brisk_set_output(int mode, int (*cb)(const char *, int));
int main(int argc, char **argv) {
    brisk_set_output(1, NULL);
    BriskResult *r = brisk_result_new();
    const char *o1[] = { "-bound", "pd", "-threads", "2" };
    for (int rep = 0; rep < 3; rep++) {         /* the same handle reused: bounds freed between solves? */
        int rc = brisk_solve_file(argv[1], 4, o1, r);
        double b[6];
        for (int s = 1; s <= 2; s++) { int q = brisk_result_bound_of(r, s, b); printf("rep %d rc %d side %d: %d value %.10g certified %g rigorous %.10g src [%s]\n", rep, rc, s, q, b[0], b[4], b[5], brisk_result_bound_src(r, s)); }
        printf("  X0 %p y %p\n", (void *)brisk_result_bound_X(r, 0), (void *)brisk_result_bound_y(r));
    }
    const char *o2[] = { "-threads", "2" };
    brisk_solve_file(argv[1], 2, o2, r);
    double b[6]; printf("after a plain solve: side1 %d side2 %d X0 %p\n", brisk_result_bound_of(r, 1, b), brisk_result_bound_of(r, 2, b), (void *)brisk_result_bound_X(r, 0));
    brisk_result_delete(r);
    return 0;
}
