/* blas_one.h (28 October 2026, the SOCP developer): OpenBLAS on one thread for a solve, its count given back after it, as
 * every library's threading is set for the call and given back (the SOCP solver's socp_solve).
 * The command line starts OpenBLAS on one thread (main.c); through the library (Python, CVXPY, Julia, MATLAB, the C API) it
 * kept the host's count, and each of its calls made inside a factorization's parallel blocks started that many threads more:
 * H_R_beamforming_2560 (SOCP) through Python 1.3 s on one thread, 2.8 s on two, 7.0 s on four, and 1.2 s on four with
 * OpenBLAS on one thread. The first of nested or concurrent solves of one file's
 * solver (the LP method's interior-point attempt beside its simplex method) sets it, the last gives it
 * back; a solve nested in another solver's finds one thread already and leaves it. OpenBLAS's functions are weak: absent
 * for other BLAS libraries, and null where Mach-O links no undefined weak references. */
/* Shared: any solver that calls BLAS inside its own parallel loops and can be reached from the library without brisk_main
 * (which sets OpenBLAS to one thread itself, brisk_threads_init) brackets its solve with blas_one_begin / blas_one_end; the
 * LP developer plans it for the LP model for branch and bound (lpmodel.c; ISSUES 628 on master). Nesting-safe within one
 * file: the counters are static, one pair per file that includes this header. */
#ifndef BRISK_BLAS_ONE_H
#define BRISK_BLAS_ONE_H
#include <stdlib.h>
#ifndef BLAS_PREFIX
#define BLAS_PREFIX
#endif
#define BO_CAT2(a, b) a##b
#define BO_CAT(a, b)  BO_CAT2(a, b)
#define BO_BL(name) BO_CAT(BLAS_PREFIX, name)
#if defined(__APPLE__) || defined(BRISK_NO_OPENBLAS)
static void (*const BO_BL(openblas_set_num_threads))(int) = 0;
static int (*const BO_BL(openblas_get_num_threads))(void) = 0;
#else
extern void BO_BL(openblas_set_num_threads)(int) __attribute__((weak));
extern int BO_BL(openblas_get_num_threads)(void) __attribute__((weak));
#endif
static int blas_one_users = 0, blas_one_saved = -1;
static inline void blas_one_begin(void) {
    static int keep = -1; if (keep < 0) keep = getenv("BRISK_BLASMT") != NULL;     /* (BRISK_BLASMT=1 keeps OpenBLAS threading, as for brisk_threads_init) */
    if (__atomic_fetch_add(&blas_one_users, 1, __ATOMIC_ACQ_REL) == 0 && !keep) {
        blas_one_saved = BO_BL(openblas_get_num_threads) && BO_BL(openblas_set_num_threads) ? BO_BL(openblas_get_num_threads)() : -1;
        if (blas_one_saved > 1) BO_BL(openblas_set_num_threads)(1);
    }
}
static inline void blas_one_end(void) {
    if (__atomic_sub_fetch(&blas_one_users, 1, __ATOMIC_ACQ_REL) == 0 && blas_one_saved > 1) BO_BL(openblas_set_num_threads)(blas_one_saved);
}
#endif
