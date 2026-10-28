/* The build without OpenMP (`make OMP=0`; ISSUES 713): the runtime's queries for one thread (omp/ompstub.h). main.c's
 * weak definitions of three of them stay for a static SuiteSparse built with OpenMP; these strong ones take precedence. */
#include <time.h>
#include "ompstub.h"

int omp_get_thread_num(void) { return 0; }
int omp_get_num_threads(void) { return 1; }
int omp_get_max_threads(void) { return 1; }
int omp_get_num_procs(void) { return 1; }
int omp_in_parallel(void) { return 0; }
void omp_set_num_threads(int n) { (void)n; }
double omp_get_wtime(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + 1e-9 * t.tv_nsec;
}
