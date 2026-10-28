/* The build without OpenMP (`make OMP=0`; ISSUES 713): the OpenMP runtime's queries that the solvers call outside their
 * `#pragma omp` regions, declared for every file (the Makefile passes `-include omp/ompstub.h`) and defined in
 * omp/ompstub.c for one thread: thread 0 of a team of one, never inside a parallel region. */
#ifndef BRISK_OMPSTUB_H
#define BRISK_OMPSTUB_H
int omp_get_thread_num(void);
int omp_get_num_threads(void);
int omp_get_max_threads(void);
int omp_get_num_procs(void);
int omp_in_parallel(void);
void omp_set_num_threads(int n);
double omp_get_wtime(void);
#endif
