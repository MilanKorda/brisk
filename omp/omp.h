/* omp.h - the OpenMP API subset BRISK uses, for the bundled runtime (omp/brisk_omp.c).
 * With it, a compiler that understands the OpenMP pragmas but ships no runtime (Apple clang:
 * -Xpreprocessor -fopenmp; any clang without libomp) builds a multithreaded BRISK with no
 * library to install. The compiler lowers the pragmas to calls of the LLVM OpenMP runtime
 * interface (__kmpc_*); brisk_omp.c implements the part of it these sources need. */
#ifndef BRISK_OMP_H
#define BRISK_OMP_H
#ifdef __cplusplus
extern "C" {
#endif
int    omp_get_thread_num(void);
int    omp_get_num_threads(void);
int    omp_get_max_threads(void);
void   omp_set_num_threads(int n);
int    omp_in_parallel(void);
int    omp_get_num_procs(void);
int    omp_get_level(void);
void   omp_set_dynamic(int d);
int    omp_get_dynamic(void);
void   omp_set_nested(int n);
int    omp_get_nested(void);
void   omp_set_max_active_levels(int n);
int    omp_get_max_active_levels(void);
double omp_get_wtime(void);
double omp_get_wtick(void);
#ifdef __cplusplus
}
#endif
#endif
