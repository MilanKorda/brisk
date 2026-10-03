/* brisk_omp.c - a small OpenMP runtime bundled with BRISK (4.35).
 *
 * Why: Apple's clang understands the OpenMP pragmas (-Xpreprocessor -fopenmp) but ships no
 * runtime library; the usual answer is "brew install libomp", and then the MEX file depends on
 * a dylib that MATLAB has to find at run time (and that may clash with an OpenMP runtime
 * MATLAB loads itself). This file implements the part of the LLVM OpenMP runtime interface
 * (the __kmpc_* entry points clang lowers the pragmas to, plus the omp_* API) that BRISK's
 * sources use, with POSIX threads, so the build needs nothing but the compiler. It is compiled
 * into the binary / MEX file, so no symbol is exported and nothing can clash.
 *
 * Supported: parallel (with if and num_threads), for with schedule(static[,c]) and
 * schedule(dynamic[,c]) / guided / runtime, reduction (critical-section method), critical,
 * single, master/masked, barrier, flush, atomic (inlined by the compiler), tasks with
 * firstprivate/shared and taskwait (no depend clauses), nested parallel regions (serialized,
 * as libomp does by default). Unsupported constructs fail at link time (missing symbol), never
 * silently.
 *
 * Design: one persistent pool of worker threads and one team for the (single) active top-level
 * region; workers spin briefly and then sleep on a condition variable between regions.
 * Barriers are counter + epoch (no ABA across regions), executing queued tasks while they
 * wait, and complete only when the team has no outstanding task. Tasks go to one team queue
 * (mutex); a thread in taskwait runs only children of the task it waits in (libgomp's rule, so
 * a suspended task's per-thread scratch data is never reused by an unrelated task). Worksharing
 * loops with dynamic schedules share a ring of dispatch buffers indexed by the per-thread loop
 * count. Environment: OMP_NUM_THREADS (default: all online processors), BRISK_OMP_SPIN_US
 * (busy-wait time before sleeping, default 1000). */
#include "omp.h"
#include <pthread.h>
#include <stdarg.h>
#include <sched.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

/* every symbol of this file is internal to the binary or MEX file it is linked into: no
 * interposition with (or of) an OpenMP runtime the host process may have loaded */
#pragma GCC visibility push(hidden)

typedef int32_t kmp_int32;
typedef uint32_t kmp_uint32;
typedef int64_t kmp_int64;
typedef uint64_t kmp_uint64;
typedef struct ident ident_t;              /* source location, unused */
typedef kmp_int32 kmp_critical_name[8];
typedef kmp_int32 (*kmp_routine_entry_t)(kmp_int32, void *);
typedef struct { void *shareds; kmp_routine_entry_t routine; kmp_int32 part_id; } kmp_task_t;   /* the head of the compiler's task struct */

#define MAXARGS 64
#define RING 8
#define MAXTHREADS 256
#define MAXDEPTH 32

static inline void cpu_relax(void) {
#if defined(__aarch64__) || defined(__arm64__)
    __asm__ __volatile__("yield");
#elif defined(__x86_64__) || defined(__i386__)
    __asm__ __volatile__("pause");
#endif
}
static double now_us(void) { struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts); return ts.tv_sec * 1e6 + ts.tv_nsec * 1e-3; }
static double g_spin_us = 1000.0;
/* waiting: busy for g_spin_us, then yield, then short sleeps */
/* with more threads than processors a spinning thread takes the processor from the one it
 * waits for: the busy phase is then cut to a few microseconds (as libgomp does) */
static atomic_int g_oversub = 0;
typedef struct { int n; double t0; } Backoff;
static void backoff(Backoff *b) {
    if (b->n++ == 0) b->t0 = now_us();
    const int ov = atomic_load_explicit(&g_oversub, memory_order_relaxed);
    const double spin = ov ? 5.0 : g_spin_us;
    if ((b->n & 63) != 0 && !ov) { cpu_relax(); return; }
    const double el = now_us() - b->t0;
    if (el < spin) cpu_relax();
    else if (el < 10 * spin + (ov ? 20000 : 0)) sched_yield();
    else { struct timespec ts = { 0, 50000 }; nanosleep(&ts, NULL); }
}

/* ---- tasks ------------------------------------------------------------------------------ */
typedef struct Task {
    struct Task *parent;                   /* the task (or implicit task) that created it */
    atomic_int children;                   /* direct children not yet completed (taskwait) */
    atomic_int refs;                       /* 1 (itself, until completed) + live children: freed at 0 */
    int implicit;
    int creator;                           /* thread number of the creating thread */
    int cls;                               /* size class of its memory block, -1 malloc */
    kmp_routine_entry_t routine;
    kmp_task_t *kt;
} Task;
#define HDR_SIZE ((sizeof(Task) + 15) & ~(size_t)15)

/* ---- worksharing dispatch buffers ------------------------------------------------------- */
typedef struct {
    atomic_flag lock;
    long gen;                              /* loop sequence number it holds, -1 none */
    atomic_int done;                       /* threads finished with it */
    atomic_uint_fast64_t next;             /* next iteration index (dynamic / guided) */
    uint64_t trip, chunk;
    int sched;
    uint64_t lb;                           /* bit pattern of the lower bound */
    int64_t st;
} Disp;

/* a task deque per thread: the owner pushes and pops at the bottom (newest first: the data
 * is in its cache), the others steal at the top (oldest: the largest pieces) */
typedef struct Deque {
    _Alignas(64) atomic_flag lock;
    atomic_int n;
    Task **a; int cap, top;                /* ring buffer: elements top .. top + n - 1 (mod cap) */
} Deque;
typedef struct Team {
    int nth;
    void *fn; int argc; void *args[MAXARGS];
    _Alignas(64) atomic_int bar_count; atomic_int bar_nth; atomic_int inbar; _Alignas(64) atomic_long bar_epoch;
    atomic_long single_seq;
    Disp disp[RING];
    _Alignas(64) atomic_long outstanding;  /* tasks created and not completed */
    _Alignas(64) atomic_int queued;        /* tasks in the deques */
    struct Deque *dq;                      /* one per thread (MAXTHREADS, allocated with the team) */
    double t_fork;                         /* statistics */
} Team;

/* per-thread state (a small stack of it for serialized nested regions) */
typedef struct {
    Team *team; int tid; int level, active;
    long loopseq, single;
    Task *task;                            /* current explicit task, or NULL: the implicit one */
    Task *impl;                            /* the implicit task of this thread at this level */
    Disp *d; uint64_t d_k;                 /* current dispatch buffer, next static chunk index */
    int d_last_done;
} TState;
static __thread Task tl_impl0 = { .implicit = 1 };   /* outside any parallel region */
static __thread TState tl;
static __thread Task tl_impl[MAXDEPTH + 1];          /* implicit tasks by depth: children still
                                                        decrementing an outer one stay correct
                                                        while a nested region runs */
static __thread int tl_push_nt = 0;
static __thread Team *tl_solo[MAXDEPTH];   /* one-thread teams for serialized regions, per depth */
static __thread TState tl_saved[MAXDEPTH];
static __thread int tl_depth = 0;

/* ---- globals ------------------------------------------------------------------------------ */
static pthread_once_t g_once = PTHREAD_ONCE_INIT;
static int g_nprocs = 1;
static atomic_int g_maxthreads = 1;
static int g_dynamic = 0, g_max_levels = 1;
static Team g_team;
static atomic_int g_busy = 0;              /* the pool's team is in use */
static pthread_mutex_t g_mx = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t g_cv = PTHREAD_COND_INITIALIZER;
static atomic_long g_gen = 0;
static atomic_int g_quit = 0;
static int g_nworkers = 0;
static pthread_t g_workers[MAXTHREADS];
static long g_seen0[MAXTHREADS];
static atomic_int g_alive = 0;
static pthread_mutex_t g_red_mx = PTHREAD_MUTEX_INITIALIZER;
/* BRISK_OMP_STATS=1: counts and waiting times, printed at exit */
static int g_stats = 0;
static atomic_long st_forks, st_serial, st_tasks, st_barriers, st_wait_ns, st_wake_ns, st_sleeps;

static void team_init(Team *t, int nth) {
    memset(t, 0, sizeof *t);
    t->nth = nth; atomic_store(&t->bar_nth, nth);
    t->dq = calloc(MAXTHREADS, sizeof(Deque));
    for (int i = 0; i < MAXTHREADS; i++) atomic_flag_clear(&t->dq[i].lock);
    for (int i = 0; i < RING; i++) { atomic_flag_clear(&t->disp[i].lock); t->disp[i].gen = -1; }
}
static void rt_init(void) {
    long np = sysconf(_SC_NPROCESSORS_ONLN);
    g_nprocs = np > 0 ? (int)np : 1;
    int nt = g_nprocs;
    const char *e = getenv("OMP_NUM_THREADS");
    if (e && atoi(e) > 0) nt = atoi(e);
    if (nt > MAXTHREADS) nt = MAXTHREADS;
    atomic_store(&g_maxthreads, nt);
    if ((e = getenv("BRISK_OMP_SPIN_US"))) g_spin_us = atof(e);
    g_stats = getenv("BRISK_OMP_STATS") != NULL;
    team_init(&g_team, 1);
}
static inline void rt_ensure(void) { pthread_once(&g_once, rt_init); }

/* call the outlined region function with its captured arguments (all pointer-sized): it is a
 * non-variadic function, so the call must not go through a variadic prototype (Apple arm64
 * passes variadic arguments on the stack) - one case per argument count */
static void invoke_microtask(void *fn, int tid, int argc, void **a) {
    int32_t g = tid, b = tid;
    switch (argc) {
    case 0: ((void (*)(int32_t *, int32_t *))fn)(&g, &b); break;
    case 1: ((void (*)(int32_t *, int32_t *, void *))fn)(&g, &b, a[0]); break;
    case 2: ((void (*)(int32_t *, int32_t *, void *, void *))fn)(&g, &b, a[0], a[1]); break;
    case 3: ((void (*)(int32_t *, int32_t *, void *, void *, void *))fn)(&g, &b, a[0], a[1], a[2]); break;
    case 4: ((void (*)(int32_t *, int32_t *, void *, void *, void *, void *))fn)(&g, &b, a[0], a[1], a[2], a[3]); break;
    case 5: ((void (*)(int32_t *, int32_t *, void *, void *, void *, void *, void *))fn)(&g, &b, a[0], a[1], a[2], a[3], a[4]); break;
    case 6: ((void (*)(int32_t *, int32_t *, void *, void *, void *, void *, void *, void *))fn)(&g, &b, a[0], a[1], a[2], a[3], a[4], a[5]); break;
    case 7: ((void (*)(int32_t *, int32_t *, void *, void *, void *, void *, void *, void *, void *))fn)(&g, &b, a[0], a[1], a[2], a[3], a[4], a[5], a[6]); break;
    case 8: ((void (*)(int32_t *, int32_t *, void *, void *, void *, void *, void *, void *, void *, void *))fn)(&g, &b, a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7]); break;
    case 9: ((void (*)(int32_t *, int32_t *, void *, void *, void *, void *, void *, void *, void *, void *, void *))fn)(&g, &b, a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8]); break;
    case 10: ((void (*)(int32_t *, int32_t *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *))fn)(&g, &b, a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8], a[9]); break;
    case 11: ((void (*)(int32_t *, int32_t *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *))fn)(&g, &b, a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8], a[9], a[10]); break;
    case 12: ((void (*)(int32_t *, int32_t *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *))fn)(&g, &b, a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8], a[9], a[10], a[11]); break;
    case 13: ((void (*)(int32_t *, int32_t *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *))fn)(&g, &b, a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8], a[9], a[10], a[11], a[12]); break;
    case 14: ((void (*)(int32_t *, int32_t *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *))fn)(&g, &b, a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8], a[9], a[10], a[11], a[12], a[13]); break;
    case 15: ((void (*)(int32_t *, int32_t *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *))fn)(&g, &b, a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8], a[9], a[10], a[11], a[12], a[13], a[14]); break;
    case 16: ((void (*)(int32_t *, int32_t *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *))fn)(&g, &b, a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8], a[9], a[10], a[11], a[12], a[13], a[14], a[15]); break;
    case 17: ((void (*)(int32_t *, int32_t *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *))fn)(&g, &b, a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8], a[9], a[10], a[11], a[12], a[13], a[14], a[15], a[16]); break;
    case 18: ((void (*)(int32_t *, int32_t *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *))fn)(&g, &b, a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8], a[9], a[10], a[11], a[12], a[13], a[14], a[15], a[16], a[17]); break;
    case 19: ((void (*)(int32_t *, int32_t *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *))fn)(&g, &b, a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8], a[9], a[10], a[11], a[12], a[13], a[14], a[15], a[16], a[17], a[18]); break;
    case 20: ((void (*)(int32_t *, int32_t *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *))fn)(&g, &b, a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8], a[9], a[10], a[11], a[12], a[13], a[14], a[15], a[16], a[17], a[18], a[19]); break;
    case 21: ((void (*)(int32_t *, int32_t *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *))fn)(&g, &b, a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8], a[9], a[10], a[11], a[12], a[13], a[14], a[15], a[16], a[17], a[18], a[19], a[20]); break;
    case 22: ((void (*)(int32_t *, int32_t *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *))fn)(&g, &b, a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8], a[9], a[10], a[11], a[12], a[13], a[14], a[15], a[16], a[17], a[18], a[19], a[20], a[21]); break;
    case 23: ((void (*)(int32_t *, int32_t *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *))fn)(&g, &b, a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8], a[9], a[10], a[11], a[12], a[13], a[14], a[15], a[16], a[17], a[18], a[19], a[20], a[21], a[22]); break;
    case 24: ((void (*)(int32_t *, int32_t *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *))fn)(&g, &b, a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8], a[9], a[10], a[11], a[12], a[13], a[14], a[15], a[16], a[17], a[18], a[19], a[20], a[21], a[22], a[23]); break;
    case 25: ((void (*)(int32_t *, int32_t *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *))fn)(&g, &b, a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8], a[9], a[10], a[11], a[12], a[13], a[14], a[15], a[16], a[17], a[18], a[19], a[20], a[21], a[22], a[23], a[24]); break;
    case 26: ((void (*)(int32_t *, int32_t *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *))fn)(&g, &b, a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8], a[9], a[10], a[11], a[12], a[13], a[14], a[15], a[16], a[17], a[18], a[19], a[20], a[21], a[22], a[23], a[24], a[25]); break;
    case 27: ((void (*)(int32_t *, int32_t *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *))fn)(&g, &b, a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8], a[9], a[10], a[11], a[12], a[13], a[14], a[15], a[16], a[17], a[18], a[19], a[20], a[21], a[22], a[23], a[24], a[25], a[26]); break;
    case 28: ((void (*)(int32_t *, int32_t *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *))fn)(&g, &b, a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8], a[9], a[10], a[11], a[12], a[13], a[14], a[15], a[16], a[17], a[18], a[19], a[20], a[21], a[22], a[23], a[24], a[25], a[26], a[27]); break;
    case 29: ((void (*)(int32_t *, int32_t *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *))fn)(&g, &b, a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8], a[9], a[10], a[11], a[12], a[13], a[14], a[15], a[16], a[17], a[18], a[19], a[20], a[21], a[22], a[23], a[24], a[25], a[26], a[27], a[28]); break;
    case 30: ((void (*)(int32_t *, int32_t *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *))fn)(&g, &b, a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8], a[9], a[10], a[11], a[12], a[13], a[14], a[15], a[16], a[17], a[18], a[19], a[20], a[21], a[22], a[23], a[24], a[25], a[26], a[27], a[28], a[29]); break;
    case 31: ((void (*)(int32_t *, int32_t *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *))fn)(&g, &b, a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8], a[9], a[10], a[11], a[12], a[13], a[14], a[15], a[16], a[17], a[18], a[19], a[20], a[21], a[22], a[23], a[24], a[25], a[26], a[27], a[28], a[29], a[30]); break;
    case 32: ((void (*)(int32_t *, int32_t *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *))fn)(&g, &b, a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8], a[9], a[10], a[11], a[12], a[13], a[14], a[15], a[16], a[17], a[18], a[19], a[20], a[21], a[22], a[23], a[24], a[25], a[26], a[27], a[28], a[29], a[30], a[31]); break;
    case 33: ((void (*)(int32_t *, int32_t *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *))fn)(&g, &b, a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8], a[9], a[10], a[11], a[12], a[13], a[14], a[15], a[16], a[17], a[18], a[19], a[20], a[21], a[22], a[23], a[24], a[25], a[26], a[27], a[28], a[29], a[30], a[31], a[32]); break;
    case 34: ((void (*)(int32_t *, int32_t *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *))fn)(&g, &b, a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8], a[9], a[10], a[11], a[12], a[13], a[14], a[15], a[16], a[17], a[18], a[19], a[20], a[21], a[22], a[23], a[24], a[25], a[26], a[27], a[28], a[29], a[30], a[31], a[32], a[33]); break;
    case 35: ((void (*)(int32_t *, int32_t *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *))fn)(&g, &b, a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8], a[9], a[10], a[11], a[12], a[13], a[14], a[15], a[16], a[17], a[18], a[19], a[20], a[21], a[22], a[23], a[24], a[25], a[26], a[27], a[28], a[29], a[30], a[31], a[32], a[33], a[34]); break;
    case 36: ((void (*)(int32_t *, int32_t *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *))fn)(&g, &b, a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8], a[9], a[10], a[11], a[12], a[13], a[14], a[15], a[16], a[17], a[18], a[19], a[20], a[21], a[22], a[23], a[24], a[25], a[26], a[27], a[28], a[29], a[30], a[31], a[32], a[33], a[34], a[35]); break;
    case 37: ((void (*)(int32_t *, int32_t *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *))fn)(&g, &b, a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8], a[9], a[10], a[11], a[12], a[13], a[14], a[15], a[16], a[17], a[18], a[19], a[20], a[21], a[22], a[23], a[24], a[25], a[26], a[27], a[28], a[29], a[30], a[31], a[32], a[33], a[34], a[35], a[36]); break;
    case 38: ((void (*)(int32_t *, int32_t *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *))fn)(&g, &b, a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8], a[9], a[10], a[11], a[12], a[13], a[14], a[15], a[16], a[17], a[18], a[19], a[20], a[21], a[22], a[23], a[24], a[25], a[26], a[27], a[28], a[29], a[30], a[31], a[32], a[33], a[34], a[35], a[36], a[37]); break;
    case 39: ((void (*)(int32_t *, int32_t *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *))fn)(&g, &b, a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8], a[9], a[10], a[11], a[12], a[13], a[14], a[15], a[16], a[17], a[18], a[19], a[20], a[21], a[22], a[23], a[24], a[25], a[26], a[27], a[28], a[29], a[30], a[31], a[32], a[33], a[34], a[35], a[36], a[37], a[38]); break;
    case 40: ((void (*)(int32_t *, int32_t *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *))fn)(&g, &b, a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8], a[9], a[10], a[11], a[12], a[13], a[14], a[15], a[16], a[17], a[18], a[19], a[20], a[21], a[22], a[23], a[24], a[25], a[26], a[27], a[28], a[29], a[30], a[31], a[32], a[33], a[34], a[35], a[36], a[37], a[38], a[39]); break;
    case 41: ((void (*)(int32_t *, int32_t *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *))fn)(&g, &b, a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8], a[9], a[10], a[11], a[12], a[13], a[14], a[15], a[16], a[17], a[18], a[19], a[20], a[21], a[22], a[23], a[24], a[25], a[26], a[27], a[28], a[29], a[30], a[31], a[32], a[33], a[34], a[35], a[36], a[37], a[38], a[39], a[40]); break;
    case 42: ((void (*)(int32_t *, int32_t *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *))fn)(&g, &b, a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8], a[9], a[10], a[11], a[12], a[13], a[14], a[15], a[16], a[17], a[18], a[19], a[20], a[21], a[22], a[23], a[24], a[25], a[26], a[27], a[28], a[29], a[30], a[31], a[32], a[33], a[34], a[35], a[36], a[37], a[38], a[39], a[40], a[41]); break;
    case 43: ((void (*)(int32_t *, int32_t *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *))fn)(&g, &b, a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8], a[9], a[10], a[11], a[12], a[13], a[14], a[15], a[16], a[17], a[18], a[19], a[20], a[21], a[22], a[23], a[24], a[25], a[26], a[27], a[28], a[29], a[30], a[31], a[32], a[33], a[34], a[35], a[36], a[37], a[38], a[39], a[40], a[41], a[42]); break;
    case 44: ((void (*)(int32_t *, int32_t *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *))fn)(&g, &b, a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8], a[9], a[10], a[11], a[12], a[13], a[14], a[15], a[16], a[17], a[18], a[19], a[20], a[21], a[22], a[23], a[24], a[25], a[26], a[27], a[28], a[29], a[30], a[31], a[32], a[33], a[34], a[35], a[36], a[37], a[38], a[39], a[40], a[41], a[42], a[43]); break;
    case 45: ((void (*)(int32_t *, int32_t *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *))fn)(&g, &b, a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8], a[9], a[10], a[11], a[12], a[13], a[14], a[15], a[16], a[17], a[18], a[19], a[20], a[21], a[22], a[23], a[24], a[25], a[26], a[27], a[28], a[29], a[30], a[31], a[32], a[33], a[34], a[35], a[36], a[37], a[38], a[39], a[40], a[41], a[42], a[43], a[44]); break;
    case 46: ((void (*)(int32_t *, int32_t *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *))fn)(&g, &b, a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8], a[9], a[10], a[11], a[12], a[13], a[14], a[15], a[16], a[17], a[18], a[19], a[20], a[21], a[22], a[23], a[24], a[25], a[26], a[27], a[28], a[29], a[30], a[31], a[32], a[33], a[34], a[35], a[36], a[37], a[38], a[39], a[40], a[41], a[42], a[43], a[44], a[45]); break;
    case 47: ((void (*)(int32_t *, int32_t *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *))fn)(&g, &b, a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8], a[9], a[10], a[11], a[12], a[13], a[14], a[15], a[16], a[17], a[18], a[19], a[20], a[21], a[22], a[23], a[24], a[25], a[26], a[27], a[28], a[29], a[30], a[31], a[32], a[33], a[34], a[35], a[36], a[37], a[38], a[39], a[40], a[41], a[42], a[43], a[44], a[45], a[46]); break;
    case 48: ((void (*)(int32_t *, int32_t *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *))fn)(&g, &b, a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8], a[9], a[10], a[11], a[12], a[13], a[14], a[15], a[16], a[17], a[18], a[19], a[20], a[21], a[22], a[23], a[24], a[25], a[26], a[27], a[28], a[29], a[30], a[31], a[32], a[33], a[34], a[35], a[36], a[37], a[38], a[39], a[40], a[41], a[42], a[43], a[44], a[45], a[46], a[47]); break;
    case 49: ((void (*)(int32_t *, int32_t *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *))fn)(&g, &b, a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8], a[9], a[10], a[11], a[12], a[13], a[14], a[15], a[16], a[17], a[18], a[19], a[20], a[21], a[22], a[23], a[24], a[25], a[26], a[27], a[28], a[29], a[30], a[31], a[32], a[33], a[34], a[35], a[36], a[37], a[38], a[39], a[40], a[41], a[42], a[43], a[44], a[45], a[46], a[47], a[48]); break;
    case 50: ((void (*)(int32_t *, int32_t *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *))fn)(&g, &b, a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8], a[9], a[10], a[11], a[12], a[13], a[14], a[15], a[16], a[17], a[18], a[19], a[20], a[21], a[22], a[23], a[24], a[25], a[26], a[27], a[28], a[29], a[30], a[31], a[32], a[33], a[34], a[35], a[36], a[37], a[38], a[39], a[40], a[41], a[42], a[43], a[44], a[45], a[46], a[47], a[48], a[49]); break;
    case 51: ((void (*)(int32_t *, int32_t *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *))fn)(&g, &b, a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8], a[9], a[10], a[11], a[12], a[13], a[14], a[15], a[16], a[17], a[18], a[19], a[20], a[21], a[22], a[23], a[24], a[25], a[26], a[27], a[28], a[29], a[30], a[31], a[32], a[33], a[34], a[35], a[36], a[37], a[38], a[39], a[40], a[41], a[42], a[43], a[44], a[45], a[46], a[47], a[48], a[49], a[50]); break;
    case 52: ((void (*)(int32_t *, int32_t *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *))fn)(&g, &b, a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8], a[9], a[10], a[11], a[12], a[13], a[14], a[15], a[16], a[17], a[18], a[19], a[20], a[21], a[22], a[23], a[24], a[25], a[26], a[27], a[28], a[29], a[30], a[31], a[32], a[33], a[34], a[35], a[36], a[37], a[38], a[39], a[40], a[41], a[42], a[43], a[44], a[45], a[46], a[47], a[48], a[49], a[50], a[51]); break;
    case 53: ((void (*)(int32_t *, int32_t *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *))fn)(&g, &b, a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8], a[9], a[10], a[11], a[12], a[13], a[14], a[15], a[16], a[17], a[18], a[19], a[20], a[21], a[22], a[23], a[24], a[25], a[26], a[27], a[28], a[29], a[30], a[31], a[32], a[33], a[34], a[35], a[36], a[37], a[38], a[39], a[40], a[41], a[42], a[43], a[44], a[45], a[46], a[47], a[48], a[49], a[50], a[51], a[52]); break;
    case 54: ((void (*)(int32_t *, int32_t *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *))fn)(&g, &b, a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8], a[9], a[10], a[11], a[12], a[13], a[14], a[15], a[16], a[17], a[18], a[19], a[20], a[21], a[22], a[23], a[24], a[25], a[26], a[27], a[28], a[29], a[30], a[31], a[32], a[33], a[34], a[35], a[36], a[37], a[38], a[39], a[40], a[41], a[42], a[43], a[44], a[45], a[46], a[47], a[48], a[49], a[50], a[51], a[52], a[53]); break;
    case 55: ((void (*)(int32_t *, int32_t *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *))fn)(&g, &b, a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8], a[9], a[10], a[11], a[12], a[13], a[14], a[15], a[16], a[17], a[18], a[19], a[20], a[21], a[22], a[23], a[24], a[25], a[26], a[27], a[28], a[29], a[30], a[31], a[32], a[33], a[34], a[35], a[36], a[37], a[38], a[39], a[40], a[41], a[42], a[43], a[44], a[45], a[46], a[47], a[48], a[49], a[50], a[51], a[52], a[53], a[54]); break;
    case 56: ((void (*)(int32_t *, int32_t *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *))fn)(&g, &b, a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8], a[9], a[10], a[11], a[12], a[13], a[14], a[15], a[16], a[17], a[18], a[19], a[20], a[21], a[22], a[23], a[24], a[25], a[26], a[27], a[28], a[29], a[30], a[31], a[32], a[33], a[34], a[35], a[36], a[37], a[38], a[39], a[40], a[41], a[42], a[43], a[44], a[45], a[46], a[47], a[48], a[49], a[50], a[51], a[52], a[53], a[54], a[55]); break;
    case 57: ((void (*)(int32_t *, int32_t *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *))fn)(&g, &b, a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8], a[9], a[10], a[11], a[12], a[13], a[14], a[15], a[16], a[17], a[18], a[19], a[20], a[21], a[22], a[23], a[24], a[25], a[26], a[27], a[28], a[29], a[30], a[31], a[32], a[33], a[34], a[35], a[36], a[37], a[38], a[39], a[40], a[41], a[42], a[43], a[44], a[45], a[46], a[47], a[48], a[49], a[50], a[51], a[52], a[53], a[54], a[55], a[56]); break;
    case 58: ((void (*)(int32_t *, int32_t *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *))fn)(&g, &b, a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8], a[9], a[10], a[11], a[12], a[13], a[14], a[15], a[16], a[17], a[18], a[19], a[20], a[21], a[22], a[23], a[24], a[25], a[26], a[27], a[28], a[29], a[30], a[31], a[32], a[33], a[34], a[35], a[36], a[37], a[38], a[39], a[40], a[41], a[42], a[43], a[44], a[45], a[46], a[47], a[48], a[49], a[50], a[51], a[52], a[53], a[54], a[55], a[56], a[57]); break;
    case 59: ((void (*)(int32_t *, int32_t *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *))fn)(&g, &b, a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8], a[9], a[10], a[11], a[12], a[13], a[14], a[15], a[16], a[17], a[18], a[19], a[20], a[21], a[22], a[23], a[24], a[25], a[26], a[27], a[28], a[29], a[30], a[31], a[32], a[33], a[34], a[35], a[36], a[37], a[38], a[39], a[40], a[41], a[42], a[43], a[44], a[45], a[46], a[47], a[48], a[49], a[50], a[51], a[52], a[53], a[54], a[55], a[56], a[57], a[58]); break;
    case 60: ((void (*)(int32_t *, int32_t *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *))fn)(&g, &b, a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8], a[9], a[10], a[11], a[12], a[13], a[14], a[15], a[16], a[17], a[18], a[19], a[20], a[21], a[22], a[23], a[24], a[25], a[26], a[27], a[28], a[29], a[30], a[31], a[32], a[33], a[34], a[35], a[36], a[37], a[38], a[39], a[40], a[41], a[42], a[43], a[44], a[45], a[46], a[47], a[48], a[49], a[50], a[51], a[52], a[53], a[54], a[55], a[56], a[57], a[58], a[59]); break;
    case 61: ((void (*)(int32_t *, int32_t *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *))fn)(&g, &b, a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8], a[9], a[10], a[11], a[12], a[13], a[14], a[15], a[16], a[17], a[18], a[19], a[20], a[21], a[22], a[23], a[24], a[25], a[26], a[27], a[28], a[29], a[30], a[31], a[32], a[33], a[34], a[35], a[36], a[37], a[38], a[39], a[40], a[41], a[42], a[43], a[44], a[45], a[46], a[47], a[48], a[49], a[50], a[51], a[52], a[53], a[54], a[55], a[56], a[57], a[58], a[59], a[60]); break;
    case 62: ((void (*)(int32_t *, int32_t *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *))fn)(&g, &b, a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8], a[9], a[10], a[11], a[12], a[13], a[14], a[15], a[16], a[17], a[18], a[19], a[20], a[21], a[22], a[23], a[24], a[25], a[26], a[27], a[28], a[29], a[30], a[31], a[32], a[33], a[34], a[35], a[36], a[37], a[38], a[39], a[40], a[41], a[42], a[43], a[44], a[45], a[46], a[47], a[48], a[49], a[50], a[51], a[52], a[53], a[54], a[55], a[56], a[57], a[58], a[59], a[60], a[61]); break;
    case 63: ((void (*)(int32_t *, int32_t *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *))fn)(&g, &b, a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8], a[9], a[10], a[11], a[12], a[13], a[14], a[15], a[16], a[17], a[18], a[19], a[20], a[21], a[22], a[23], a[24], a[25], a[26], a[27], a[28], a[29], a[30], a[31], a[32], a[33], a[34], a[35], a[36], a[37], a[38], a[39], a[40], a[41], a[42], a[43], a[44], a[45], a[46], a[47], a[48], a[49], a[50], a[51], a[52], a[53], a[54], a[55], a[56], a[57], a[58], a[59], a[60], a[61], a[62]); break;
    case 64: ((void (*)(int32_t *, int32_t *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *))fn)(&g, &b, a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8], a[9], a[10], a[11], a[12], a[13], a[14], a[15], a[16], a[17], a[18], a[19], a[20], a[21], a[22], a[23], a[24], a[25], a[26], a[27], a[28], a[29], a[30], a[31], a[32], a[33], a[34], a[35], a[36], a[37], a[38], a[39], a[40], a[41], a[42], a[43], a[44], a[45], a[46], a[47], a[48], a[49], a[50], a[51], a[52], a[53], a[54], a[55], a[56], a[57], a[58], a[59], a[60], a[61], a[62], a[63]); break;
    default: fprintf(stderr, "brisk_omp: a parallel region with %d captured variables (at most 64 supported)\n", argc); abort();
    }
}

/* ---- task queue ------------------------------------------------------------------------- */
static void blk_free(void *p, int cls);
static void task_release(Task *t) {
    if (!t->implicit && atomic_fetch_sub(&t->refs, 1) == 1) blk_free(t, t->cls);
}
static void run_task(Task *t) {
    Task *prev = tl.task;
    tl.task = t;
    t->routine(tl.tid, t->kt);
    tl.task = prev;
    Task *p = t->parent;
    Team *tm = tl.team;
    atomic_fetch_sub(&p->children, 1);
    task_release(p);                       /* the child's reference on its parent */
    task_release(t);                       /* its own */
    if (tm) atomic_fetch_sub(&tm->outstanding, 1);
}
static inline void dq_lock(Deque *d) { while (atomic_flag_test_and_set_explicit(&d->lock, memory_order_acquire)) cpu_relax(); }
static inline void dq_unlock(Deque *d) { atomic_flag_clear_explicit(&d->lock, memory_order_release); }
static void dq_push(Team *tm, int who, Task *t) {
    Deque *d = &tm->dq[who];
    dq_lock(d);
    const int n = atomic_load_explicit(&d->n, memory_order_relaxed);
    if (n == d->cap) {                     /* grow, unrolling the ring */
        const int nc = d->cap ? 2 * d->cap : 256;
        Task **na = malloc(sizeof(Task *) * nc);
        for (int i = 0; i < n; i++) na[i] = d->a[(d->top + i) % d->cap];
        free(d->a); d->a = na; d->cap = nc; d->top = 0;
    }
    d->a[(d->top + n) % d->cap] = t;
    atomic_store_explicit(&d->n, n + 1, memory_order_release);
    dq_unlock(d);
    atomic_fetch_add_explicit(&tm->queued, 1, memory_order_release);
}
/* own deque: the newest task, or (taskwait) the newest child of `parent` */
static Task *dq_pop_own(Team *tm, Task *parent) {
    Deque *d = &tm->dq[tl.tid];
    if (atomic_load_explicit(&d->n, memory_order_acquire) == 0) return NULL;
    dq_lock(d);
    int n = atomic_load_explicit(&d->n, memory_order_relaxed);
    Task *t = NULL;
    if (n > 0) {
        int i = n - 1;
        if (parent) while (i >= 0 && d->a[(d->top + i) % d->cap]->parent != parent) i--;
        if (i >= 0) {
            t = d->a[(d->top + i) % d->cap];
            for (int j = i; j < n - 1; j++) d->a[(d->top + j) % d->cap] = d->a[(d->top + j + 1) % d->cap];
            atomic_store_explicit(&d->n, n - 1, memory_order_release);
        }
    }
    dq_unlock(d);
    if (t) atomic_fetch_sub_explicit(&tm->queued, 1, memory_order_acq_rel);
    return t;
}
static Task *dq_steal(Team *tm) {
    const int nth = tm->nth, me = tl.tid;
    for (int k = 1; k < nth; k++) {
        Deque *d = &tm->dq[(me + k) % nth];
        if (atomic_load_explicit(&d->n, memory_order_acquire) == 0) continue;
        dq_lock(d);
        Task *t = NULL;
        const int n = atomic_load_explicit(&d->n, memory_order_relaxed);
        if (n > 0) { t = d->a[d->top]; d->top = (d->top + 1) % d->cap; atomic_store_explicit(&d->n, n - 1, memory_order_release); }
        dq_unlock(d);
        if (t) { atomic_fetch_sub_explicit(&tm->queued, 1, memory_order_acq_rel); return t; }
    }
    return NULL;
}
/* any queued task (barriers), or only a child of `parent` (taskwait: the children a thread
 * made are in its own deque unless stolen, and then they are running) */
static Task *q_pop(Team *tm, Task *parent) {
    if (atomic_load_explicit(&tm->queued, memory_order_acquire) == 0) return NULL;
    Task *t = dq_pop_own(tm, parent);
    if (!t && !parent) t = dq_steal(tm);
    return t;
}

/* ---- barrier: counter + epoch; runs queued tasks; completes when no task is outstanding ---- */
static void team_barrier(Team *tm) {
    if (tm->nth == 1) {                    /* tasks of a one-thread team ran when created */
        Task *t; while ((t = q_pop(tm, NULL))) run_task(t);
        return;
    }
    atomic_fetch_add_explicit(&tm->inbar, 1, memory_order_acq_rel);
    const long my = atomic_load_explicit(&tm->bar_epoch, memory_order_acquire);
    atomic_fetch_add_explicit(&tm->bar_count, 1, memory_order_acq_rel);
    Backoff b = { 0, 0 };
    const double tw0 = g_stats ? now_us() : 0; double trun = 0;
    if (g_stats) atomic_fetch_add(&st_barriers, 1);
    for (;;) {
        if (atomic_load_explicit(&tm->bar_epoch, memory_order_acquire) != my) {
            if (g_stats) atomic_fetch_add(&st_wait_ns, (long)(1e3 * (now_us() - tw0 - trun)));
            atomic_fetch_sub_explicit(&tm->inbar, 1, memory_order_acq_rel);
            return;
        }
        Task *t = q_pop(tm, NULL);
        if (t) { const double r0 = g_stats ? now_us() : 0; run_task(t); if (g_stats) trun += now_us() - r0; b.n = 0; continue; }
        /* bar_nth, not nth: a thread still leaving the previous region's barrier may read it
         * while the next region is being set up (the release predicate is the same for it) */
        const int bn = atomic_load_explicit(&tm->bar_nth, memory_order_acquire);
        if (atomic_load_explicit(&tm->bar_count, memory_order_acquire) == bn &&
            atomic_load_explicit(&tm->outstanding, memory_order_acquire) == 0) {
            int exp = bn;
            if (atomic_compare_exchange_strong(&tm->bar_count, &exp, 0)) {
                atomic_fetch_add_explicit(&tm->bar_epoch, 1, memory_order_acq_rel);
                if (g_stats) atomic_fetch_add(&st_wait_ns, (long)(1e3 * (now_us() - tw0 - trun)));
                atomic_fetch_sub_explicit(&tm->inbar, 1, memory_order_acq_rel);
                return;
            }
        }
        backoff(&b);
    }
}

/* ---- thread state for an implicit task ---------------------------------------------------- */
static void enter_region(Team *tm, int tid, int active) {
    if (tl_depth >= MAXDEPTH) { fprintf(stderr, "brisk_omp: parallel regions nested deeper than %d\n", MAXDEPTH); abort(); }
    tl_saved[tl_depth] = tl; tl_depth++;
    const int level = tl.level + 1, act = tl.active + (active ? 1 : 0);
    memset(&tl, 0, sizeof tl);
    tl.team = tm; tl.tid = tid; tl.level = level; tl.active = act;
    Task *im = &tl_impl[tl_depth];
    memset(im, 0, sizeof *im); im->implicit = 1;
    tl.impl = im;
}
static void leave_region(void) { tl_depth--; tl = tl_saved[tl_depth]; }
static Team *solo_team(void) {
    const int d = tl_depth;
    if (!tl_solo[d]) { tl_solo[d] = malloc(sizeof(Team)); team_init(tl_solo[d], 1); }
    Team *t = tl_solo[d];
    t->single_seq = 0;
    for (int i = 0; i < RING; i++) t->disp[i].gen = -1;
    return t;
}

/* ---- the pool --------------------------------------------------------------------------- */
static void *worker_main(void *arg) {
    const int idx = (int)(intptr_t)arg;     /* thread number in a team: idx + 1 */
    long seen = g_seen0[idx];                /* the generation before the fork that created it */
    atomic_fetch_add(&g_alive, 1);
    for (;;) {
        Backoff b = { 0, 0 };
        long gnow;
        while ((gnow = atomic_load_explicit(&g_gen, memory_order_acquire)) == seen && !atomic_load(&g_quit)) {
            if (b.n > 0 && now_us() - b.t0 > (atomic_load_explicit(&g_oversub, memory_order_relaxed) ? 5.0 : g_spin_us)) {
                pthread_mutex_lock(&g_mx);
                if (g_stats) atomic_fetch_add(&st_sleeps, 1);
                while (atomic_load(&g_gen) == seen && !atomic_load(&g_quit)) pthread_cond_wait(&g_cv, &g_mx);
                pthread_mutex_unlock(&g_mx);
                break;
            }
            backoff(&b);
        }
        if (atomic_load(&g_quit)) break;
        seen = atomic_load_explicit(&g_gen, memory_order_acquire);
        Team *tm = &g_team;
        if (g_stats) atomic_fetch_add(&st_wake_ns, (long)(1e3 * (now_us() - tm->t_fork)));
        if (idx + 1 < tm->nth) {
            enter_region(tm, idx + 1, 1);
            invoke_microtask(tm->fn, idx + 1, tm->argc, tm->args);
            team_barrier(tm);
            leave_region();
        }
    }
    atomic_fetch_sub(&g_alive, 1);
    return NULL;
}
static void ensure_workers(int n) {
    while (g_nworkers < n) {
        pthread_attr_t at; pthread_attr_init(&at);
        pthread_attr_setstacksize(&at, (size_t)16 << 20);
        g_seen0[g_nworkers] = atomic_load(&g_gen);
        if (pthread_create(&g_workers[g_nworkers], &at, worker_main, (void *)(intptr_t)g_nworkers) != 0) { pthread_attr_destroy(&at); break; }
        pthread_attr_destroy(&at);
        g_nworkers++;
    }
}
/* unloading the MEX file (clear mex) with threads still running its code would crash MATLAB */
__attribute__((destructor)) static void rt_shutdown(void) {
    if (g_stats) fprintf(stderr, "brisk_omp: %ld forks, %ld serialized, %ld queued tasks, %ld barriers, idle in barriers %.3f s, fork-to-wake %.3f s total, %ld sleeps\n",
                         atomic_load(&st_forks), atomic_load(&st_serial), atomic_load(&st_tasks), atomic_load(&st_barriers), atomic_load(&st_wait_ns) * 1e-9, atomic_load(&st_wake_ns) * 1e-9, atomic_load(&st_sleeps));
    if (g_nworkers == 0) return;
    atomic_store(&g_quit, 1);
    pthread_mutex_lock(&g_mx); pthread_cond_broadcast(&g_cv); pthread_mutex_unlock(&g_mx);
    for (int w = 0; w < 400 && atomic_load(&g_alive) > 0; w++) { struct timespec ts = { 0, 1000000 }; nanosleep(&ts, NULL); }
    if (atomic_load(&g_alive) == 0) for (int i = 0; i < g_nworkers; i++) pthread_join(g_workers[i], NULL);
}

static void run_serialized(void *fn, int argc, void **args) {
    if (g_stats) atomic_fetch_add(&st_serial, 1);
    enter_region(solo_team(), 0, 0);
    invoke_microtask(fn, 0, argc, args);
    team_barrier(tl.team);
    leave_region();
}
static void fork_team(void *fn, int argc, void **args) {
    rt_ensure();
    int nth = tl_push_nt > 0 ? tl_push_nt : atomic_load(&g_maxthreads);
    tl_push_nt = 0;
    if (nth > MAXTHREADS) nth = MAXTHREADS;
    int zero = 0;
    if (nth <= 1 || tl.active > 0 || !atomic_compare_exchange_strong(&g_busy, &zero, 1)) { run_serialized(fn, argc, args); return; }
    ensure_workers(nth - 1);
    if (g_nworkers < nth - 1) nth = g_nworkers + 1;
    Team *tm = &g_team;
    /* the previous region's threads may still be on their way out of its final barrier: they
     * must not see this region's data (or take its tasks) with their old thread numbers */
    { Backoff b = { 0, 0 }; while (atomic_load_explicit(&tm->inbar, memory_order_acquire) > 0) backoff(&b); }
    atomic_store_explicit(&g_oversub, nth > g_nprocs, memory_order_relaxed);
    tm->nth = nth; atomic_store(&tm->bar_nth, nth); tm->fn = fn; tm->argc = argc;
    for (int i = 0; i < argc; i++) tm->args[i] = args[i];
    atomic_store(&tm->single_seq, 0);
    for (int i = 0; i < RING; i++) { tm->disp[i].gen = -1; atomic_store(&tm->disp[i].done, 0); }
    atomic_store(&tm->outstanding, 0);
    if (g_stats) { tm->t_fork = now_us(); atomic_fetch_add(&st_forks, 1); }
    pthread_mutex_lock(&g_mx);
    atomic_fetch_add_explicit(&g_gen, 1, memory_order_acq_rel);
    pthread_cond_broadcast(&g_cv);
    pthread_mutex_unlock(&g_mx);
    enter_region(tm, 0, 1);
    invoke_microtask(fn, 0, argc, args);
    team_barrier(tm);
    leave_region();
    atomic_store(&g_busy, 0);
}

/* ---- entry points: parallel -------------------------------------------------------------- */
void __kmpc_fork_call(ident_t *loc, kmp_int32 argc, void *fn, ...) {
    void *a[MAXARGS];
    if (argc > MAXARGS) { fprintf(stderr, "brisk_omp: %d captured variables (at most %d)\n", argc, MAXARGS); abort(); }
    va_list ap; va_start(ap, fn);
    for (int i = 0; i < argc; i++) a[i] = va_arg(ap, void *);
    va_end(ap);
    fork_team(fn, argc, a);
}
void __kmpc_fork_call_if(ident_t *loc, kmp_int32 argc, void *fn, kmp_int32 cond, void *args) {
    void *a[1] = { args };
    if (cond) fork_team(fn, argc > 0 ? 1 : 0, a);
    else { rt_ensure(); run_serialized(fn, argc > 0 ? 1 : 0, a); }
}
void __kmpc_serialized_parallel(ident_t *loc, kmp_int32 gtid) { rt_ensure(); enter_region(solo_team(), 0, 0); }
void __kmpc_end_serialized_parallel(ident_t *loc, kmp_int32 gtid) { team_barrier(tl.team); leave_region(); }
void __kmpc_push_num_threads(ident_t *loc, kmp_int32 gtid, kmp_int32 n) { tl_push_nt = n; }
kmp_int32 __kmpc_global_thread_num(ident_t *loc) { return tl.team ? tl.tid : 0; }
kmp_int32 __kmpc_ok_to_fork(ident_t *loc) { return 1; }
void __kmpc_begin(ident_t *loc, kmp_int32 flags) { rt_ensure(); }
void __kmpc_end(ident_t *loc) {}
void __kmpc_flush(ident_t *loc) { atomic_thread_fence(memory_order_seq_cst); }
void __kmpc_barrier(ident_t *loc, kmp_int32 gtid) { if (tl.team) team_barrier(tl.team); }

/* ---- single, master, critical, reductions ------------------------------------------------ */
kmp_int32 __kmpc_single(ident_t *loc, kmp_int32 gtid) {
    if (!tl.team) return 1;
    const long mine = ++tl.single;
    long exp = mine - 1;
    return atomic_compare_exchange_strong(&tl.team->single_seq, &exp, mine) ? 1 : 0;
}
void __kmpc_end_single(ident_t *loc, kmp_int32 gtid) {}
kmp_int32 __kmpc_master(ident_t *loc, kmp_int32 gtid) { return tl.tid == 0; }
void __kmpc_end_master(ident_t *loc, kmp_int32 gtid) {}
kmp_int32 __kmpc_masked(ident_t *loc, kmp_int32 gtid, kmp_int32 filter) { return tl.tid == filter; }
void __kmpc_end_masked(ident_t *loc, kmp_int32 gtid) {}

/* critical sections: one mutex per name, found by the address of the name */
#define NCRIT 64
static void *g_crit_key[NCRIT]; static pthread_mutex_t g_crit_mx[NCRIT]; static int g_ncrit = 0;
static pthread_mutex_t g_crit_tab = PTHREAD_MUTEX_INITIALIZER;
static pthread_mutex_t *crit_lock(void *key) {
    pthread_mutex_lock(&g_crit_tab);
    int i; for (i = 0; i < g_ncrit; i++) if (g_crit_key[i] == key) break;
    if (i == g_ncrit) {
        if (g_ncrit == NCRIT) { fprintf(stderr, "brisk_omp: more than %d named critical sections\n", NCRIT); abort(); }
        g_crit_key[i] = key; pthread_mutex_init(&g_crit_mx[i], NULL); g_ncrit++;
    }
    pthread_mutex_unlock(&g_crit_tab);
    return &g_crit_mx[i];
}
void __kmpc_critical(ident_t *loc, kmp_int32 gtid, kmp_critical_name *crit) { pthread_mutex_lock(crit_lock(crit)); }
void __kmpc_critical_with_hint(ident_t *loc, kmp_int32 gtid, kmp_critical_name *crit, uint32_t hint) { pthread_mutex_lock(crit_lock(crit)); }
void __kmpc_end_critical(ident_t *loc, kmp_int32 gtid, kmp_critical_name *crit) { pthread_mutex_unlock(crit_lock(crit)); }
/* reductions: every thread combines its private copy into the shared one under a lock
 * (return value 1: "this thread does the reduction", libomp's critical method) */
kmp_int32 __kmpc_reduce_nowait(ident_t *loc, kmp_int32 gtid, kmp_int32 nvars, size_t size, void *data, void (*fn)(void *, void *), kmp_critical_name *lck) {
    pthread_mutex_lock(&g_red_mx); return 1;
}
void __kmpc_end_reduce_nowait(ident_t *loc, kmp_int32 gtid, kmp_critical_name *lck) { pthread_mutex_unlock(&g_red_mx); }
kmp_int32 __kmpc_reduce(ident_t *loc, kmp_int32 gtid, kmp_int32 nvars, size_t size, void *data, void (*fn)(void *, void *), kmp_critical_name *lck) {
    pthread_mutex_lock(&g_red_mx); return 1;
}
void __kmpc_end_reduce(ident_t *loc, kmp_int32 gtid, kmp_critical_name *lck) { pthread_mutex_unlock(&g_red_mx); if (tl.team) team_barrier(tl.team); }

/* ---- tasks ------------------------------------------------------------------------------- */
/* task blocks: a per-thread free list per size class (64-byte steps up to 1 KB), so that the
 * thousands of small tasks of a sparse triangular solve do not go through malloc */
#define NCLASS 16
typedef struct Blk { struct Blk *next; } Blk;
static __thread Blk *tl_free[NCLASS];
static __thread int tl_nfree[NCLASS];
static void *blk_alloc(size_t size, int *cls) {
    const int c = (int)((size + 63) / 64) - 1;
    if (c < NCLASS) {
        *cls = c;
        Blk *b = tl_free[c];
        if (b) { tl_free[c] = b->next; tl_nfree[c]--; memset(b, 0, (size_t)(c + 1) * 64); return b; }
        return calloc(1, (size_t)(c + 1) * 64);
    }
    *cls = -1;
    return calloc(1, size);
}
static void blk_free(void *p, int cls) {
    if (cls >= 0 && tl_nfree[cls] < 4096) { Blk *b = p; b->next = tl_free[cls]; tl_free[cls] = b; tl_nfree[cls]++; }
    else free(p);
}
kmp_task_t *__kmpc_omp_task_alloc(ident_t *loc, kmp_int32 gtid, kmp_int32 flags, size_t sizeof_task, size_t sizeof_shareds, kmp_routine_entry_t entry) {
    const size_t kt = (sizeof_task + 15) & ~(size_t)15;
    int cls;
    char *p = blk_alloc(HDR_SIZE + kt + sizeof_shareds + 16, &cls);
    if (!p) { fprintf(stderr, "brisk_omp: out of memory (task)\n"); abort(); }
    Task *t = (Task *)p;
    t->cls = cls;
    kmp_task_t *k = (kmp_task_t *)(p + HDR_SIZE);
    k->shareds = sizeof_shareds ? (void *)(p + HDR_SIZE + kt) : NULL;
    k->routine = entry; k->part_id = 0;
    t->routine = entry; t->kt = k;
    atomic_store(&t->refs, 1); atomic_store(&t->children, 0);
    return k;
}
static Task *task_of(kmp_task_t *k) { return (Task *)((char *)k - HDR_SIZE); }
static Task *current_task(void) { return tl.task ? tl.task : (tl.impl ? tl.impl : &tl_impl0); }
kmp_int32 __kmpc_omp_task(ident_t *loc, kmp_int32 gtid, kmp_task_t *k) {
    Task *t = task_of(k), *par = current_task();
    t->parent = par; t->creator = tl.tid;
    atomic_fetch_add(&par->children, 1);
    if (!par->implicit) atomic_fetch_add(&par->refs, 1);
    Team *tm = tl.team;
    if (!tm || tm->nth == 1) {             /* nobody to share with: run it now */
        if (tm) atomic_fetch_add(&tm->outstanding, 1);
        run_task(t);
        return 0;
    }
    atomic_fetch_add(&tm->outstanding, 1);
    if (g_stats) atomic_fetch_add(&st_tasks, 1);
    dq_push(tm, tl.tid, t);
    return 0;
}
kmp_int32 __kmpc_omp_taskwait(ident_t *loc, kmp_int32 gtid) {
    Task *cur = current_task();
    Team *tm = tl.team;
    Backoff b = { 0, 0 };
    while (atomic_load_explicit(&cur->children, memory_order_acquire) > 0) {
        Task *t = tm ? q_pop(tm, cur) : NULL;
        if (t) { run_task(t); b.n = 0; } else backoff(&b);
    }
    return 0;
}
/* if(0) tasks: the compiler runs the body itself between these calls */
void __kmpc_omp_task_begin_if0(ident_t *loc, kmp_int32 gtid, kmp_task_t *k) {
    Task *t = task_of(k), *par = current_task();
    t->parent = par; atomic_fetch_add(&par->children, 1);
    if (!par->implicit) atomic_fetch_add(&par->refs, 1);
    if (tl.team) atomic_fetch_add(&tl.team->outstanding, 1);
    tl.task = t;
}
void __kmpc_omp_task_complete_if0(ident_t *loc, kmp_int32 gtid, kmp_task_t *k) {
    Task *t = task_of(k), *par = t->parent;
    tl.task = par->implicit ? NULL : par;
    atomic_fetch_sub(&par->children, 1); task_release(par); task_release(t);
    if (tl.team) atomic_fetch_sub(&tl.team->outstanding, 1);
}
kmp_int32 __kmpc_omp_taskyield(ident_t *loc, kmp_int32 gtid, int end_part) { return 0; }

/* ---- worksharing loops --------------------------------------------------------------------
 * schedule codes (kmp_sched_t): 33 static chunked, 34 static, 35 dynamic, 36 guided,
 * 37 runtime, 38 auto, 45/46/47 balanced/simd variants; +32 ordered; bits 29/30 modifiers */
static int sched_base(int s) { s &= ~((1 << 29) | (1 << 30)); if (s >= 65 && s <= 79) s -= 32; if (s == 37 || s == 38 || s == 47) s = 35; if (s == 46) s = 36; if (s == 45) s = 34; return s; }
static inline int team_nth(void) { return tl.team ? tl.team->nth : 1; }

#define STATIC_INIT(NAME, T, ST)                                                                     \
void NAME(ident_t *loc, kmp_int32 gtid, kmp_int32 schedtype, kmp_int32 *plast, T *plower, T *pupper, \
          ST *pstride, ST incr, ST chunk) {                                                            \
    const int nth = team_nth(), tid = tl.tid, s = sched_base(schedtype);                              \
    const T lo = *plower, hi = *pupper;                                                               \
    uint64_t trip;                                                                                    \
    if (incr > 0) trip = hi >= lo ? (uint64_t)(hi - lo) / (uint64_t)incr + 1 : 0;                     \
    else trip = lo >= hi ? (uint64_t)(lo - hi) / (uint64_t)(-(int64_t)incr) + 1 : 0;                  \
    if (plast) *plast = 0;                                                                            \
    if (trip == 0) { *pstride = incr; return; }                                                       \
    if (nth == 1) { if (plast) *plast = 1; *pstride = incr > 0 ? (ST)(hi - lo + 1) : (ST)(-(int64_t)(lo - hi + 1)); return; } \
    if (s == 33 && chunk > 0) {                                                                       \
        const uint64_t c = (uint64_t)chunk;                                                           \
        *plower = (T)(lo + (T)((int64_t)c * tid * incr));                                             \
        *pupper = (T)(*plower + (T)((int64_t)(c - 1) * incr));                                        \
        *pstride = (ST)((int64_t)c * nth * incr);                                                     \
        if (plast) *plast = (int)(((trip - 1) / c) % (uint64_t)nth) == tid;                           \
        return;                                                                                       \
    }                                                                                                 \
    /* balanced static: tid < extra gets small + 1 */                                                 \
    const uint64_t small = trip / (uint64_t)nth, extra = trip % (uint64_t)nth;                        \
    const uint64_t first = (uint64_t)tid * small + ((uint64_t)tid < extra ? (uint64_t)tid : extra);   \
    const uint64_t cnt = small + ((uint64_t)tid < extra ? 1 : 0);                                     \
    if (cnt == 0) { *plower = (T)(hi + incr); *pupper = hi; *pstride = incr; return; }                \
    *plower = (T)(lo + (T)((int64_t)first * incr));                                                   \
    *pupper = (T)(*plower + (T)((int64_t)(cnt - 1) * incr));                                          \
    *pstride = incr > 0 ? (ST)(hi - lo + 1) : (ST)(-(int64_t)(lo - hi + 1));                                \
    if (plast) *plast = first + cnt == trip;                                                          \
}
STATIC_INIT(__kmpc_for_static_init_4, kmp_int32, kmp_int32)
STATIC_INIT(__kmpc_for_static_init_4u, kmp_uint32, kmp_int32)
STATIC_INIT(__kmpc_for_static_init_8, kmp_int64, kmp_int64)
STATIC_INIT(__kmpc_for_static_init_8u, kmp_uint64, kmp_int64)
void __kmpc_for_static_fini(ident_t *loc, kmp_int32 gtid) {}

static void disp_init(int sched, uint64_t lb_bits, uint64_t trip, int64_t st, int64_t chunk) {
    Team *tm = tl.team;
    if (!tm) { rt_ensure(); fprintf(stderr, "brisk_omp: worksharing loop outside a parallel region\n"); abort(); }
    const long seq = tl.loopseq++;
    Disp *d = &tm->disp[seq % RING];
    Backoff b = { 0, 0 };
    for (;;) {
        while (atomic_flag_test_and_set_explicit(&d->lock, memory_order_acquire)) cpu_relax();
        if (d->gen == seq) break;                                /* someone initialised it */
        if (d->gen < 0 || atomic_load(&d->done) >= tm->nth) {    /* free, or its previous loop is over */
            d->gen = seq; atomic_store(&d->done, 0); atomic_store(&d->next, 0);
            d->trip = trip; d->sched = sched; d->lb = lb_bits; d->st = st;
            d->chunk = chunk > 0 ? (uint64_t)chunk : 1;
            break;
        }
        atomic_flag_clear_explicit(&d->lock, memory_order_release);
        backoff(&b);
    }
    atomic_flag_clear_explicit(&d->lock, memory_order_release);
    tl.d = d; tl.d_k = 0; tl.d_last_done = 0;
}
/* next chunk as iteration indices [i, j]; 0 when the loop is exhausted for this thread */
static int disp_next(uint64_t *pi, uint64_t *pj) {
    Disp *d = tl.d; const int nth = team_nth(), tid = tl.tid;
    uint64_t i, n;
    if (!d) return 0;
    const uint64_t trip = d->trip;
    if (d->sched == 34 || d->sched == 33) {
        if (d->sched == 34) {                                    /* one balanced block per thread */
            if (tl.d_k++ > 0) goto done;
            const uint64_t small = trip / nth, extra = trip % nth;
            i = (uint64_t)tid * small + ((uint64_t)tid < extra ? (uint64_t)tid : extra);
            n = small + ((uint64_t)tid < extra ? 1 : 0);
            if (n == 0) goto done;
        } else {                                                 /* chunks tid, tid + nth, ... */
            i = (tl.d_k++ * nth + tid) * d->chunk;
            if (i >= trip) goto done;
            n = d->chunk;
        }
    } else if (d->sched == 36) {                                 /* guided */
        uint64_t cur = atomic_load(&d->next);
        for (;;) {
            if (cur >= trip) goto done;
            uint64_t c = (trip - cur) / (2 * (uint64_t)nth);
            if (c < d->chunk) c = d->chunk;
            if (atomic_compare_exchange_weak(&d->next, &cur, cur + c)) { i = cur; n = c; break; }
        }
    } else {                                                     /* dynamic */
        i = atomic_fetch_add(&d->next, d->chunk);
        if (i >= trip) goto done;
        n = d->chunk;
    }
    if (i + n > trip) n = trip - i;
    *pi = i; *pj = i + n - 1;
    return 1;
done:
    if (!tl.d_last_done) { tl.d_last_done = 1; atomic_fetch_add(&d->done, 1); }
    tl.d = NULL;
    return 0;
}
#define DISPATCH(SUF, T, ST)                                                                         \
void __kmpc_dispatch_init_##SUF(ident_t *loc, kmp_int32 gtid, int sched, T lb, T ub, ST st, ST chunk) { \
    uint64_t trip;                                                                                    \
    if (st > 0) trip = ub >= lb ? (uint64_t)(ub - lb) / (uint64_t)st + 1 : 0;                         \
    else trip = lb >= ub ? (uint64_t)(lb - ub) / (uint64_t)(-(int64_t)st) + 1 : 0;                    \
    disp_init(sched_base(sched), (uint64_t)lb, trip, (int64_t)st, (int64_t)chunk);                    \
}                                                                                                     \
int __kmpc_dispatch_next_##SUF(ident_t *loc, kmp_int32 gtid, kmp_int32 *plast, T *plb, T *pub, ST *pst) { \
    Disp *d = tl.d; uint64_t i, j;                                                                    \
    if (!disp_next(&i, &j)) return 0;                                                                 \
    const T lb0 = (T)d->lb; const int64_t st = d->st;                                                 \
    *plb = (T)(lb0 + (T)((int64_t)i * st)); *pub = (T)(lb0 + (T)((int64_t)j * st));                   \
    if (pst) *pst = (ST)st;                                                                           \
    if (plast) *plast = j + 1 == d->trip;                                                             \
    return 1;                                                                                         \
}                                                                                                     \
void __kmpc_dispatch_fini_##SUF(ident_t *loc, kmp_int32 gtid) {}
DISPATCH(4, kmp_int32, kmp_int32)
DISPATCH(4u, kmp_uint32, kmp_int32)
DISPATCH(8, kmp_int64, kmp_int64)
DISPATCH(8u, kmp_uint64, kmp_int64)
void __kmpc_dispatch_deinit(ident_t *loc, kmp_int32 gtid) {}

/* ---- the omp_* API ------------------------------------------------------------------------ */
int omp_get_thread_num(void) { return tl.team ? tl.tid : 0; }
int omp_get_num_threads(void) { return tl.team ? tl.team->nth : 1; }
int omp_get_max_threads(void) { rt_ensure(); return atomic_load(&g_maxthreads); }   /* the nthreads ICV, also inside a region (as libomp) */
void omp_set_num_threads(int n) { rt_ensure(); if (n < 1) n = 1; if (n > MAXTHREADS) n = MAXTHREADS; atomic_store(&g_maxthreads, n); }
int omp_in_parallel(void) { return tl.active > 0; }
int omp_get_num_procs(void) { rt_ensure(); return g_nprocs; }
int omp_get_level(void) { return tl.level; }
void omp_set_dynamic(int d) { g_dynamic = d != 0; }
int omp_get_dynamic(void) { return g_dynamic; }
void omp_set_nested(int n) { g_max_levels = n ? 2 : 1; }
int omp_get_nested(void) { return g_max_levels > 1; }
void omp_set_max_active_levels(int n) { g_max_levels = n > 0 ? n : 1; }
int omp_get_max_active_levels(void) { return g_max_levels; }
double omp_get_wtime(void) { return now_us() * 1e-6; }
double omp_get_wtick(void) { return 1e-9; }

#pragma GCC visibility pop
