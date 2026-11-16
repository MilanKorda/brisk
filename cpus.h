/* cpus.h - the CPUs a BRISK process can keep busy, for every program that sets threads (8 November 2026, the SOCP developer:
 * moved here unchanged from solver.c, the LP developer's code of 31 October and 8 November 2026, so that the standalone QP
 * program brisk_qp, which does not link solver.c, applies the same cap; solver.c keeps brisk_cpu_quota and brisk_avail_cpus
 * of brisk.h on top of these). Static functions: each program that includes it has its own copy of the compiled code, and
 * the source is in one place. Include it after _GNU_SOURCE (sched_getaffinity, CPU_COUNT) and <omp.h>. */
#ifndef BRISK_CPUS_H
#define BRISK_CPUS_H
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#ifdef __linux__
#include <sched.h>
#endif

/* (31 October 2026, the LP developer; PLAN_lp_1031.md P1.5) the CPUs the container's quota allows, rounded up, or 0 when
 * there is no quota (or it cannot be read): cgroup v2's cpu.max ("quota period", or "max") of the process's cgroup and of
 * each cgroup above it, the smallest; else cgroup v1's cpu.cfs_quota_us / cpu.cfs_period_us. The runtime counts the CPUs
 * the process may run on (the affinity mask), not the time the quota lets it run: in the owner's independent benchmarks
 * of 29 October (5 CPUs visible, a quota of 4) BRISK ran 5 busy threads on 4 CPUs' worth of time and was throttled in 15 %
 * of the periods; Docker's --cpus, Kubernetes' CPU limits and cloud notebooks set such quotas. Rounded up as the Java and
 * .NET runtimes do (a quota of 3.5 CPUs: 4 threads) */
static double cpus_cg_ratio(const char *qf, const char *pf) {
    FILE *f = fopen(qf, "r"); if (!f) return 0;
    char b[64] = ""; const int ok = fgets(b, sizeof b, f) != NULL; fclose(f);
    if (!ok) return 0;
    double q = 0, per = 0;
    if (pf) {                                       /* (v1: two files, a quota of -1 is none) */
        q = atof(b);
        FILE *g = fopen(pf, "r"); if (!g) return 0;
        char c[64] = ""; const int ok2 = fgets(c, sizeof c, g) != NULL; fclose(g);
        if (!ok2) return 0;
        per = atof(c);
    } else {                                        /* (v2: "max 100000" or "400000 100000") */
        if (!strncmp(b, "max", 3)) return 0;
        if (sscanf(b, "%lf %lf", &q, &per) != 2) return 0;
    }
    return q > 0 && per > 0 ? q / per : 0;
}
static int cpus_quota(void) {
#ifdef __linux__
    static int cached = -1;
    if (cached >= 0) return cached;
    double best = 0;
    FILE *f = fopen("/proc/self/cgroup", "r");
    if (f) {
        char line[4096];
        while (fgets(line, sizeof line, f)) {
            line[strcspn(line, "\n")] = 0;
            if (!strncmp(line, "0::", 3)) {          /* (v2: the unified hierarchy, from the process's cgroup up) */
                char path[4096]; snprintf(path, sizeof path, "%s", line + 3);
                for (;;) {
                    char fn[4200]; snprintf(fn, sizeof fn, "/sys/fs/cgroup%s/cpu.max", strcmp(path, "/") ? path : "");
                    const double r = cpus_cg_ratio(fn, NULL);
                    if (r > 0 && (best == 0 || r < best)) best = r;
                    char *sl = strrchr(path, '/');
                    if (!sl || sl == path) { if (strcmp(path, "/")) { snprintf(path, sizeof path, "/"); continue; } break; }
                    *sl = 0;
                }
            } else if (strstr(line, ":cpu,") || strstr(line, ":cpu:") || strstr(line, ",cpu:") || strstr(line, ",cpu,")) {
                const double r = cpus_cg_ratio("/sys/fs/cgroup/cpu/cpu.cfs_quota_us", "/sys/fs/cgroup/cpu/cpu.cfs_period_us");
                const double r2 = r > 0 ? r : cpus_cg_ratio("/sys/fs/cgroup/cpu,cpuacct/cpu.cfs_quota_us", "/sys/fs/cgroup/cpu,cpuacct/cpu.cfs_period_us");
                if (r2 > 0 && (best == 0 || r2 < best)) best = r2;
            }
        }
        fclose(f);
    }
    if (best == 0) { const double r = cpus_cg_ratio("/sys/fs/cgroup/cpu.max", NULL); if (r > 0) best = r; }   /* (a namespace without the line) */
    if (getenv("BRISK_CPUQUOTA")) best = atof(getenv("BRISK_CPUQUOTA"));      /* (tests: a quota as if read) */
    cached = best > 0 ? (int)ceil(best - 1e-9) : 0;
    if (cached < 0) cached = 0;
    return cached;
#else
    return 0;
#endif
}
#ifdef __linux__
/* the cores this process may use: the affinity mask, or with thread binding (which narrows
 * the initial thread's mask to its place) the union of the OpenMP places */
static int cpus_allowed(cpu_set_t *ms) {
    CPU_ZERO(ms);
#if defined(_OPENMP) && _OPENMP >= 201511 && !defined(BRISK_OMP_H)
    if (omp_get_proc_bind() != omp_proc_bind_false) {
        const int np = omp_get_num_places();
        int ids[CPU_SETSIZE];
        for (int i = 0; i < np; i++) {
            const int n = omp_get_place_num_procs(i);
            if (n <= 0 || n > CPU_SETSIZE) continue;
            omp_get_place_proc_ids(i, ids);
            for (int j = 0; j < n; j++) if (ids[j] >= 0 && ids[j] < CPU_SETSIZE) CPU_SET(ids[j], ms);
        }
        if (CPU_COUNT(ms) > 0) return CPU_COUNT(ms);
    }
#endif
    if (sched_getaffinity(0, sizeof *ms, ms) != 0) return -1;
    return CPU_COUNT(ms);
}
#endif
/* (8 November 2026, the LP developer) the CPUs the process can keep busy: those it may run on (the affinity mask), or fewer
 * when the container's quota allows fewer (cpus_quota); 0 when unknown */
static int cpus_avail(void) {
#ifdef __linux__
    cpu_set_t ms; const int n = cpus_allowed(&ms); const int q = cpus_quota();
    return q > 0 && (n <= 0 || q < n) ? q : (n > 0 ? n : 0);
#else
    return 0;
#endif
}
#endif
