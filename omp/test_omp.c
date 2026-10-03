/* test_omp.c - stress test of the bundled OpenMP runtime (brisk_omp.c) against serial results.
 *   clang -O2 -Xpreprocessor -fopenmp -Iomp omp/test_omp.c omp/brisk_omp.c -lpthread -o test_omp  (no libomp; make omptest)
 *   ./test_omp [repetitions]           exits 0 when every check passes, for 1..8 threads */
#include <omp.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int fails = 0;
#define CHECK(c, ...) do { if (!(c)) { fails++; if (fails < 20) { printf("FAIL %s:%d: ", __FILE__, __LINE__); printf(__VA_ARGS__); printf("\n"); } } } while (0)

static long fib_tasks(int n) {
    if (n < 2) return n;
    long a, b;
    if (n < 12) return fib_tasks(n - 1) + fib_tasks(n - 2);
    #pragma omp task shared(a) firstprivate(n)
    a = fib_tasks(n - 1);
    #pragma omp task shared(b) firstprivate(n)
    b = fib_tasks(n - 2);
    #pragma omp taskwait
    return a + b;
}
static long fib(int n) { return n < 2 ? n : fib(n - 1) + fib(n - 2); }

/* BRISK's pattern: a tree whose nodes run when their last child finishes (atomic counters) */
#define TN 2000
static int tpar[TN], tcnt[TN], tdone[TN];
static atomic_int tvisits;
static void tnode(int k) {
    for (;;) {
        tdone[k]++;
        atomic_fetch_add(&tvisits, 1);
        const int p = tpar[k];
        if (p < 0) return;
        int left;
        #pragma omp atomic capture
        left = --tcnt[p];
        if (left != 0) return;
        k = p;
    }
}

int main(int argc, char **argv) {
    const int reps = argc > 1 ? atoi(argv[1]) : 50;
    const int N = 100003;
    double *x = malloc(sizeof(double) * N);
    int *cnt = malloc(sizeof(int) * N);
    for (int i = 0; i < N; i++) x[i] = (i % 97) * 0.5 - 7.0;
    double sref = 0, mref = 1e300; for (int i = 0; i < N; i++) { sref += x[i]; if (x[i] < mref) mref = x[i]; }
    const int nt0 = getenv("NT") ? atoi(getenv("NT")) : 1, nt1 = getenv("NT") ? atoi(getenv("NT")) : 8;
    for (int nt = nt0; nt <= nt1; nt++) {
        omp_set_num_threads(nt);
        CHECK(omp_get_max_threads() == nt, "max threads %d != %d", omp_get_max_threads(), nt);
        for (int r = 0; r < reps; r++) {
            if (getenv("TRACE")) { printf("nt %d rep %d\n", nt, r); fflush(stdout); }
            /* 1. static reduction, int loop */
            double s = 0;
            #pragma omp parallel for schedule(static) reduction(+:s)
            for (int i = 0; i < N; i++) s += x[i];
            CHECK(s == sref || (s - sref) * (s - sref) < 1e-12 * sref * sref, "static sum %g %g", s, sref);
            /* 2. dynamic, size_t loop: every index exactly once */
            memset(cnt, 0, sizeof(int) * N);
            #pragma omp parallel for schedule(dynamic, 1)
            for (size_t i = 0; i < (size_t)N; i++) {
                #pragma omp atomic
                cnt[i]++;
            }
            int bad = 0; for (int i = 0; i < N; i++) bad += cnt[i] != 1;
            CHECK(bad == 0, "dynamic,1 size_t: %d wrong", bad);
            /* 3. static chunked, decreasing loop */
            memset(cnt, 0, sizeof(int) * N);
            #pragma omp parallel for schedule(static, 7)
            for (int i = N - 1; i >= 0; i--) cnt[i] += 1;
            bad = 0; for (int i = 0; i < N; i++) bad += cnt[i] != 1;
            CHECK(bad == 0, "static,7 decreasing: %d wrong", bad);
            /* 4. dynamic chunked, unsigned loop, min reduction */
            double mn = 1e300;
            #pragma omp parallel for schedule(dynamic, 64) reduction(min:mn)
            for (unsigned i = 3; i < (unsigned)N; i += 2) if (x[i] < mn) mn = x[i];
            double mn_ref = 1e300; for (unsigned i = 3; i < (unsigned)N; i += 2) if (x[i] < mn_ref) mn_ref = x[i];
            CHECK(mn == mn_ref, "dynamic,64 unsigned min %g %g", mn, mn_ref);
            /* 5. a region with a sequence of worksharing loops, nowait included (dispatch ring) */
            memset(cnt, 0, sizeof(int) * N);
            #pragma omp parallel
            {
                for (int L = 0; L < 20; L++) {
                    #pragma omp for schedule(dynamic, 5) nowait
                    for (int i = 0; i < 1000; i++) {
                        #pragma omp atomic
                        cnt[L * 1000 + i]++;
                    }
                }
                #pragma omp barrier
                #pragma omp for schedule(guided)
                for (int i = 20000; i < 30000; i++) cnt[i]++;
                #pragma omp for schedule(runtime)
                for (long i = 30000; i < 40000; i++) cnt[i]++;
                int tid = omp_get_thread_num(), n = omp_get_num_threads();
                if (tid < 0 || tid >= n || n != nt) {
                    #pragma omp atomic
                    cnt[N - 1] += 1000;
                }
            }
            bad = 0; for (int i = 0; i < 40000; i++) bad += cnt[i] != 1;
            CHECK(bad == 0 && cnt[N - 1] == 0, "loop sequence: %d wrong, ids %d", bad, cnt[N - 1]);
            /* 6. critical + atomic capture + single */
            long crit = 0; int cap = 0, singles = 0;
            #pragma omp parallel
            {
                for (int k = 0; k < 100; k++) {
                    #pragma omp critical
                    crit += k;
                    int v;
                    #pragma omp atomic capture
                    v = ++cap;
                    (void)v;
                }
                #pragma omp single
                singles++;
                #pragma omp single
                singles++;
            }
            CHECK(crit == 4950L * nt && cap == 100 * nt && singles == 2, "critical %ld atomic %d singles %d", crit, cap, singles);
            /* 7. tasks: recursive, taskwait children */
            long f = 0;
            #pragma omp parallel
            #pragma omp single
            f = fib_tasks(22);
            CHECK(f == fib(22), "task fib %ld", f);
            /* 8. BRISK's counter tree with tasks from a single */
            for (int k = 0; k < TN; k++) { tpar[k] = k == TN - 1 ? -1 : (k / 3 + TN / 2 > k ? k / 3 + TN / 2 : TN - 1); tcnt[k] = 0; tdone[k] = 0; }
            for (int k = 0; k < TN; k++) if (tpar[k] >= 0) tcnt[tpar[k]]++;
            static char leaf[TN]; for (int k = 0; k < TN; k++) leaf[k] = tcnt[k] == 0;
            atomic_store(&tvisits, 0);
            #pragma omp parallel
            #pragma omp single
            {
                for (int k = 0; k < TN; k++) if (leaf[k]) {
                    const int kk = k;
                    #pragma omp task firstprivate(kk)
                    tnode(kk);
                }
            }
            bad = 0; for (int k = 0; k < TN; k++) bad += tdone[k] != 1;
            CHECK(bad == 0 && atomic_load(&tvisits) == TN, "task tree: %d wrong, %d visits", bad, atomic_load(&tvisits));
            /* 9. nested parallel is serialized; if(0) region */
            int inner_bad = 0;
            #pragma omp parallel for schedule(static) reduction(+:inner_bad)
            for (int i = 0; i < 64; i++) {
                int ok = 1;
                #pragma omp parallel if(!omp_in_parallel())
                { if (omp_get_num_threads() != 1 || omp_get_thread_num() != 0) ok = 0; }
                double s2 = 0;
                #pragma omp parallel for reduction(+:s2)
                for (int j = 0; j < 100; j++) s2 += j;
                if (s2 != 4950 || (nt > 1 && !omp_in_parallel())) ok = 0;
                inner_bad += !ok;
            }
            CHECK(inner_bad == 0, "nested: %d bad", inner_bad);
            double s3 = 0;
            #pragma omp parallel for reduction(+:s3) if(0)
            for (int i = 0; i < N; i++) s3 += x[i];
            CHECK(s3 == sref, "if(0) sum");
            /* 10. many captured variables */
            double a0 = 1, a1 = 2, a2 = 3, a3 = 4, a4 = 5, a5 = 6, a6 = 7, a7 = 8, a8 = 9, a9 = 10, b0 = 11, b1 = 12, b2 = 13, b3 = 14, b4 = 15, b5 = 16, b6 = 17, b7 = 18, b8 = 19, b9 = 20;
            double tot = 0;
            #pragma omp parallel for reduction(+:tot)
            for (int i = 0; i < 1000; i++) tot += a0 + a1 + a2 + a3 + a4 + a5 + a6 + a7 + a8 + a9 + b0 + b1 + b2 + b3 + b4 + b5 + b6 + b7 + b8 + b9 + i;
            CHECK(tot == 1000 * 210.0 + 499500.0, "captures %g", tot);
            /* 11. tasks created by every thread inside a worksharing loop, each with taskwait */
            atomic_int tk = 0;
            #pragma omp parallel for schedule(dynamic)
            for (int i = 0; i < 200; i++) {
                for (int j = 0; j < 5; j++) {
                    #pragma omp task
                    atomic_fetch_add(&tk, 1);
                }
                #pragma omp taskwait
            }
            CHECK(atomic_load(&tk) == 1000, "tasks in loop %d", atomic_load(&tk));
            /* 12. empty and tiny loops */
            int z = 0;
            #pragma omp parallel for reduction(+:z) schedule(dynamic, 3)
            for (int i = 0; i < 2; i++) z += 1;
            #pragma omp parallel for reduction(+:z)
            for (int i = 5; i < 5; i++) z += 100;
            CHECK(z == 2, "tiny loops %d", z);
        }
    }
    printf("%s (%d failures)\n", fails ? "FAILED" : "all checks passed", fails);
    free(x); free(cnt);
    return fails != 0;
}
