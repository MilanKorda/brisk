#!/bin/bash
# The command-line solver with bound OpenMP threads (OMP_PROC_BIND, OMP_PLACES): its threads
# must sit on different processors. (Until 1.3.1 the re-exec that starts OpenBLAS with one
# thread kept the affinity mask of the initial thread, already pinned by the OpenMP runtime, and
# all threads ran on one core.)  Linux only; needs two processors.  usage: tools/bind_test.sh [brisk]
B=${1:-./brisk}; F=examples/theta3.dat-s
[ "$(uname)" = Linux ] || { echo "bind test: skipped (not Linux)"; exit 0; }
[ "$(nproc)" -ge 2 ] || { echo "bind test: skipped (one processor)"; exit 0; }
fail=0
check() {   # name, expected minimum number of distinct processors, environment...
    local name=$1 want=$2; shift 2
    # the masks are read as soon as both threads exist (the solve takes under a second)
    env "$@" "$B" $F -threads 2 -maxit 100000 -tol 1e-30 -q > /dev/null 2>&1 &
    local p=$! nthr=0 masks="" i
    for i in $(seq 1 400); do
        sleep 0.02
        [ -d /proc/$p ] || break
        nthr=$(ls /proc/$p/task 2>/dev/null | wc -l)
        if [ "$nthr" -ge 2 ]; then
            sleep 0.05                                  # (a new thread is bound right after its creation)
            local n2; n2=$(ls /proc/$p/task 2>/dev/null | wc -l); [ "$n2" -ge 2 ] && nthr=$n2
            masks=$(for t in /proc/$p/task/*; do awk '/Cpus_allowed_list/ {print $2}' "$t/status" 2>/dev/null; done | sort | uniq)
            break
        fi
    done
    kill $p 2>/dev/null; wait $p 2>/dev/null
    local single; single=$(echo "$masks" | grep -c -E '^[0-9]+$')
    if [ "$want" = bound ]; then
        if [ "$single" -ge 2 ] && [ "$nthr" -eq 2 ]; then echo "  ok    $name: $nthr threads on $(echo $masks | tr '\n' ' ')"; else echo "  FAIL  $name: $nthr threads, masks: $(echo $masks | tr '\n' ' ')"; fail=1; fi
    else
        if [ "$single" -eq 0 ] && [ "$nthr" -eq 2 ]; then echo "  ok    $name: $nthr threads, not bound ($(echo $masks | tr '\n' ' '))"; else echo "  FAIL  $name: $nthr threads, masks: $(echo $masks | tr '\n' ' ')"; fail=1; fi
    fi
}
echo "bind test ($B):"
check "no binding" free X=1
check "OMP_PROC_BIND=true" bound OMP_PROC_BIND=true
check "OMP_PROC_BIND=close OMP_PLACES=cores" bound OMP_PROC_BIND=close OMP_PLACES=cores
check "OMP_PLACES=threads" bound OMP_PLACES=threads
check "GOMP_CPU_AFFINITY=0-1" bound GOMP_CPU_AFFINITY=0-1
exit $fail
